from localutils.debugger import enable_debug
enable_debug()

import flax.linen as nn
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import elements
import ml_collections
import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
import matplotlib.pyplot as plt
from typing import Any

from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.fid import get_fid_network, fid_from_stats

from train import VQGANModel
from models.vqvae import VQVAE
from models.discriminator import Discriminator

delattr(flags.FLAGS, 'dataset_name')
delattr(flags.FLAGS, 'load_dir')
delattr(flags.FLAGS, 'batch_size')

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet128', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Load dir (if not None, load params from here).')
flags.DEFINE_integer('batch_size', 256, 'Total Batch size.')

def main(_):
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)

    def get_dataset(is_train):
        if 'imagenet' in FLAGS.dataset_name:
            def deserialization_fn(data):
                image = data['image']
                min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
                image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
                if 'imagenet256' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (256, 256))
                elif 'imagenet128' in FLAGS.dataset_name:
                    image = tf.image.resize(image, (128, 128))
                else:
                    raise ValueError(f"Unknown dataset {FLAGS.dataset_name}")
                if is_train:
                    image = tf.image.random_flip_left_right(image)
                image = tf.cast(image, tf.float32) / 255.0
                return image

            split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
            dataset = tfds.load('imagenet2012', split=split, data_dir='gs://rll-tpus-kvfrans/tfds')
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
            dataset = dataset.batch(local_batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            dataset = tfds.as_numpy(dataset)
            dataset = iter(dataset)
            return dataset
        else:
            raise ValueError(f"Unknown dataset {FLAGS.dataset_name}")
    
    dataset = get_dataset(is_train=True)
    dataset_valid = get_dataset(is_train=False)
    example_obs = next(dataset)[:1]

    get_fid_activations = get_fid_network()
    truth_fid_stats = np.load('data/imagenet256_fidstats_openai.npz')

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key = jax.random.split(rng)
    print("Total Memory on device:", float(jax.local_devices()[0].memory_stats()['bytes_limit']) / 1024**3, "GB")

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs.shape[-1]
    FLAGS.model.image_size = example_obs.shape[1]
    vqvae_def = VQVAE(FLAGS.model, train=True)
    vqvae_params = vqvae_def.init({'params': param_key, 'noise': param_key}, example_obs)['params']
    tx = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    vqvae_ts = TrainState.create(vqvae_def, vqvae_params, tx=tx)
    vqvae_def_eps = VQVAE(FLAGS.model, train=False)
    vqvae_eps_ts = TrainState.create(vqvae_def_eps, vqvae_params)
    print("Total num of VQVAE parameters:", sum(x.size for x in jax.tree_util.tree_leaves(vqvae_params)))

    discriminator_def = Discriminator(FLAGS.model)
    discriminator_params = discriminator_def.init(param_key, example_obs)['params']
    tx = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    discriminator_ts = TrainState.create(discriminator_def, discriminator_params, tx=tx)
    print("Total num of Discriminator parameters:", sum(x.size for x in jax.tree_util.tree_leaves(discriminator_params)))

    model = VQGANModel(rng=rng, vqvae=vqvae_ts, vqvae_eps=vqvae_eps_ts, discriminator=discriminator_ts, config=FLAGS.model)

    assert FLAGS.load_dir is not None
    cp = Checkpoint(FLAGS.load_dir)
    model = cp.load_model(model)
    print("Loaded model with step", model.vqvae.step)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    jax.debug.visualize_array_sharding(model.vqvae.params['decoder']['Conv_0']['bias'])

    ###################################
    # FID Evaluation.
    ###################################

    activations = []
    for valid_images in dataset_valid:
        valid_images = next(dataset_valid)
        if valid_images.shape[0] < local_batch_size:
            valid_images = np.concatenate([valid_images, np.zeros((local_batch_size - valid_images.shape[0], *valid_images.shape[1:]))], axis=0)
            zeros_added = local_batch_size - valid_images.shape[0]
        else:
            zeros_added = 0
        
        valid_images = valid_images.reshape((len(jax.local_devices()), -1, *valid_images.shape[1:])) # [devices, batch//devices, etc..]
        valid_reconstructed_images = model.reconstruction(valid_images) # [devices, 8, 256, 256, 3]
        valid_reconstructed_images = jax.image.resize(valid_reconstructed_images, (valid_images.shape[0], valid_images.shape[1], 299, 299, 3),
                                                        method='bilinear', antialias=False)
        valid_reconstructed_images = 2 * valid_reconstructed_images - 1
        acts = np.array(get_fid_activations(valid_reconstructed_images))[..., 0, 0, :]
        if zeros_added > 0:
            acts = acts[:-zeros_added]
        activations.append(acts)
        print(len(activations) * FLAGS.batch_size)
    activations = np.concatenate(activations, axis=0)
    activations = activations.reshape((-1, activations.shape[-1]))
    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)
    fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])

    print("FID:", fid)

if __name__ == '__main__':
    app.run(main)