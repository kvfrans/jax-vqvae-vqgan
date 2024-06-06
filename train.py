try: # For debugging
    from localutils.debugger import enable_debug
    enable_debug()
except ImportError:
    pass

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
import ml_collections
import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
import matplotlib.pyplot as plt
from typing import Any
import os

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.pretrained_resnet import get_pretrained_embs, get_pretrained_model
from utils.fid import get_fid_network, fid_from_stats
from models.vqvae import VQVAE
from models.discriminator import Discriminator

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('save_dir', None, 'Save dir (if not None, save params).')
flags.DEFINE_string('load_dir', None, 'Load dir (if not None, load params from here).')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 200000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Total Batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')

model_config = ml_collections.ConfigDict({
    # VQVAE
    'lr': 0.0001,
    'beta1': 0.0,
    'beta2': 0.99,
    'lr_warmup_steps': 2000,
    'lr_decay_steps': 98_000,
    'filters': 128,
    'num_res_blocks': 2,
    'channel_multipliers': (1, 1, 2, 2, 4),
    'embedding_dim': 256, # For FSQ, a good default is 4.
    'norm_type': 'GN',
    'weight_decay': 0.05,
    'clip_gradient': 1.0,
    'l2_loss_weight': 1.0,
    'eps_update_rate': 0.9999,
    # Quantizer
    'quantizer_type': 'vq', # or 'fsq', 'kl'
    # Quantizer (VQ)
    'quantizer_loss_ratio': 1,
    'codebook_size': 1024,
    'entropy_loss_ratio': 0.1,
    'entropy_loss_type': 'softmax',
    'entropy_temperature': 0.01,
    'commitment_cost': 0.25,
    # Quantizer (FSQ)
    'fsq_levels': 5, # Bins per dimension.
    # Quantizer (KL)
    'kl_weight': 0.001,
    # GAN
    'g_adversarial_loss_weight': 0.1,
    'g_grad_penalty_cost': 10,
    'perceptual_loss_weight': 0.1,
    'gan_warmup_steps': 10000,
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'vqvae',
    'name': 'vqvae_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)

##############################################
## Model Definitions.
##############################################

@jax.vmap
def sigmoid_cross_entropy_with_logits(*, labels: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    """https://github.com/google-research/maskgit/blob/main/maskgit/libml/losses.py
    """
    zeros = jnp.zeros_like(logits, dtype=logits.dtype)
    condition = (logits >= zeros)
    relu_logits = jnp.where(condition, logits, zeros)
    neg_abs_logits = jnp.where(condition, -logits, logits)
    return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))

class VQGANModel(flax.struct.PyTreeNode):
    rng: Any
    config: dict = flax.struct.field(pytree_node=False)
    vqvae: TrainState
    vqvae_eps: TrainState
    discriminator: TrainState

    # Train G and D.
    @partial(jax.pmap, axis_name='data', in_axes=(0, 0))
    def update(self, images, pmap_axis='data'):
        new_rng, curr_key = jax.random.split(self.rng, 2)

        resnet, resnet_params = get_pretrained_model('resnet50', 'data/resnet_pretrained.npy')

        is_gan_training = 1.0 - (self.vqvae.step < self.config['gan_warmup_steps']).astype(jnp.float32)

        def loss_fn(params_vqvae, params_disc):
            # Reconstruct image
            reconstructed_images, result_dict = self.vqvae(images, params=params_vqvae, rngs={'noise': curr_key})
            print("Reconstructed images shape", reconstructed_images.shape)
            print("Input images shape", images.shape)
            assert reconstructed_images.shape == images.shape

            # GAN loss on VQVAE output.
            discriminator_fn = lambda x: self.discriminator(x, params=params_disc)
            real_logit, vjp_fn = jax.vjp(discriminator_fn, images, has_aux=False)
            gradient = vjp_fn(jnp.ones_like(real_logit))[0] # Gradient of discriminator output wrt. real images.
            gradient = gradient.reshape((images.shape[0], -1))
            gradient = jnp.asarray(gradient, jnp.float32)
            penalty = jnp.sum(jnp.square(gradient), axis=-1)
            penalty = jnp.mean(penalty) # Gradient penalty for training D.
            fake_logit = discriminator_fn(reconstructed_images)
            d_loss_real = sigmoid_cross_entropy_with_logits(labels=jnp.ones_like(real_logit), logits=real_logit).mean()
            d_loss_fake = sigmoid_cross_entropy_with_logits(labels=jnp.zeros_like(fake_logit), logits=fake_logit).mean()
            loss_d = d_loss_real + d_loss_fake + (penalty * self.config['g_grad_penalty_cost'])

            d_loss_for_vae = sigmoid_cross_entropy_with_logits(labels=jnp.ones_like(fake_logit), logits=fake_logit).mean()
            d_loss_for_vae = d_loss_for_vae * is_gan_training

            real_pools, _ = get_pretrained_embs(resnet_params, resnet, images=images)
            fake_pools, _ = get_pretrained_embs(resnet_params, resnet, images=reconstructed_images)
            perceptual_loss = jnp.mean((real_pools - fake_pools)**2)

            l2_loss = jnp.mean((reconstructed_images - images) ** 2)
            quantizer_loss = result_dict['quantizer_loss'] if 'quantizer_loss' in result_dict else 0.0
            if self.config['quantizer_type'] == 'kl':
                quantizer_loss = quantizer_loss * self.config['kl_weight']
            loss_vae = (l2_loss * FLAGS.model['l2_loss_weight']) \
                + (quantizer_loss * FLAGS.model['quantizer_loss_ratio']) \
                + (d_loss_for_vae * FLAGS.model['g_adversarial_loss_weight']) \
                + (perceptual_loss * FLAGS.model['perceptual_loss_weight'])
            codebook_usage = result_dict['usage'] if 'usage' in result_dict else 0.0
            return (loss_vae, loss_d), {
                'loss_vae': loss_vae,
                'loss_d': loss_d,
                'l2_loss': l2_loss,
                'd_loss_for_vae': d_loss_for_vae,
                'perceptual_loss': perceptual_loss,
                'quantizer_loss': quantizer_loss,
                'codebook_usage': codebook_usage,
            }
        
        # This is a fancy way to do 'jax.grad' so (loss_vae, params_vqvae) and (loss_d, params_disc) are differentiated.
        _, grad_fn, info = jax.vjp(loss_fn, self.vqvae.params, self.discriminator.params, has_aux=True)
        vae_grads, _ = grad_fn((1., 0.))
        _, d_grads = grad_fn((0., 1.))

        vae_grads = jax.lax.pmean(vae_grads, axis_name=pmap_axis)
        d_grads = jax.lax.pmean(d_grads, axis_name=pmap_axis)
        d_grads = jax.tree_map(lambda x: x * is_gan_training, d_grads)

        info = jax.lax.pmean(info, axis_name=pmap_axis)
        if self.config['quantizer_type'] == 'fsq':
            info['codebook_usage'] = jnp.sum(info['codebook_usage'] > 0) / info['codebook_usage'].shape[-1]

        updates, new_opt_state = self.vqvae.tx.update(vae_grads, self.vqvae.opt_state, self.vqvae.params)
        new_params = optax.apply_updates(self.vqvae.params, updates)
        new_vqvae = self.vqvae.replace(step=self.vqvae.step + 1, params=new_params, opt_state=new_opt_state)

        updates, new_opt_state = self.discriminator.tx.update(d_grads, self.discriminator.opt_state, self.discriminator.params)
        new_params = optax.apply_updates(self.discriminator.params, updates)
        new_discriminator = self.discriminator.replace(step=self.discriminator.step + 1, params=new_params, opt_state=new_opt_state)

        info['grad_norm_vae'] = optax.global_norm(vae_grads)
        info['grad_norm_d'] = optax.global_norm(d_grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['is_gan_training'] = is_gan_training

        new_vqvae_eps = target_update(new_vqvae, self.vqvae_eps, 1-self.config['eps_update_rate'])

        new_model = self.replace(rng=new_rng, vqvae=new_vqvae, vqvae_eps=new_vqvae_eps, discriminator=new_discriminator)
        return new_model, info
    
    @partial(jax.pmap, axis_name='data', in_axes=(0, 0))
    def reconstruction(self, images, pmap_axis='data'):
        reconstructed_images, _ = self.vqvae_eps(images)
        reconstructed_images = jnp.clip(reconstructed_images, 0, 1)
        return reconstructed_images

##############################################
## Training Code.
##############################################
def main(_):
    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Device count", device_count)
    print("Global device count", global_device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

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
            dataset = tfds.load('imagenet2012', split=split)
            dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
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
    if not os.path.exists('data/imagenet256_fidstats_openai.npz'):
        raise ValueError("Please download the FID stats file! See the README.")
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

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.vqvae.step)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    jax.debug.visualize_array_sharding(model.vqvae.params['decoder']['Conv_0']['bias'])

    ###################################
    # Train Loop
    ###################################
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        batch_images = next(dataset)
        batch_images = batch_images.reshape((len(jax.local_devices()), -1, *batch_images.shape[1:])) # [devices, batch//devices, etc..]

        model, update_info = model.update(batch_images)

        if i % FLAGS.log_interval == 0:
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            # Print some images
            reconstructed_images = model.reconstruction(batch_images) # [devices, 8, 256, 256, 3]
            valid_images = next(dataset_valid)
            valid_images = valid_images.reshape((len(jax.local_devices()), -1, *valid_images.shape[1:])) # [devices, batch//devices, etc..]
            valid_reconstructed_images = model.reconstruction(valid_images) # [devices, 8, 256, 256, 3]

            if jax.process_index() == 0:
                wandb.log({'batch_image_mean': batch_images.mean()}, step=i)
                wandb.log({'reconstructed_images_mean': reconstructed_images.mean()}, step=i)
                wandb.log({'batch_image_std': batch_images.std()}, step=i)
                wandb.log({'reconstructed_images_std': reconstructed_images.std()}, step=i)

                # plot comparison witah matplotlib. put each reconstruction side by side.
                fig, axs = plt.subplots(2, 8, figsize=(30, 15))
                for j in range(8):
                    axs[0, j].imshow(batch_images[j, 0], vmin=0, vmax=1)
                    axs[1, j].imshow(reconstructed_images[j, 0], vmin=0, vmax=1)
                wandb.log({'reconstruction': wandb.Image(fig)}, step=i)
                plt.close(fig)
                fig, axs = plt.subplots(2, 8, figsize=(30, 15))
                for j in range(8):
                    axs[0, j].imshow(valid_images[j, 0], vmin=0, vmax=1)
                    axs[1, j].imshow(valid_reconstructed_images[j, 0], vmin=0, vmax=1)
                wandb.log({'reconstruction_valid': wandb.Image(fig)}, step=i)
                plt.close(fig)

            # Validation Losses
            _, valid_update_info = model.update(valid_images)
            valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
            valid_metrics = {f'validation/{k}': v for k, v in valid_update_info.items()}
            if jax.process_index() == 0:
                wandb.log(valid_metrics, step=i)

            # FID measurement.
            activations = []
            for _ in range(64):
                valid_images = next(dataset_valid)
                valid_images = valid_images.reshape((len(jax.local_devices()), -1, *valid_images.shape[1:])) # [devices, batch//devices, etc..]
                valid_reconstructed_images = model.reconstruction(valid_images) # [devices, 8, 256, 256, 3]
                valid_reconstructed_images = jax.image.resize(valid_reconstructed_images, (valid_images.shape[0], valid_images.shape[1], 299, 299, 3),
                                                               method='bilinear', antialias=False)
                valid_reconstructed_images = 2 * valid_reconstructed_images - 1
                activations += [np.array(get_fid_activations(valid_reconstructed_images))[..., 0, 0, :]]
                # TODO: use all_gather to get activations from all devices.
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            if jax.process_index() == 0:
                wandb.log({'validation/fid': fid}, step=i)



        if (i % FLAGS.save_interval == 0) and (FLAGS.save_dir is not None):
            if jax.process_index() == 0:
                model_single = flax.jax_utils.unreplicate(model)
                cp = Checkpoint(FLAGS.save_dir)
                cp.set_model(model_single)
                cp.save()

if __name__ == '__main__':
    app.run(main)