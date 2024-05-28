from typing import Any
import flax.linen as nn
import jax.numpy as jnp
import functools
import ml_collections
import jax

###########################
### Helper Modules
### https://github.com/google-research/maskgit/blob/main/maskgit/nets/layers.py
###########################

def get_norm_layer(norm_type):
    """Normalization layer."""
    if norm_type == 'BN':
        raise NotImplementedError
    elif norm_type == 'LN':
        norm_fn = functools.partial(nn.LayerNorm)
    elif norm_type == 'GN':
        norm_fn = functools.partial(nn.GroupNorm)
    else:
        raise NotImplementedError
    return norm_fn


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
    pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
    pool_denom = jax.lax.reduce_window(
        jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
        (1,) + strides + (1,), padding)
    return pool_sum / pool_denom

def upsample(x, factor=2):
    n, h, w, c = x.shape
    x = jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')
    return x

def dsample(x):
    return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding='same')

def squared_euclidean_distance(a: jnp.ndarray,
                               b: jnp.ndarray,
                               b2: jnp.ndarray = None) -> jnp.ndarray:
    """Computes the pairwise squared Euclidean distance.

    Args:
        a: float32: (n, d): An array of points.
        b: float32: (m, d): An array of points.
        b2: float32: (d, m): b square transpose.

    Returns:
        d: float32: (n, m): Where d[i, j] is the squared Euclidean distance between
        a[i] and b[j].
    """
    if b2 is None:
        b2 = jnp.sum(b.T**2, axis=0, keepdims=True)
    a2 = jnp.sum(a**2, axis=1, keepdims=True)
    ab = jnp.matmul(a, b.T)
    d = a2 - 2 * ab + b2
    return d

def entropy_loss_fn(affinity, loss_type="softmax", temperature=1.0):
    """Calculates the entropy loss. Affinity is the similarity/distance matrix."""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = jax.nn.softmax(flat_affinity, axis=-1)
    log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = jnp.argmax(flat_affinity, axis=-1)
        onehots = jax.nn.one_hot(
            codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype)
        onehots = probs - jax.lax.stop_gradient(probs - onehots)
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = jnp.mean(target_probs, axis=0)
    avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
    sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss

def sg(x):
    return jax.lax.stop_gradient(x)




###########################
### Modules
###########################

class ResBlock(nn.Module):
    """Basic Residual Block."""
    filters: int
    norm_fn: Any
    activation_fn: Any

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        residual = x
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = nn.Conv(self.filters, kernel_size=(3, 3), use_bias=False)(x)
        x = self.norm_fn()(x)
        x = self.activation_fn(x)
        x = nn.Conv(self.filters, kernel_size=(3, 3), use_bias=False)(x)

        if input_dim != self.filters:
            residual = nn.Conv(self.filters, kernel_size=(1, 1), use_bias=False)(x)
        return x + residual
    
class Encoder(nn.Module):
    """From [H,W,D] image to [H',W',D'] embedding. Using Conv layers."""
    config: ml_collections.ConfigDict

    def setup(self):
        self.filters = self.config.filters
        self.num_res_blocks = self.config.num_res_blocks
        self.channel_multipliers = self.config.channel_multipliers
        self.embedding_dim = self.config.embedding_dim
        self.norm_type = self.config.norm_type
        self.activation_fn = nn.swish

    @nn.compact
    def __call__(self, x):
        print("Initializing encoder.")
        norm_fn = get_norm_layer(norm_type=self.norm_type)
        block_args = dict(norm_fn=norm_fn, activation_fn=self.activation_fn)
        x = nn.Conv(self.filters, kernel_size=(3, 3), use_bias=False)(x)
        print('Encoder layer', x.shape)
        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **block_args)(x)
            if i < num_blocks - 1:
                x = dsample(x)
            print('Encoder layer', x.shape)

        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **block_args)(x)
            print('Encoder layer', x.shape)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        last_dim = self.embedding_dim*2 if self.config['quantizer_type'] == 'kl' else self.embedding_dim
        x = nn.Conv(last_dim, kernel_size=(1, 1))(x)
        print("Final embeddings are size", x.shape)
        return x
    
class Decoder(nn.Module):
    """From [H',W',D'] embedding to [H,W,D] embedding. Using Conv layers."""

    config: ml_collections.ConfigDict

    def setup(self):
        self.filters = self.config.filters
        self.num_res_blocks = self.config.num_res_blocks
        self.channel_multipliers = self.config.channel_multipliers
        self.norm_type = self.config.norm_type
        self.image_channels = self.config.image_channels
        self.activation_fn = nn.swish

    @nn.compact
    def __call__(self, x):
        norm_fn = get_norm_layer(norm_type=self.norm_type)
        block_args = dict(norm_fn=norm_fn, activation_fn=self.activation_fn,)
        num_blocks = len(self.channel_multipliers)
        filters = self.filters * self.channel_multipliers[-1]
        x = nn.Conv(filters, kernel_size=(3, 3), use_bias=True)(x)
        for _ in range(self.num_res_blocks):
            x = ResBlock(filters, **block_args)(x)
            print('Decoder layer', x.shape)
        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                x = ResBlock(filters, **block_args)(x)
            if i > 0:
                x = upsample(x, 2)
                x = nn.Conv(filters, kernel_size=(3, 3))(x)
            print('Decoder layer', x.shape)
        x = norm_fn()(x)
        x = self.activation_fn(x)
        x = nn.Conv(self.image_channels, kernel_size=(3, 3))(x)
        return x
    
class VectorQuantizer(nn.Module):
    """Basic vector quantizer."""
    config: ml_collections.ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x):
        codebook_size = self.config.codebook_size
        emb_dim = x.shape[-1]
        codebook = self.param(
            "codebook",
            jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="uniform"),
            (codebook_size, emb_dim))
        codebook = jnp.asarray(codebook) # (codebook_size, emb_dim)
        distances = jnp.reshape(
            squared_euclidean_distance(jnp.reshape(x, (-1, emb_dim)), codebook),
            x.shape[:-1] + (codebook_size,)) # [x, codebook_size] similarity matrix.
        encoding_indices = jnp.argmin(distances, axis=-1)
        encoding_onehot = jax.nn.one_hot(encoding_indices, codebook_size)
        quantized = self.quantize(encoding_onehot)
        result_dict = dict()
        if self.train:
            e_latent_loss = jnp.mean((sg(quantized) - x)**2) * self.config.commitment_cost
            q_latent_loss = jnp.mean((quantized - sg(x))**2)
            entropy_loss = 0.0
            if self.config.entropy_loss_ratio != 0:
                entropy_loss = entropy_loss_fn(
                    -distances,
                    loss_type=self.config.entropy_loss_type,
                    temperature=self.config.entropy_temperature
                ) * self.config.entropy_loss_ratio
            e_latent_loss = jnp.asarray(e_latent_loss, jnp.float32)
            q_latent_loss = jnp.asarray(q_latent_loss, jnp.float32)
            entropy_loss = jnp.asarray(entropy_loss, jnp.float32)
            loss = e_latent_loss + q_latent_loss + entropy_loss
            result_dict = dict(
                quantizer_loss=loss,
                e_latent_loss=e_latent_loss,
                q_latent_loss=q_latent_loss,
                entropy_loss=entropy_loss)
            quantized = x + jax.lax.stop_gradient(quantized - x)

        result_dict.update({
            "z_ids": encoding_indices,
        })
        return quantized, result_dict

    def quantize(self, encoding_onehot: jnp.ndarray) -> jnp.ndarray:
        codebook = jnp.asarray(self.variables["params"]["codebook"])
        return jnp.dot(encoding_onehot, codebook)

    def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
        codebook = self.variables["params"]["codebook"]
        return jnp.take(codebook, ids, axis=0)

class KLQuantizer(nn.Module):
    config: ml_collections.ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x):
        emb_dim = x.shape[-1] // 2 # Use half as means, half as logvars.
        means = x[..., :emb_dim]
        logvars = x[..., emb_dim:]
        if not self.train:
            result_dict = dict()
            return means, result_dict
        else:
            noise = jax.random.normal(self.make_rng("noise"), means.shape)
            stds = jnp.exp(0.5 * logvars)
            z = means + stds * noise
            kl_loss = -0.5 * jnp.mean(1 + logvars - means**2 - jnp.exp(logvars))
            result_dict = dict(quantizer_loss=kl_loss)
            return z, result_dict
        
class FSQuantizer(nn.Module):
    config: ml_collections.ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x):
        assert self.config['fsq_levels'] % 2 == 1, "FSQ levels must be odd."
        z = jnp.tanh(x) # [-1, 1]
        z = z * (self.config['fsq_levels']-1) / 2 # [-fsq_levels/2, fsq_levels/2]
        zhat = jnp.round(z) # e.g. [-2, -1, 0, 1, 2]
        quantized = z + jax.lax.stop_gradient(zhat - z)
        quantized = quantized / (self.config['fsq_levels'] // 2) # [-1, 1], but quantized.
        result_dict = dict()

        # Diagnostics for codebook usage.
        zhat_scaled = zhat + self.config['fsq_levels'] // 2
        basis = jnp.concatenate((jnp.array([1]), jnp.cumprod(jnp.array([self.config['fsq_levels']] * (x.shape[-1]-1))))).astype(jnp.uint32)
        idx = (zhat_scaled * basis).sum(axis=-1).astype(jnp.uint32)
        idx_flat = idx.reshape(-1)
        usage = jnp.bincount(idx_flat, length=self.config['fsq_levels']**x.shape[-1])

        result_dict.update({
            "z_ids": zhat,
            'usage': usage
        })
        return quantized, result_dict

class VQVAE(nn.Module):
    """VQVAE model."""
    config: ml_collections.ConfigDict
    train: bool

    def setup(self):
        """VQVAE setup."""
        if self.config['quantizer_type'] == 'vq':
            self.quantizer = VectorQuantizer(config=self.config, train=self.train)
        elif self.config['quantizer_type'] == 'kl':
            self.quantizer = KLQuantizer(config=self.config, train=self.train)
        elif self.config['quantizer_type'] == 'fsq':
            self.quantizer = FSQuantizer(config=self.config, train=self.train)
        self.encoder = Encoder(config=self.config)
        self.decoder = Decoder(config=self.config)

    def encode(self, image):
        encoded_feature = self.encoder(image)
        quantized, result_dict = self.quantizer(encoded_feature)
        return quantized, result_dict

    def decode(self, z_vectors):
        reconstructed = self.decoder(z_vectors)
        return reconstructed

    def decode_from_indices(self, z_ids):
        z_vectors = self.quantizer.decode_ids(z_ids)
        reconstructed_image = self.decode(z_vectors)
        return reconstructed_image

    def encode_to_indices(self, image):
        encoded_feature = self.encoder(image)
        _, result_dict = self.quantizer(encoded_feature)
        ids = result_dict["z_ids"]
        return ids

    def __call__(self, input_dict):
        quantized, result_dict = self.encode(input_dict)
        outputs = self.decoder(quantized)
        return outputs, result_dict