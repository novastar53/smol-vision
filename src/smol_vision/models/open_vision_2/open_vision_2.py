import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.models.open_vision_2.config import Config
from smol_vision.modules.mlp import MLP
from smol_vision.modules.attn import SelfAttention
'''
CLIP(
  (visual): VisionTransformer(
    (conv1): Conv2d(3, 192, kernel_size=(16, 16), stride=(16, 16), bias=False)
    (patch_dropout): Identity()
    (ln_pre): Identity()
    (transformer): Transformer(
      (resblocks): ModuleList(
        (0-11): 12 x ResidualAttentionBlock(
          (ln_1): LayerNorm((192,), eps=1e-06, elementwise_affine=True)
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=192, out_features=192, bias=True)
          )
          (ls_1): Identity()
          (ln_2): LayerNorm((192,), eps=1e-06, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=192, out_features=768, bias=True)
            (gelu): GELU(approximate='none')
            (c_proj): Linear(in_features=768, out_features=192, bias=True)
          )
          (ls_2): Identity()
        )
      )
    )
    (ln_post): LayerNorm((192,), eps=1e-06, elementwise_affine=True)
  )
  (transformer): Transformer(
    (resblocks): ModuleList(
      (0-11): 12 x ResidualAttentionBlock(
        (ln_1): LayerNorm((192,), eps=1e-06, elementwise_affine=True)
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=192, out_features=192, bias=True)
        )
        (ls_1): Identity()
        (ln_2): LayerNorm((192,), eps=1e-06, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=192, out_features=768, bias=True)
          (gelu): GELU(approximate='tanh')-
          (c_proj): Linear(in_features=768, out_features=192, bias=True)
        )
        (ls_2): Identity()
      )
    )
  )
  (token_embedding): Embedding(32000, 192)
  (ln_final): LayerNorm((192,), eps=1e-06, elementwise_affine=True)
)
'''

config_tiny_patch16_160 = Config(
    block_size=80,
    vocab_size=32000,
    n_embed=192,
    n_hidden=768,
    image_size=160,
    patch_size=16,
    ln_epsilon=1e-6,
    hf_tokenizer_name="bert-base-uncased",
    n_heads=3,
    n_text_layers=12,
    n_vision_layers=12,
    mlp_bias=True,
    conv_bias=True,
)


class VisionBlock(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.attn = SelfAttention(config, rngs=rngs, causal=False)
        self.rms_n_1 = nnx.RMSNorm(config.n_embed, 
                                   epsilon=config.ln_epsilon,
                                   dtype=config.dtype, rngs=rngs)
        self.rms_n_2 = nnx.RMSNorm(config.n_embed, 
                                   epsilon=config.ln_epsilon,
                                   dtype=config.dtype, rngs=rngs)
        self.mlp = MLP(config, rngs, approx_gelu=False)


    def __call__(self, x):
        x = x + self.attn(self.rms_n_1(x))
        x = x + self.mlp(self.rms_n_2(x))
        return x


class VisionEncoder(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.blocks = [ VisionBlock(config, rngs) for _ in range(config.n_vision_layers) ]
        if config.patch_embed == "linear":
            self.patch_embed = nnx.Linear(config.n_channels * config.patch_size * config.patch_size, 
                                        config.n_embed, 
                                        kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
                                        use_bias=config.mlp_bias,
                                        dtype=config.dtype,
                                        rngs=rngs)
        elif config.patch_embed == "conv":
            self.patch_embed = nnx.Conv(
                in_features=3,  # assuming RGB input
                out_features=config.n_embed,
                kernel_size=config.patch_size,              # int or (patch_size, patch_size)
                strides=config.patch_size,                  # int or (patch_size, patch_size)
                padding='VALID',
                use_bias=config.conv_bias,
                dtype=config.dtype,
                kernel_init=nnx.initializers.glorot_uniform,
                rngs=rngs
            )


    def __call__(self, x_image):
        if self.config.patch_embed == "linear":
            x_image = x_image.reshape(B, self.config.n_channels, 
                                    self.config.grid_size, self.config.patch_size, 
                                    self.config.grid_size, self.config.patch_size)
            x_image = jnp.transpose(x_image, (0, 2, 4, 1, 3, 5))
            x_image = x_image.reshape(B, 
                                self.config.grid_size * self.config.grid_size, 
                                self.config.n_channels * self.config.patch_size * self.config.patch_size)
            x_image = self.patch_embed(x_image) 
        elif self.config.patch_embed == "conv":
            x_image = self.patch_embed(x_image)
            x_image = jax.debug.breakpoint()
            _, h, w, c = x.shape
            x_image = jnp.reshape(x_image, [_, h * w, c])
        for i in range(self.config.n_vision_layers):
            x_image = self.blocks[i](x_image)
        return x_image


class TextBlock(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.attn = SelfAttention(config, rngs=rngs, causal=True)
        self.rms_n_1 = nnx.RMSNorm(config.n_embed, 
                                   epsilon=config.ln_epsilon,
                                   dtype=config.dtype, rngs=rngs)
        self.rms_n_2 = nnx.RMSNorm(config.n_embed, 
                                   epsilon=config.ln_epsilon,
                                   dtype=config.dtype, rngs=rngs)
        self.mlp = MLP(config, rngs, approx_gelu=True)


    def __call__(self, x):
        x = x + self.attn(self.rms_n_1(x))
        x = x + self.mlp(self.rms_n_2(x))
        return x
    

class TextDecoder(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.h = [ TextBlock(config, rngs) for _ in range(config.n_text_layers) ]
        self.wte = nnx.Embed(config.vocab_size, config.n_embed, 
                                     embedding_init=nnx.initializers.normal(stddev=config.init_stddev),
                                     dtype=config.dtype, rngs=rngs)
        self.rms_n_f = nnx.RMSNorm(config.n_embed, 
                                   epsilon=config.ln_epsilon,
                                   dtype=config.dtype, rngs=rngs)
    

    def __call__(self, x_image, x_text):
        _, T = x_text.shape
        _, IT, _ = x_image.shape
        x_text = self.wte(x_text)
        x = jnp.concat([x_image, x_text], axis=1)
        for i in range(self.config.n_text_layers):
            x = self.h[i](x)
        x = self.rms_n_f(x)
        x = self.wte.attend(x[:, -T-1:-1, :])
        return x


class OpenVision2(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.vision_encoder = VisionEncoder(config, rngs)
        self.text_decoder = TextDecoder(config, rngs)


    def __call__(self, x_image, x_text):
        x_image = self.vision_encoder(x_image)
        y = self.text_decoder(x_image, x_text)
        return y

