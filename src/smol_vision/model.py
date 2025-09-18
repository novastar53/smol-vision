import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.config import Config
from smol_vision.glu import GLU
from smol_vision.attn import SelfAttention


class VisionBlock(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.attn = SelfAttention(config, rngs=rngs, causal=False)
        self.rms_n_1 = nnx.RMSNorm(config.n_embed, rngs=rngs)
        self.rms_n_2 = nnx.RMSNorm(config.n_embed, rngs=rngs)
        self.glu = GLU(config, rngs)


    def __call__(self, x):
        x = x + self.attn(self.rms_n_1(x))
        x = x + self.glu(self.rms_n_2(x))
        return x


class TextBlock(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.attn = SelfAttention(config, rngs=rngs, causal=True)
        self.rms_n_1 = nnx.RMSNorm(config.n_embed, rngs=rngs)
        self.rms_n_2 = nnx.RMSNorm(config.n_embed, rngs=rngs)
        self.glu = GLU(config, rngs)


    def __call__(self, x):
        x = x + self.attn(self.rms_n_1(x))
        x = x + self.glu(self.rms_n_2(x))
        return x


class SmolVision(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.vision_blocks = [ VisionBlock(config, rngs) for _ in range(config.n_vision_layers) ]
        self.text_blocks = [ TextBlock(config, rngs) for _ in range(config.n_text_layers) ]
        self.patch_embed = nnx.Linear(config.n_channels * config.patch_size * config.patch_size, 
                                      config.n_embed, 
                                      rngs=rngs)
        self.token_embed = nnx.Embed(config.vocab_size, config.n_embed, rngs=rngs)


    def __call__(self, x_image, x_text):
        B, _, _, _ = x_image.shape
        _, T = x_text.shape
        x_image = x_image.reshape(B, self.config.n_channels, 
                                  self.config.grid_size, self.config.patch_size, 
                                  self.config.grid_size, self.config.patch_size)
        x_image = jnp.transpose(x_image, (0, 2, 4, 1, 3, 5))
        x_image = x_image.reshape(B, 
                              self.config.grid_size * self.config.grid_size, 
                              self.config.n_channels * self.config.patch_size * self.config.patch_size)
        x_image = self.patch_embed(x_image) 
        for i in range(self.config.n_vision_layers):
            x_image = self.vision_blocks[i](x_image)
        x_text = self.token_embed(x_text)
        x = jnp.concat([x_image, x_text], axis=1)
        for i in range(self.config.n_text_layers):
            x = self.text_blocks[i](x)
        x_tokens = x[:, -T:, :]
        y = self.token_embed.attend(x_tokens)
        return y








