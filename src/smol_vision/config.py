from typing import Literal

import jax.numpy as jnp

from dataclasses import dataclass


@dataclass
class Config:
    dtype: jnp.dtype = jnp.float32
    vocab_size: int = 50304
    block_size: int = 2048 
    n_vision_layers: int = 4
    n_text_layers: int = 4
    n_embed: int = 576
    n_hidden: int = n_embed * 3
    n_heads: int = 4 
    image_size: int = 224
    patch_size: int = 28
    grid_size: int = image_size // patch_size
    n_channels: int = 3

    rope_theta: float = 1e-4  # base frequency for rope
    sdpa_implementation: Literal["xla", "cudnn", "slow"] = (
        "xla"  # self-attention kernel implementation
    )

