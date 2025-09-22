from typing import Literal

import jax.numpy as jnp

from dataclasses import dataclass


@dataclass
class Config:
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    init_stddev: float = 0.02
    vocab_size: int = 49152
    block_size: int = 2048 
    n_vision_layers: int = 4
    n_text_layers: int = 30 
    n_embed: int = 576
    n_hidden: int = 1536
    n_heads: int = 9 
    n_kv_heads: int = 3
    ln_epsilon: float = 1e6
    image_size: int = 224
    patch_size: int = 28
    grid_size: int = image_size // patch_size
    n_channels: int = 3
    rope_theta: float = 1e-4  # base frequency for rope
    ln_epsilon: float = 1e-5 # constant to prevent division by zero
    sdpa_implementation: Literal["xla", "cudnn", "slow"] = (
        "xla"  # self-attention kernel implementation
    )

