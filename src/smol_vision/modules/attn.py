from typing import Literal

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.models.smol_vision.config import Config
from smol_vision.modules.rope import RoPE


def _calc_attention(
    query,
    key,
    value,
    mask=None,
    bias=None,
    implementation: Literal["xla", "cudnn", "slow"] | None = None,
):
    output_shape = jnp.asarray(query).shape

    if mask is not None:
        # Convert a (B x T) mask to a (B x 1 x T x T) mask
        mask1 = mask[..., :, None]
        mask2 = mask[..., None, :]
        mask = mask1 & mask2
        mask = mask[:, None, :, :]
    out = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=mask,
        bias=bias,
        is_causal=True,
        implementation=implementation,
    )

    return jnp.reshape(out, output_shape)


class SelfAttention(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs, causal: bool = False):
        self.config = config
        slef.wqkv = nnx.Linear(
            config.n_embed,
            3 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            use_bias=config.use_bias,
            dtype=config.dtype,
            rngs=rngs
        )
        self.c_proj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init = nnx.initializers.normal(
                stddev=config.init_stddev * (2 * (config.n_text_layers + config.n_vision_layers) ** -0.5)
            ),
            use_bias=config.use_bias,
            dtype=config.dtype,
            rngs=rngs
        )
        self.apply_rope = RoPE(config)
    

    def __call__(self, x):
        B, _, C = x.shape
        qkv = self.wqkv(x)
        q, k, v = jnp.split(kv, 3, axis=-1) 

        q = q.reshape((B, -1, self.config.n_heads, C // self.config.n_heads))
        k = k.reshape((B, -1, self.config.n_heads, C // self.config.n_heads))
        v = v.reshape((B, -1, self.config.n_heads, C // self.config.n_heads))
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        y = _calc_attention(
            q, k, v, mask=mask 
        )  # (B, T, n_head, C // n_head)
        y = jnp.reshape(y, (B, -1, C))
        y = self.c_proj(y)
        return y



class GQAttention(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs, causal: bool = False):
        self.config = config
        self.wq = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            use_bias=config.use_bias,
            dtype=config.dtype,
            rngs=rngs
        )
        self.wkv = nnx.Linear(
            config.n_embed,
            2 * config.n_kv_heads * config.n_embed // config.n_heads,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            use_bias=config.use_bias,
            dtype=config.dtype,
            rngs=rngs
        )
        self.c_proj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init = nnx.initializers.normal(
                stddev=config.init_stddev * (2 * (config.n_text_layers + config.n_vision_layers) ** -0.5)
            ),
            use_bias=config.use_bias,
            dtype=config.dtype,
            rngs=rngs
        )
        self.apply_rope = RoPE(config)
    

    def __call__(self, x, mask=None):
        B, _, C = x.shape
        q = self.wq(x) 
        kv = self.wkv(x)
        k, v = jnp.split(kv, 2, axis=-1) 

        q = q.reshape((B, -1, self.config.n_heads, C // self.config.n_heads))
        k = k.reshape((B, -1, self.config.n_kv_heads, C // self.config.n_heads))
        v = v.reshape((B, -1, self.config.n_kv_heads, C // self.config.n_heads))
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        y = _calc_attention(
            q, k, v, mask=mask 
        )  # (B, T, n_head, C // n_head)
        y = jnp.reshape(y, (B, -1, C))
        y = self.c_proj(y)
        return y
 