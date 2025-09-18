import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.config import Config
from smol_vision.rope import RoPE


class SelfAttention(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs, causal: bool = False):
        self.config = config
        self.causal=causal
        self.qkv = nnx.Linear(config.n_embed, 3 * config.n_embed, rngs=rngs)
        self.rope = RoPE(config)


    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.config.n_heads, C // self.config.n_heads)        
        k = k.reshape(B, T, self.config.n_heads, C // self.config.n_heads)        
        v = v.reshape(B, T, self.config.n_heads, C // self.config.n_heads)        

        q = self.rope(q)
        k = self.rope(k)
    
        y = jax.nn.dot_product_attention(
                q,
                k,
                v,
                implementation=self.config.sdpa_implementation,
                is_causal=True,
        )
        y = y.reshape(B, T, C)
        return y
