import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.config import Config


class GLU(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.c_fc = nnx.Linear(config.n_embed, config.n_hidden, rngs=rngs)
        self.gate = nnx.Linear(config.n_embed, config.n_hidden, rngs=rngs)
        self.c_proj = nnx.Linear(config.n_hidden, config.n_embed, rngs=rngs)

    
    def __call__(self, x):
        h = self.c_fc(x)
        g = self.gate(x)
        g = nnx.silu(g)
        h = g * h
        y = self.c_proj(h)
        return y