import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.models.smol_vision.config import Config


class GLU(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs, approx_gelu=False):
        self.config = config
        self.c_fc = nnx.Linear(config.n_embed, config.n_hidden,
                                kernel_init=nnx.initializers.normal(stddev=0.02),
                                use_bias=config.mlp_bias,
                                rngs=rngs)
        self.gate = nnx.Linear(config.n_embed, config.n_hidden, 
                                kernel_init=nnx.initializers.normal(stddev=0.02),
                                use_bias=config.mlp_bias,
                                rngs=rngs)
        self.c_proj = nnx.Linear(config.n_hidden, config.n_embed, 
                                 kernel_init=nnx.initializers.normal(
                                    stddev=0.02 * (2 * (config.n_text_layers + config.n_vision_layers)) ** -0.5
                                 ),
                                 use_bias=config.mlp_bias,
                                 rngs=rngs)

    
    def __call__(self, x):
        h = self.c_fc(x)
        g = self.gate(x)
        g = nnx.gelu(g, approximate=approx_gelu)
        h = g * h
        y = self.c_proj(h)
        return y