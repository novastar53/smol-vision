import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.models.smol_vision.config import Config


class MLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs, approx_gelu=False):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_hidden,
            kernel_init= nnx.initializers.normal(stddev=0.02),
            dtype=config.dtype,
            use_bias=config.mlp_bias,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_hidden,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                    stddev=0.02 * (2 * (config.n_vision_layers + config.n_text_layers)) ** -0.5
            ),
            dtype=config.dtype,
            use_bias=config.mlp_bias,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=approx_gelu)
        x = self.c_proj(x)
        return x
