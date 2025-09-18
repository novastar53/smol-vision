import jax
import jax.numpy as jnp
import flax.nnx as nnx

from smol_vision.config import Config


def calc_rope_omega_llama(
    n_embed: int,
    n_head: int,
    block_size: int,
    rope_base_freq: float,
    dtype: jnp.dtype,
) -> nnx.Variable:
    query_size = n_embed // n_head
    pow = jnp.arange(0, query_size, 2, dtype=dtype)
    omega = rope_base_freq ** (pow / query_size)
    omega = jnp.concat([omega, omega], axis=0)
    pos = jnp.arange(0, block_size, dtype=dtype)
    pos = jnp.expand_dims(pos, axis=1)
    omega = omega * pos
    return nnx.Variable(omega)


class RoPE:
    def __init__(self, config: Config):
        omega = calc_rope_omega_llama(
            config.n_embed,
            config.n_heads,
            config.block_size,
            config.rope_theta,
            config.dtype,
        )
        self.omega = omega

    def rotate_half(self, x):
        n = x.shape[-1] // 2
        return jnp.concat((-x[..., n:], x[..., :n]), axis=-1)

    def __call__(self, v, offset=0):
        v = v.swapaxes(1, 2)
        omega = self.omega[offset : offset + v.shape[-2], :]
        a = v * jnp.cos(omega)
        b = self.rotate_half(v) * jnp.sin(omega)
        y = a + b
        y = y.swapaxes(1, 2)
        return y

