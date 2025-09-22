import sys

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from smol_vision.model import Config, SmolVision
from smol_vision.datasets.recap_coco import DataConfig, make_dataloader, visualize_batch
from smol_vision.from_pretrained import from_hf_pretrained


def count_params(model):
    graphdef, state = nnx.split(m, ...)
    counts = jax.tree_util.tree_map(lambda x: x.size, state)
    total = jax.tree_util.tree_reduce(lambda x,y: x + y, counts)
    return total


config = Config()
rngs = nnx.Rngs(default=0)
#m = SmolVision(config, rngs)
m = from_hf_pretrained(config, rngs)
total_params = count_params(m)
print(f"{total_params=:,}")


cfg = DataConfig(batch_size=8, num_epochs=1, shuffle=False, augment=False)
it = make_dataloader("train", cfg, "UCSC-VLAA/Recap-COCO-30K")


def loss_fn(m, imgs, toks):
    logits = m(imgs, toks)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, toks)
    loss = loss.mean()
    return loss


@nnx.jit
def step_fn(m, optimizer, imgs, toks):
    loss, grads = nnx.value_and_grad(loss_fn)(m, imgs, toks)
    optimizer.update(m, grads)
    return loss


tx = optax.adam(1e-4)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)


#visualize_batch(imgs, toks)
#imgs, toks = next(it)
#print(f"{imgs.shape=}")
#print(f"{toks.shape=}")
for i in range(80):
    imgs, toks = next(it)
    print(toks.mean())
    loss = step_fn(m, optimizer, imgs, toks)
    print(f"{loss=}")