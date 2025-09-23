import sys

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np

from smol_vision.model import Config, SmolVision
from smol_vision.datasets.recap_coco import DataConfig, make_dataloader, visualize_batch
from smol_vision.from_pretrained import from_hf_pretrained

from transformers import AutoTokenizer

# --- Helpers: truncate at EOS and decode ---

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", use_fast=True)

# Determine EOS id (prefer explicit token string; fallback to tokenizer's eos_token_id)
_eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
if _eos_id is None or (hasattr(tok, "unk_token_id") and _eos_id == tok.unk_token_id):
    _eos_id = tok.eos_token_id

def truncate_at_eos(batch_ids, eos_id):
    """batch_ids: array-like [B, T]; returns a list of python lists, each cut after first EOS (inclusive)."""
    out = []
    for ids in np.asarray(batch_ids):
        seq = ids.tolist()
        if eos_id in seq:
            idx = seq.index(eos_id)
            seq = seq[: idx + 1]
        out.append(seq)
    return out


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


cfg = DataConfig(batch_size=256, num_epochs=10, shuffle=True, augment=False)
it = make_dataloader("train", cfg, "UCSC-VLAA/Recap-COCO-30K")


def loss_fn(m, imgs, toks, mask, labels):
    logits = m(imgs, toks)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    mask = mask.astype(logits.dtype)
    loss = loss * mask
    loss = loss.sum()/jnp.sum(mask)
    return loss, logits


@nnx.jit
def step_fn(m, optimizer, imgs, toks, mask, labels):
    diff_state = nnx.DiffState(0, head_params) # filter head params of the first argument
    (loss, logits), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(m, imgs, toks, mask, labels)
    optimizer.update(m, grads)
    return loss, logits

head_params = nnx.All(nnx.Param, nnx.PathContains('text_decoder'))
tx = optax.adam(1e-3)
optimizer = nnx.Optimizer(m, tx, wrt=head_params)


#visualize_batch(imgs, toks)
#imgs, toks = next(it)
#print(f"{imgs.shape=}")
#print(f"{toks.shape=}")
#visualize_batch(imgs, toks)
#imgs, toks = next(it)
#print(f"{imgs.shape=}")
#print(f"{toks.shape=}")

# (Optional) Greedy generation function that halts on EOS
def greedy_generate(model, images, tok, max_new_tokens=64, eos_token="<|endoftext|>"):
    eos_id = tok.convert_tokens_to_ids(eos_token) or tok.eos_token_id
    # Start with a single BOS per sample (adjust if your model expects a different prompt format)
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|bos|>")
    if bos_id is None:
        bos_id = tok.convert_tokens_to_ids("<|startoftext|>")
    B = images.shape[0]
    ids = jnp.full((B, 1), bos_id, dtype=jnp.int32)
    finished = jnp.zeros((B,), dtype=jnp.bool_)
    for _ in range(max_new_tokens):
        logits = model(images, ids)  # assumes model returns logits over the last dimension for text positions
        next_id = jnp.argmax(logits[:, -1, :], axis=-1)
        ids = jnp.concatenate([ids, next_id[:, None]], axis=1)
        finished = jnp.logical_or(finished, next_id == eos_id)
        if bool(jnp.all(finished)):
            break
    return ids

step = 0
for imgs, toks, mask, labels in it:
    loss, logits = step_fn(m, optimizer, imgs, toks, mask, labels)
    print(f"{step=},{loss=}")

    # Truncate sequences at EOS for both predictions and targets
    pred_ids = jnp.argmax(logits, axis=-1)
    trunc_pred = truncate_at_eos(pred_ids, _eos_id)
    trunc_gt   = truncate_at_eos(toks, _eos_id)

    # Decode (skip special tokens for readability)
    pred_texts = [tok.decode(seq, skip_special_tokens=True) for seq in trunc_pred]
    gt_texts   = [tok.decode(seq,   skip_special_tokens=True) for seq in trunc_gt]

    step += 1

print("Predicted  - GT ")
for pred, gt in zip(pred_texts, gt_texts):
    print("  ", pred, "|", gt)