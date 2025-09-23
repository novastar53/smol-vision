"""HuggingFace `datasets` backed dataloader for Recap COCO.

This module provides a `make_dataloader` function which yields batches of
JAX arrays suitable for feeding into a Flax model: images as float32
arrays with shape [B, C, H, W], and tokenized captions as int32 arrays
with shape [B, L], along with a mask indicating valid tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from typing import Sequence

# ImageNet-like normalization (float32 channel-last mean/std)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class DataConfig:
    batch_size: int = 32
    num_epochs: Optional[int] = None
    shuffle: bool = True
    shuffle_buffer: int = 50_000
    drop_last: bool = True
    data_dir: Optional[str] = None
    seed: int = 0
    shard_for_pmap: bool = False
    prefetch: int = 2

    # Image preprocessing
    resize_to: int = 224
    augment: bool = True

    # Caption/tokenization
    tokenizer: Optional[Callable[[list[str], int], np.ndarray]] = None
    max_length: int = 64
    pad_id: int = 0
    add_bos: bool = False
    add_eos: bool = True


def _default_tokenizer(texts: list[str], max_length: int) -> np.ndarray:
    """Tokenize using the HuggingFace SmolLM tokenizer.

    This function assumes `transformers` is installed and will raise an
    import error otherwise (handled at module import time).
    """
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", use_fast=True)
    tok.pad_token = tok.eos_token
    out = tok(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    return out["input_ids"].astype(np.int32)


def _hf_batch_generator(dataset_name: str, split: str, cfg: DataConfig) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Iterate over a HuggingFace `datasets` dataset and yield JAX batches.

    This function performs minimal preprocessing (resize, normalize)
    using PIL and numpy, tokenizes captions with the configured tokenizer
    (or the default SmolLM tokenizer), and yields batches as JAX arrays along with a mask.
    The mask indicates valid tokens, zeroing out tokens after the EOS token.
    """
    # direct imports are used; allow ImportError to propagate if dependencies are missing

    ds = load_dataset(dataset_name, split=split)

    # Optional shuffling for train split
    if cfg.shuffle and split == "train":
        try:
            ds = ds.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)
        except Exception:
            # Some dataset backends may not support shuffling metadata; fall back gracefully
            pass

    tokenizer = cfg.tokenizer or (lambda texts, L: _default_tokenizer(texts, L))

    # Determine EOS token id once
    sample_tokens = tokenizer([""], cfg.max_length)
    eos_id = sample_tokens[0][-1]

    def make_mask(toks: np.ndarray) -> np.ndarray:
        mask = np.ones_like(toks, dtype=np.int32)
        for i, seq in enumerate(toks):
            eos_positions = np.where(seq == eos_id)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0]
                mask[i, first_eos+1:] = 0
        return mask

    def proc_image(img_obj):
        # HF Image feature often yields PIL Image objects; accept numpy arrays too
        if isinstance(img_obj, (bytes, bytearray)):
            import io
            im = Image.open(io.BytesIO(img_obj)).convert("RGB")
        elif hasattr(img_obj, "convert"):
            im = img_obj.convert("RGB")
        else:
            # assume numpy array HWC
            arr = np.asarray(img_obj)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            im = Image.fromarray(arr.astype(np.uint8))

        if cfg.resize_to is not None:
            im = im.resize((cfg.resize_to, cfg.resize_to), resample=Image.BILINEAR)

        arr = np.asarray(im).astype(np.float32) / 255.0

        # simple augment: horizontal flip
        if cfg.augment and np.random.rand() < 0.5:
            arr = arr[:, ::-1, :]

        # Normalize
        arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
        # transpose to C,H,W
        arr = arr.transpose(2, 0, 1)
        return arr

    epoch = 0
    while True:
        if cfg.num_epochs is not None and epoch >= (cfg.num_epochs or 0):
            break
        batch_imgs = []
        batch_caps = []
        for example in ds:
            # get image and caption
            img_obj = example.get("image")
            cap = example.get("caption") or example.get("captions")
            if isinstance(cap, (list, tuple)):
                # pick first caption if multiple
                cap = cap[0] if len(cap) else ""

            img_arr = proc_image(img_obj)
            batch_imgs.append(img_arr)
            batch_caps.append(str(cap))

            if len(batch_imgs) == cfg.batch_size:
                imgs_np = np.stack(batch_imgs, axis=0)
                token_ids = tokenizer(batch_caps, cfg.max_length)
                mask_np = make_mask(token_ids)
                imgs_j = jnp.asarray(np.ascontiguousarray(imgs_np))
                toks_j = jnp.asarray(np.ascontiguousarray(token_ids).astype(np.int32))
                mask_j = jnp.asarray(mask_np)
                labels_np = np.concatenate(
                    [token_ids[:, 1:], np.full((token_ids.shape[0], 1), 0, dtype=np.int32)],
                    axis=1
                )
                labels_j = jnp.asarray(labels_np)
                if cfg.shard_for_pmap:
                    n_dev = jax.local_device_count()
                    b = imgs_j.shape[0]
                    assert b % n_dev == 0, f"Batch size {b} must be divisible by local_device_count {n_dev}."
                    per = b // n_dev
                    imgs_j = imgs_j.reshape(n_dev, per, *imgs_j.shape[1:])
                    toks_j = toks_j.reshape(n_dev, per, *toks_j.shape[1:])
                    mask_j = mask_j.reshape(n_dev, per, *mask_j.shape[1:])
                    labels_j = labels_j.reshape(n_dev, per, *labels_j.shape[1:])

                yield imgs_j, toks_j, mask_j, labels_j
                batch_imgs = []
                batch_caps = []

        # leftover
        if batch_imgs and not cfg.drop_last:
            imgs_np = np.stack(batch_imgs, axis=0)
            token_ids = tokenizer(batch_caps, cfg.max_length)
            mask_np = make_mask(token_ids)
            imgs_j = jnp.asarray(np.ascontiguousarray(imgs_np))
            toks_j = jnp.asarray(np.ascontiguousarray(token_ids).astype(np.int32))
            mask_j = jnp.asarray(mask_np)
            mask_j = jnp.asarray(mask_np)
            labels_np = np.concatenate(
                [token_ids[:, 1:], np.full((token_ids.shape[0], 1), 0, dtype=np.int32)],
                axis=1
            )
            labels_j = jnp.asarray(labels_np)
            if cfg.shard_for_pmap:
                n_dev = jax.local_device_count()
                b = imgs_j.shape[0]
                # pad to multiple of devices
                if b % n_dev != 0:
                    pad = n_dev - (b % n_dev)
                    imgs_j = jnp.pad(imgs_j, ((0, pad), (0, 0), (0, 0), (0, 0)))
                    toks_j = jnp.pad(toks_j, ((0, pad), (0, 0)))
                    mask_j = jnp.pad(mask_j, ((0, pad), (0, 0)))
                    b = imgs_j.shape[0]
                per = b // n_dev
                imgs_j = imgs_j.reshape(n_dev, per, *imgs_j.shape[1:])
                toks_j = toks_j.reshape(n_dev, per, *toks_j.shape[1:])
                mask_j = mask_j.reshape(n_dev, per, *mask_j.shape[1:])
                labels_j = labels_j.reshape(n_dev, per, *labels_j.shape[1:])
        
            yield imgs_j, toks_j, mask_j, labels_j

        epoch += 1



def make_dataloader(split: str, cfg: DataConfig, dataset_name: str) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Create a dataloader iterator producing (images, caption_token_ids, mask) from HF `datasets`.

    - images: [B,C,H,W] float32
    - caption_token_ids: [B, L] int32
    - mask: [B, L] int32, with zeros after the EOS token
    """
    assert split in {"train", "validation", "test"} or split.startswith("train") or split.startswith("test")
    return _hf_batch_generator(dataset_name, split, cfg)


def visualize_batch(images: jnp.ndarray,
                    token_ids: jnp.ndarray,
                    tokenizer: Optional[Callable[[list[int]], str]] = None,
                    max_display: int = 16,
                    figsize: Tuple[int, int] = (12, 8)) -> None:
    """Display a batch of images with decoded captions.

    - images: [B, C, H, W] float32 (normalized)
    - token_ids: [B, L] int32
    - tokenizer: optional callable to decode a list of token ids -> string. If
      not provided, we use the default SmolLM tokenizer to decode.
    """
    # move to CPU and numpy
    imgs = np.array(images)
    toks = np.array(token_ids)

    B = imgs.shape[0]
    n = min(B, max_display)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # denormalize images back to [H,W,C] uint8
    def denorm(im: np.ndarray) -> np.ndarray:
        # im shape C,H,W
        im = im.transpose(1, 2, 0)
        im = (im * IMAGENET_STD.reshape(1, 1, 3)) + IMAGENET_MEAN.reshape(1, 1, 3)
        im = np.clip(im * 255.0, 0, 255).astype(np.uint8)
        return im

    # decoder
    if tokenizer is None:
        hf_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", use_fast=True)

        def decode_fn(ids: Sequence[int]) -> str:
            return hf_tok.decode([int(i) for i in ids], skip_special_tokens=True)
    else:
        decode_fn = tokenizer

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        im = denorm(imgs[i])
        ax.imshow(im)
        ax.axis("off")
        caption = decode_fn(toks[i])
        ax.set_title(caption, fontsize=9)

    # turn off remaining axes
    for j in range(n, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = DataConfig(batch_size=8, num_epochs=1, shuffle=False, resize_to=128, augment=False)
    it = make_dataloader("train", cfg, "UCSC-VLAA/Recap-COCO-30K")
    imgs, toks, mask, labels = next(it)
    print("imgs:", imgs.shape, imgs.dtype)
    print("toks:", toks.shape, toks.dtype)
    print("labels:", labels.shape, labels.dtype)
    print("mask:", mask.shape, mask.dtype)
    visualize_batch(imgs, toks)