"""Dataloader for Decap-DataComp-1B (HF datasets).

This module provides `make_dataloader` which yields batches of images and
tokenized captions as JAX arrays suitable for Flax models.

Usage:
    cfg = DataConfig(batch_size=8, num_epochs=1, resize_to=224)
    it = make_dataloader("train", cfg, dataset_name="UCSC-VLAA/Recap-DataComp-1B")
    imgs, toks = next(it)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Tuple, Sequence, Mapping, cast

import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

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
    seed: int = 0
    shard_for_pmap: bool = False

    # Image preprocessing
    resize_to: int = 224
    augment: bool = True

    # Caption/tokenization
    tokenizer: Optional[Callable[[Sequence[str], int], np.ndarray]] = None
    max_length: int = 64

    # If set, only request a slice of the dataset (e.g. 512) using HF split syntax
    sample_size: Optional[int] = None


def _default_tokenizer(texts: Sequence[str], max_length: int) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", use_fast=True)
    out = tok(list(texts), padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    return out["input_ids"].astype(np.int32)


def _get_caption(example: Any) -> str:
    # Common caption/text keys
    if isinstance(example, Mapping):
        for k in ("caption", "captions", "text", "caption_text", "label"):
            if k in example:
                v = example[k]
                if isinstance(v, (list, tuple)):
                    return str(v[0]) if len(v) else ""
                return str(v)
    # fallback: if the example itself is a string, return it
    if isinstance(example, str):
        return example
    return ""


def _get_image(example: Any):
    # Common image keys
    if isinstance(example, Mapping):
        for k in ("image", "img", "image_data", "image_bytes"):
            if k in example:
                return example[k]
        # fallback: try any value that looks like an image
        for v in example.values():
            if hasattr(v, "shape") or hasattr(v, "convert") or isinstance(v, (bytes, bytearray)):
                return v
    # if the example itself is bytes or a PIL image-like object, return it
    if isinstance(example, (bytes, bytearray)):
        return example
    if hasattr(example, "convert"):
        return example
    return None


def _proc_image(img_obj, cfg: DataConfig) -> np.ndarray:
    # Accept bytes, PIL Image, or numpy array
    if isinstance(img_obj, (bytes, bytearray)):
        import io
        im = Image.open(io.BytesIO(img_obj)).convert("RGB")
    elif hasattr(img_obj, "convert"):
        im = img_obj.convert("RGB")
    else:
        arr = np.asarray(img_obj)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        im = Image.fromarray(arr.astype(np.uint8))

    if cfg.resize_to is not None:
        try:
            # PIL >= 9 exposes Resampling enum
            resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
        except Exception:
            # Older PIL versions expose constants on the Image module
            resample = Image.BILINEAR  # type: ignore[attr-defined]
        im = im.resize((cfg.resize_to, cfg.resize_to), resample=resample)

    arr = np.asarray(im).astype(np.float32) / 255.0
    if cfg.augment and np.random.rand() < 0.5:
        arr = arr[:, ::-1, :]
    arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    arr = arr.transpose(2, 0, 1)  # C,H,W
    return arr


def _hf_batch_generator(dataset_name: str, split: str, cfg: DataConfig) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    # If a small sample_size is requested, use HF split slicing to avoid
    # downloading the entire dataset (e.g. 'train[:512]').
    if cfg.sample_size is not None:
        split_expr = f"{split}[:{cfg.sample_size}]"
        jax.debug.breakpoint()
        ds = load_dataset(dataset_name, "default", split="preview", streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split)

    # Optional shuffle (datasets' shuffle API varies; try best-effort)
    if cfg.shuffle and split == "train":
        try:
            ds = ds.shuffle(seed=cfg.seed)
        except Exception:
            pass

    tokenizer = cfg.tokenizer or _default_tokenizer

    epoch = 0
    while True:
        if cfg.num_epochs is not None and epoch >= (cfg.num_epochs or 0):
            break

        batch_imgs = []
        batch_caps = []
        for example in ds:
            img_obj = _get_image(example)
            cap = _get_caption(example)

            if img_obj is None:
                continue

            img_arr = _proc_image(img_obj, cfg)
            batch_imgs.append(img_arr)
            batch_caps.append(cap)

            if len(batch_imgs) == cfg.batch_size:
                imgs_np = np.stack(batch_imgs, axis=0)
                token_ids = tokenizer(batch_caps, cfg.max_length)
                imgs_j = jnp.asarray(np.ascontiguousarray(imgs_np))
                toks_j = jnp.asarray(np.ascontiguousarray(token_ids).astype(np.int32))
                if cfg.shard_for_pmap:
                    n_dev = jax.local_device_count()
                    b = imgs_j.shape[0]
                    assert b % n_dev == 0, f"Batch size {b} must be divisible by local_device_count {n_dev}."
                    per = b // n_dev
                    imgs_j = imgs_j.reshape(n_dev, per, *imgs_j.shape[1:])
                    toks_j = toks_j.reshape(n_dev, per, *toks_j.shape[1:])
                yield imgs_j, toks_j
                batch_imgs = []
                batch_caps = []

        # leftover
        if batch_imgs and not cfg.drop_last:
            imgs_np = np.stack(batch_imgs, axis=0)
            token_ids = tokenizer(batch_caps, cfg.max_length)
            imgs_j = jnp.asarray(np.ascontiguousarray(imgs_np))
            toks_j = jnp.asarray(np.ascontiguousarray(token_ids).astype(np.int32))
            if cfg.shard_for_pmap:
                n_dev = jax.local_device_count()
                b = imgs_j.shape[0]
                if b % n_dev != 0:
                    pad = n_dev - (b % n_dev)
                    imgs_j = jnp.pad(imgs_j, ((0, pad), (0, 0), (0, 0), (0, 0)))
                    toks_j = jnp.pad(toks_j, ((0, pad), (0, 0)))
                    b = imgs_j.shape[0]
                per = b // n_dev
                imgs_j = imgs_j.reshape(n_dev, per, *imgs_j.shape[1:])
                toks_j = toks_j.reshape(n_dev, per, *toks_j.shape[1:])
            yield imgs_j, toks_j

        epoch += 1


def make_dataloader(split: str, cfg: DataConfig, dataset_name: str) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    assert split in {"train", "validation", "test"} or split.startswith("train") or split.startswith("test")
    return _hf_batch_generator(dataset_name, split, cfg)


def visualize_batch(images: jnp.ndarray,
                    token_ids: jnp.ndarray,
                    tokenizer: Optional[Callable[[Sequence[int]], str]] = None,
                    max_display: int = 16,
                    figsize: Tuple[int, int] = (12, 8)) -> None:
    imgs = np.array(images)
    toks = np.array(token_ids)
    B = imgs.shape[0]
    n = min(B, max_display)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    def denorm(im: np.ndarray) -> np.ndarray:
        im = im.transpose(1, 2, 0)
        im = (im * IMAGENET_STD.reshape(1, 1, 3)) + IMAGENET_MEAN.reshape(1, 1, 3)
        im = np.clip(im * 255.0, 0, 255).astype(np.uint8)
        return im

    if tokenizer is None:
        hf_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", use_fast=True)

        def decode_fn(ids: Sequence[int]) -> str:
            return hf_tok.decode([int(i) for i in ids], skip_special_tokens=True)
    else:
        # Wrap the provided tokenizer to a decode_fn(ids) -> str signature.
        assert tokenizer is not None

        def decode_fn(ids: Sequence[int]) -> str:
            return tokenizer(ids)  # type: ignore[call-arg]

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        im = denorm(imgs[i])
        ax.imshow(im)
        ax.axis("off")
        caption = decode_fn(toks[i])
        ax.set_title(caption, fontsize=9)

    for j in range(n, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick demo: load one batch and visualize
    cfg = DataConfig(batch_size=8, num_epochs=1, shuffle=False, resize_to=224, augment=False, sample_size=64)
    dataset_name = "UCSC-VLAA/Recap-DataComp-1B"
    print(f"Building dataloader for {dataset_name} ...")
    it = make_dataloader("train", cfg, dataset_name=dataset_name)
    imgs, toks = next(it)
    print("Loaded batch:", imgs.shape, toks.shape)
    visualize_batch(imgs, toks)
