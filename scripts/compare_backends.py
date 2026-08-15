#!/usr/bin/env python
"""Compare an optimized attention backend with the dense reference using real weights."""
from __future__ import annotations

import argparse
import warnings

import torch

from tiny_mistral.device import resolve_device, resolve_dtype
from tiny_mistral.loading import load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="checkpoints/TinyMistral-248M-v3")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--optimized-backend", choices=["auto", "flex", "local"], default="auto")
    p.add_argument("--lengths", nargs="+", type=int, default=[31, 32, 33, 64, 129])
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--rtol", type=float, default=None)
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    assert dtype is not None
    model = load_model(
        args.model_dir,
        attention_backend="reference",
        device=device,
        dtype=dtype,
        compile_flex=True,
    ).eval()

    if dtype == torch.float32:
        default_tol = 3e-5
    elif dtype == torch.bfloat16:
        default_tol = 1.5e-2
    else:
        default_tol = 3e-2
    atol = default_tol if args.atol is None else args.atol
    rtol = default_tol if args.rtol is None else args.rtol

    for T in args.lengths:
        ids = ((torch.arange(T, device=device)[None, :] * 101 + 1) % model.config.vocab_size).long()
        with torch.no_grad():
            model.set_attention_backend("reference")
            ref = model(ids, use_cache=False).logits[:, :, :128].float()
            model.set_attention_backend(args.optimized_backend, compile_flex=(device.type == "cuda"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                opt = model(ids, use_cache=False).logits[:, :, :128].float()
        diff = (ref - opt).abs()
        print(
            f"T={T:4d} backend={args.optimized_backend:5s} "
            f"max_abs_diff={diff.max().item():.8g} mean_abs_diff={diff.mean().item():.8g}"
        )
        torch.testing.assert_close(opt, ref, atol=atol, rtol=rtol)
    print(f"PASS: {args.optimized_backend} and reference paths agree on all tested lengths")


if __name__ == "__main__":
    main()
