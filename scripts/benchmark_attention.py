#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
import warnings

import torch

from tiny_mistral.config import tiny_mistral_248m_config
from tiny_mistral.device import resolve_device, resolve_dtype, synchronize
from tiny_mistral.modeling import MistralAttention


def bench(module, x, positions, iters: int) -> float:
    for _ in range(2):
        module(x, attention_mask=None, position_ids=positions, use_cache=False)
    synchronize(x.device)
    start = time.perf_counter()
    for _ in range(iters):
        module(x, attention_mask=None, position_ids=positions, use_cache=False)
    synchronize(x.device)
    return (time.perf_counter() - start) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--backends", nargs="+", choices=["reference", "local", "flex", "auto"], default=None)
    p.add_argument("--reference-max-len", type=int, default=512)
    args = p.parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    assert dtype is not None
    cfg = tiny_mistral_248m_config()

    if args.backends is None:
        backends = ["local", "reference"] if device.type == "mps" else ["flex", "local", "reference"]
    else:
        backends = args.backends

    print(f"device={device} dtype={dtype} window={cfg.sliding_window} backends={backends}")
    for T in args.lengths:
        x = torch.randn(1, T, cfg.hidden_size, device=device, dtype=dtype)
        pos = torch.arange(T, device=device)[None, :]
        rows = []
        for backend in backends:
            if backend == "reference" and T > args.reference_max_len:
                continue
            if backend == "flex" and device.type == "mps":
                rows.append("flex=unsupported-on-mps")
                continue
            attn = MistralAttention(
                cfg,
                0,
                attention_backend=backend,
                compile_flex=(device.type == "cuda"),
            ).to(device=device, dtype=dtype).eval()
            try:
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    sec = bench(attn, x, pos, args.iters)
                rows.append(f"{backend}={sec*1000:.2f}ms")
            except Exception as exc:
                rows.append(f"{backend}=ERROR({type(exc).__name__})")
        print(f"T={T:5d}  " + "  ".join(rows))


if __name__ == "__main__":
    main()
