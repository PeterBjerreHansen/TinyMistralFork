#!/usr/bin/env python
"""Compare real-checkpoint full logits with rolling-cache logits."""
from __future__ import annotations

import argparse

import torch

from tiny_mistral.loading import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="checkpoints/TinyMistral-248M-v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--length", type=int, default=96)
    args = parser.parse_args()
    if args.length < 1:
        raise SystemExit("length must be positive")

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = load_model(
        args.model_dir,
        attention_backend="reference",
        device=device,
        dtype=dtype,
        compile_flex=False,
    ).eval()
    ids = ((torch.arange(args.length, device=device)[None, :] * 101 + 1) % model.config.vocab_size).long()

    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        cache = None
        pieces: list[torch.Tensor] = []
        for position in range(args.length):
            out = model(ids[:, position : position + 1], past_key_values=cache, use_cache=True)
            pieces.append(out.logits)
            cache = out.past_key_values
        incremental = torch.cat(pieces, dim=1)

    diff = (full - incremental).abs().float()
    print(f"length={args.length} max_abs_diff={diff.max().item():.8g}")
    print(f"mean_abs_diff={diff.mean().item():.8g}")
    # Full and incremental paths use different matmul groupings after the
    # rolling window starts evicting keys, so allow small accumulated FP32
    # rounding differences while remaining far below model-scale changes.
    atol, rtol = ((5e-5, 5e-5) if dtype == torch.float32 else (8e-3, 8e-3))
    torch.testing.assert_close(incremental, full, atol=atol, rtol=rtol)
    print("PASS: real-checkpoint rolling-cache logits match full recomputation")


if __name__ == "__main__":
    main()
