#!/usr/bin/env python
"""One optimizer-step smoke test on the real checkpoint.

This is deliberately not a training recipe. It proves the vanilla local fork can
participate in autograd/optimization before any later research modifications.
"""
from __future__ import annotations

import argparse

import torch

from tiny_mistral.device import resolve_device, resolve_dtype
from tiny_mistral.loading import load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="checkpoints/TinyMistral-248M-v3")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--backend", choices=["reference", "flex", "local", "auto"], default="auto")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model = load_model(
        args.model_dir,
        attention_backend=args.backend,
        device=device,
        dtype=dtype,
    ).train()
    ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    before = model(ids, labels=ids, use_cache=False).loss
    assert before is not None and torch.isfinite(before)
    optimizer.zero_grad(set_to_none=True)
    before.backward()
    finite_grad = all(
        p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters()
    )
    if not finite_grad:
        raise RuntimeError("non-finite gradient detected")
    optimizer.step()
    after = model(ids, labels=ids, use_cache=False).loss
    print(f"device={device} dtype={next(model.parameters()).dtype} backend={args.backend}")
    print(f"loss_before={before.item():.6f}")
    print(f"loss_after ={after.item():.6f}")
    print("PASS: forward/backward/optimizer step completed with finite gradients")


if __name__ == "__main__":
    main()
