#!/usr/bin/env python
"""Tiny ordinary continued-pretraining baseline on a local text file.

This is not a competitive training recipe. It exists to exercise the real
pretrained checkpoint, tokenizer, local attention backend, optimizer, and causal
LM objective without introducing any MPTT/recurrent-memory machinery.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from tiny_mistral.device import resolve_device, resolve_dtype, synchronize
from tiny_mistral.loading import load_model


def load_token_stream(tokenizer_path: Path, text_path: Path, bos_token_id: int) -> list[int]:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit('install tokenizer support with: pip install -e ".[io]"') from exc

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = text_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise RuntimeError(f"tokenizer produced no tokens for {text_path}")
    # Repeated corpus boundaries receive a BOS separator when blocks wrap.
    return [bos_token_id] + ids


def sample_batch(
    stream: list[int],
    *,
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    if len(stream) < 2:
        raise ValueError("token stream is too short")
    needed = seq_len + 1
    # Repeat the tiny offline corpus enough to make any requested smoke length.
    repeats = (needed // len(stream)) + 2
    tiled = stream * repeats
    max_start = len(tiled) - needed
    rows = []
    for _ in range(batch_size):
        start = rng.randint(0, max_start)
        rows.append(tiled[start : start + seq_len])
    return torch.tensor(rows, dtype=torch.long, device=device)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="checkpoints/TinyMistral-248M-v3")
    p.add_argument("--text-file", default=str(root / "examples" / "baseline_corpus.txt"))
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--backend", choices=["auto", "reference", "flex", "local"], default="auto")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    if args.steps <= 0 or args.seq_len < 2 or args.batch_size <= 0:
        raise SystemExit("steps and batch-size must be positive; seq-len must be >= 2")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model = load_model(
        args.model_dir,
        attention_backend=args.backend,
        device=device,
        dtype=dtype,
    ).train()
    stream = load_token_stream(
        Path(args.model_dir) / "tokenizer.json",
        Path(args.text_file),
        model.config.bos_token_id,
    )
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    print(
        f"device={device} dtype={next(model.parameters()).dtype} backend={args.backend} "
        f"steps={args.steps} batch={args.batch_size} seq_len={args.seq_len}"
    )
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        ids = sample_batch(
            stream,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            rng=rng,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        out = model(ids, labels=ids, use_cache=False)
        if out.loss is None or not bool(torch.isfinite(out.loss).item()):
            raise RuntimeError(f"non-finite loss at step {step}")
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        synchronize(device)
        value = float(out.loss.detach().cpu())
        losses.append(value)
        print(f"step={step:03d} loss={value:.6f} grad_norm={float(grad_norm.detach().cpu()):.4f}")

    print(f"first_loss={losses[0]:.6f} last_loss={losses[-1]:.6f}")
    print("PASS: ordinary single-pass causal-LM baseline training completed")


if __name__ == "__main__":
    main()
