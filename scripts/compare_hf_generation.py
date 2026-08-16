#!/usr/bin/env python
"""Compare greedy cached generation with Transformers 4.45.2."""
from __future__ import annotations

import argparse
import gc

import torch

from tiny_mistral.loading import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="checkpoints/TinyMistral-248M-v3")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("max-new-tokens must be non-negative")

    import transformers
    from transformers import AutoModelForCausalLM

    if transformers.__version__ != "4.45.2":
        raise RuntimeError(f"expected transformers==4.45.2, got {transformers.__version__}")

    ids = torch.tensor([[1, 42, 314, 2718, 7, 99, 1234, 17]], dtype=torch.long)
    mask = torch.ones_like(ids)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    with torch.no_grad():
        hf_ids = hf.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=hf.config.eos_token_id,
        )
    del hf
    gc.collect()

    ours = load_model(
        args.model_dir,
        attention_backend="reference",
        device="cpu",
        dtype=torch.float32,
        compile_flex=False,
    ).eval()
    with torch.no_grad():
        ours_ids = ours.generate(ids, args.max_new_tokens, temperature=0.0)

    expected_length = ids.shape[1] + args.max_new_tokens
    print(f"HF shape: {tuple(hf_ids.shape)}")
    print(f"local shape: {tuple(ours_ids.shape)}")
    if hf_ids.shape != (1, expected_length) or ours_ids.shape != (1, expected_length):
        raise AssertionError(
            "generation stopped before all requested tokens were produced: "
            f"expected length {expected_length}, HF={hf_ids.shape[1]}, local={ours_ids.shape[1]}"
        )
    torch.testing.assert_close(ours_ids, hf_ids)
    print("PASS: greedy cached generation matches Transformers 4.45.2")


if __name__ == "__main__":
    main()
