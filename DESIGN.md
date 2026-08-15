# Design notes

## Goal

This repository is a vanilla TinyMistral/Mistral reference implementation. It exists to give later architecture experiments a fixed control implementation whose checkpoint loading, masking, attention, caching, generation, and autograd behavior have already been tested.

No MPTT-specific abstraction is present in this baseline.

## Parameter compatibility

Module names mirror the Hugging Face Mistral hierarchy so the pinned safetensors state dict loads with `strict=True` without renaming tensors.

## Attention hierarchy

Three full-sequence attention implementations share the same semantics:

1. `reference`: dense eager implementation used as the correctness oracle.
2. `flex`: CUDA-oriented block-sparse FlexAttention backend.
3. `local`: MPS-oriented O(TW) exact local-window implementation using ordinary PyTorch operators.

The `local` backend left-pads K/V by `W-1`, uses `unfold` to create a view of each causal K/V window, groups query heads by KV head for GQA, computes `[T,W]` scores with batched matmul, masks only the invalid leading padding, and performs FP32 softmax before returning to the query dtype.

It deliberately does not support arbitrary padding or nonzero/noncontiguous position IDs in its fast path. Those cases fall back to `reference` so correctness is preserved before generality is optimized.

## MPS

`auto` selects `local` for compatible MPS full-sequence forwards. No FlexAttention call is attempted on MPS. The hardware test under `scripts/mps_smoke.py` verifies forward agreement and backward on an actual MPS runtime.

The default MPS dtype is FP16. Users may explicitly request BF16 if their macOS/PyTorch/hardware combination supports it.

## Cached decode

The custom cache stores already-RoPE-rotated K/V. Under the v4.45.2 sliding-window convention the next token needs at most `W-1` previous cached positions; after the current token is appended, at most W keys participate in attention. Cached decode therefore uses the obvious reference math because it is already bounded in sequence length.

## Research boundary

Future recurrence/memory work should happen on a branch made after the vanilla acceptance gate. A research branch should retain a mode that reproduces this baseline exactly.
