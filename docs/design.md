# Design and provenance

## Purpose

TinyMistralFork is a vanilla PyTorch reference implementation of
`M4-ai/TinyMistral-248M-v3`. It exists to provide a small, auditable control
implementation whose checkpoint loading, masking, attention, caching,
generation, and autograd behavior can be compared against later experiments.

This baseline contains no MPTT, latent feedback, MemoryTape, cross-attention
memory, special memory tokens, multipass loss, or distributed trainer.

## Pinned upstream

- Repository: `M4-ai/TinyMistral-248M-v3`
- Revision: `5afbc96ddc964c68282cd970ef49e8d1a5e81c52`
- Architecture: `MistralForCausalLM`
- Transformers reference: `4.45.2`
- Weight file SHA-256:
  `9432ee6e0681473a9ed513e43362d9911832f9a5c7faded76f46ec66c55a9d3b`

The mathematical and modeling reference is the Hugging Face Transformers
`v4.45.2` implementation. The local module hierarchy preserves the original
checkpoint parameter names, so strict safetensors loading requires no renaming
or conversion step.

## Architecture

The target has 12 decoder layers, hidden size 1024, intermediate size 4096,
32 query heads, 8 KV heads, head dimension 32, vocabulary size 32,005, and a
32-position sliding window.

The implementation preserves the standard Mistral structure: RMSNorm,
pre-normalized residual blocks, RoPE, grouped-query attention, and a gated
SwiGLU-style MLP.

## Attention backends

Three attention implementations share the same semantics:

- `reference`: dense eager PyTorch attention used as the numerical oracle.
- `flex`: block-sparse FlexAttention intended for CUDA.
- `local`: ordinary PyTorch O(TW) attention intended primarily for Apple MPS.

`auto` selects FlexAttention for compatible unpadded CUDA sequences, local
attention for compatible unpadded MPS sequences, and the reference path for
CPU, cached decoding, padding, and nonstandard positions.

For `sliding_window=W`, a key is visible when:

```text
key_position <= query_position
query_position - key_position < W
```

Thus a window of 32 includes the current position plus at most 31 previous
positions. Information can still propagate farther through successive layers.

## Cached decoding

Each layer retains only the `sliding_window - 1` previous K/V states needed for
the next token. The cache stores already-RoPE-rotated K/V tensors and a scalar
contiguous position range. Cached calls therefore require contiguous absolute
positions shared across the batch.

## MPS and training

FP16 is the default MPS dtype for inference and attention checks. On the
validated Mac/PyTorch stack, direct FP16 AdamW updates can produce non-finite
weights even when the preceding loss and gradients are finite. The vanilla
training acceptance baseline therefore uses FP32 on MPS. Any later mixed-
precision training path needs an explicit numerical-stability design.

## Research boundary

Future recurrence, memory, FBT, and MPTT work should branch from the vanilla
baseline. Experimental branches should retain a mode that reproduces this
implementation exactly.
