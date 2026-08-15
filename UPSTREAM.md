# Upstream provenance

This repository is intentionally pinned to a reproducible vanilla reference point.

## Target checkpoint

- Repository: `M4-ai/TinyMistral-248M-v3`
- Pinned revision: `5afbc96ddc964c68282cd970ef49e8d1a5e81c52`
- Architecture declared by checkpoint: `MistralForCausalLM`
- Checkpoint `transformers_version`: `4.45.2`
- Weight file: `model.safetensors` (not redistributed here)

The download helper uses the pinned revision rather than following `main`.

## Source reference

The mathematical/modeling reference is Hugging Face Transformers tag `v4.45.2`:

- `src/transformers/models/mistral/modeling_mistral.py`
- `src/transformers/models/mistral/configuration_mistral.py`
- `src/transformers/modeling_attn_mask_utils.py`

The local implementation intentionally omits Hugging Face framework machinery
(`PreTrainedModel`, `GenerationMixin`, generic cache classes, output classes,
multiple task heads, etc.) while preserving the decoder architecture, parameter
names/shapes, RoPE, GQA, RMSNorm, MLP, and v4.45.2 sliding-window semantics.

## Sliding-window convention

Transformers v4.45.2 Mistral's `_update_causal_mask` masks keys satisfying
`key_position <= query_position - sliding_window`. Equivalently, an allowed key
must have `query_position - key_position < sliding_window`. Thus
`sliding_window=32` permits 32 total positions after warm-up: the current token
plus up to 31 preceding positions. The local reference and FlexAttention path use
the same convention.
