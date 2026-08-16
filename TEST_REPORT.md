# Vanilla baseline acceptance report

This report records the final local acceptance run on 2026-08-16 before freezing the vanilla TinyMistral baseline. The tag is intentionally still pending until these changes are committed and the resulting GitHub Actions run is green.

## Environment

```text
Machine architecture: arm64
macOS:               26.5.1 (25F80)
Python:              3.13.15
PyTorch:             2.13.0
safetensors:         0.8.0
pytest:              9.1.1
Transformers:        4.45.2
MPS built:           true
MPS available:       true
CUDA:                not applicable on this machine
```

`uv sync --locked` resolved 55 packages and checked the 30 installed packages without changing the lockfile. The pinned download helper then resolved all six checkpoint/tokenizer files from the fixed upstream revision.

## Local and checkpoint gates

- Python byte-compilation of `src/`, `scripts/`, and `tests/`: passed.
- Unit suite: **43 passed**.
- Strict target checkpoint verification: no missing keys, unexpected keys, shape mismatches, recognized config mismatches, or weight checksum mismatch.
- Actual and expected parameter counts: **248,024,064**.
- Pinned `model.safetensors` SHA-256: `9432ee6e0681473a9ed513e43362d9911832f9a5c7faded76f46ec66c55a9d3b`.
- Short CPU FP32 Hugging Face logits smoke test over the complete vocabulary: exact equality.
- Mandatory 40-token layer oracle across the 32-token sliding-window boundary: exact equality for every element of all 13 hidden-state snapshots and every vocabulary logit.
- Real-checkpoint 96-token rolling-cache parity: maximum absolute difference `3.3378601e-05`, mean absolute difference `3.7221016e-06`.
- Greedy Hugging Face generation oracle: exact token equality, with both implementations producing the complete requested shape `(1, 72)` (8 prompt tokens plus 64 generated tokens).

## Apple MPS gates

- FP16 local-window micro-model forward/backward smoke test: passed with exact forward equality; loss was `5.549587`.
- FP16 real-weight local versus reference comparison: passed at lengths 31, 32, 33, 64, and 129 while comparing every vocabulary logit. The largest observed maximum absolute difference was `0.01953125`, within the declared FP16 tolerance.
- FP32 real-weight optimizer smoke test: passed. Loss changed from `13.341655` to `13.251093`, with finite gradients and finite post-update loss.
- FP16 deterministic real-weight generation: completed successfully for 40 requested tokens.
- FP32 ten-step ordinary continued-pretraining baseline: passed with finite losses and gradient norms at every step. First loss was `5.505820`; final loss was `5.174341`.
- FP16 attention scaling sanity check: local backend completed at lengths 128, 256, 512, 1024, and 2048. Observed local timings were 2.62, 2.23, 4.82, 5.08, and 9.04 ms respectively.

For the short lengths where both paths were timed, the dense reference backend was faster (1.61, 1.63, and 2.87 ms through length 512). The benchmark is therefore evidence of bounded local-window scaling and successful long-sequence execution, not a claim that the local backend is faster at every length.

## Known training limitation

Direct FP16 AdamW training on this Mac/PyTorch stack produced a non-finite loss after the optimizer update even when the preceding forward loss and gradients were finite. The smoke test now detects this failure. FP32 is the accepted MPS training dtype for the vanilla baseline and initial MPTT work; FP16 remains validated for inference and attention comparisons. Any later mixed-precision training path must add an explicit numerical-stability strategy and its own acceptance tests.

## Freeze status

All applicable local CPU and MPS acceptance gates are green with FP32 designated for training. Before creating `v0.2.0-vanilla-mps`, commit the reviewed changes, confirm the GitHub Actions run for that exact commit is green, and then tag that commit. Future MPTT work should branch from the tag and retain a mode that reproduces this vanilla baseline.
