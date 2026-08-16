# Vanilla baseline validation

This document records the acceptance criteria and observed results for the
vanilla TinyMistral reference baseline.

Validated implementation commit:
`e44420d4190b6cfc1dc002c0ac67e364ef2f2de1`

The acceptance suite was rerun on `main` at this implementation state. The
documentation update that records these results is intentionally a separate
docs-only commit.

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

`uv sync --locked` resolved 55 packages and checked the 30 installed packages.
The pinned download helper resolved all six checkpoint/tokenizer files from
the fixed upstream revision.

## Architecture oracle

- 52 tests passed.
- Strict target verification found no missing keys, unexpected keys, shape
  mismatches, config mismatches, or weight checksum mismatch.
- Actual and expected parameter count: **248,024,064**.
- Pinned `model.safetensors` SHA-256:
  `9432ee6e0681473a9ed513e43362d9911832f9a5c7faded76f46ec66c55a9d3b`.
- Complete CPU FP32 Hugging Face logits matched exactly.
- The 40-token layer oracle matched every hidden channel and every vocabulary
  logit across the 32-token window boundary.
- The real-checkpoint `inputs_embeds` oracle matched every hidden state, full
  vocabulary logit, and input-embedding gradient exactly against Transformers
  4.45.2.

## Cache and generation oracles

- 96-token rolling-cache parity passed with maximum absolute difference
  `3.3378601e-05` and mean absolute difference `3.7221016e-06`.
- Greedy generation matched Transformers exactly. Both implementations
  produced `(1, 72)`: 8 prompt tokens plus all 64 requested tokens.
- The optimized generation prefill projects only the final prompt hidden state;
  the public full-sequence forward behavior is unchanged.

## Apple MPS validation

- FP16 local-window micro-model forward/backward smoke test passed with exact
  forward equality and finite loss.
- `backend-check` resolved `auto` to the MPS `local` backend and passed against
  the reference path at lengths 31, 32, 33, 64, and 129.
- FP16 real-weight local/reference parity passed at lengths 31, 32, 33, 64,
  and 129. The largest maximum absolute difference was `0.01953125`, within
  the declared FP16 tolerance.
- FP32 real-weight optimizer smoke test passed. Loss changed from `13.532436`
  to `13.439516`, with finite gradients and post-update loss.
- FP16 deterministic real-weight generation completed for 40 tokens.
- FP32 ten-step ordinary causal-LM baseline passed with finite losses and
  gradient norms. First loss was `5.505820`; final loss was `5.174341`.
- FP16 local attention completed at lengths 128, 256, 512, 1024, and 2048.
  Local timings were 2.74, 2.62, 3.59, 5.06, and 8.89 ms respectively.

At lengths where both paths were timed, dense reference attention was faster
(1.38, 1.50, and 2.59 ms through length 512). The benchmark demonstrates
bounded local-window scaling and long-sequence execution; it is not a claim
that local attention is faster at every short length.

## Known numerical limitation

Direct FP16 AdamW training on this Mac/PyTorch stack produced a non-finite
loss after the optimizer update even when the preceding forward loss and
gradients were finite. FP32 is therefore the accepted MPS training dtype for
the vanilla baseline and initial MPTT work. FP16 remains validated for
inference and attention comparisons.

## Reproducing the gates

The Makefile provides the main commands:

```bash
make test
make download
make verify
make hf-check
make hf-layers
make hf-embeds-check
make cache-check
make hf-generation-check
make backend-check
make mps-smoke
make train-smoke
make baseline
make benchmark
```

The complete acceptance sequence is documented in the repository README.
