# TinyMistral PyTorch Reference

TinyMistralFork is a small, auditable PyTorch implementation of
`M4-ai/TinyMistral-248M-v3`. It is the vanilla single-pass control
implementation for later architecture experiments.

It provides:

- Transformers 4.45.2-compatible Mistral semantics.
- Strict checkpoint compatibility and pinned external-weight verification.
- Sliding-window attention with GQA and RoPE.
- Dense reference, CUDA FlexAttention, and Apple MPS local attention backends.
- Rolling KV-cache decoding.
- Ordinary causal-language-model training.

It intentionally contains no MPTT, MemoryTape, latent feedback, cross-
attention memory, special memory tokens, multipass loss, or distributed
trainer. See [the design notes](docs/design.md) for the architecture and
research boundary.

## Target checkpoint

```text
M4-ai/TinyMistral-248M-v3
revision 5afbc96ddc964c68282cd970ef49e8d1a5e81c52
```

The target has 12 layers, hidden size 1024, intermediate size 4096, 32 query
heads, 8 KV heads, head dimension 32, vocabulary size 32,005, a 32-position
sliding window, and 248,024,064 parameters.

The validated vanilla tag is [`v0.2.0-vanilla-mps`](https://github.com/PeterBjerreHansen/TinyMistralFork/tree/v0.2.0-vanilla-mps).

## Install

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --locked
uv run pytest -q
```

`uv` reads `.python-version`, creates `.venv`, and installs the locked project
and development dependencies. You do not need to activate the virtual
environment.

For a minimal environment without test and comparison tools:

```bash
uv sync --no-dev
uv sync --no-dev --extra io
```

## Download and run

Download the pinned checkpoint and verify its exact structure, configuration,
parameter count, and external weight checksum:

```bash
make download
make verify
```

Generate on a Mac with MPS:

```bash
uv run python scripts/generate.py \
  "The meaning of life is" \
  --device mps \
  --dtype float16 \
  --backend auto \
  --max-new-tokens 40
```

The minimal generation API supports one prompt at a time (batch size 1).

## Attention backends

The three backends have the same mathematical semantics:

- `reference`: dense eager attention and the numerical correctness oracle.
- `flex`: block-sparse FlexAttention intended for CUDA.
- `local`: ordinary PyTorch O(TW) attention intended primarily for Apple MPS.

`auto` selects FlexAttention for compatible CUDA sequences, local attention for
compatible MPS sequences, and the reference path for CPU, cached decoding,
padding, and nonstandard positions.

For `sliding_window=W`, a key is visible when:

```text
key_position <= query_position
query_position - key_position < W
```

The 32-position window therefore includes the current token and at most 31
previous tokens in each layer.

## Validation

The Makefile is the command-line interface for the main checks:

```bash
make test
make download
make verify
make hf-check
make hf-layers
make cache-check
make hf-generation-check
make backend-check
make mps-smoke
make train-smoke
make baseline
make benchmark
```

The full acceptance criteria and observed CPU/MPS results are recorded in
[docs/validation.md](docs/validation.md). The most important oracles compare
complete hidden states and vocabulary logits against Transformers 4.45.2,
rolling-cache logits against full recomputation, and greedy generation through
repeated sliding-window eviction.

### MPS training dtype

FP16 is validated for MPS inference and attention checks. Direct FP16 AdamW
updates produced a non-finite post-update loss on the validated Mac/PyTorch
stack, so the vanilla training baseline uses FP32:

```bash
make train-smoke
make baseline
```

Any future mixed-precision training path needs an explicit numerical-stability
strategy and its own acceptance tests.

## Project layout

```text
src/tiny_mistral/   implementation
tests/              unit and model-contract tests
scripts/            download, validation, generation, training, benchmarks
configs/            checked-in target configuration snapshot
examples/           tiny offline baseline corpus
docs/               design, provenance, and validation records
```

## Status and research boundary

`main` is the vanilla control implementation. Future MPTT, MemoryTape, FBT,
or other recurrent-memory work should branch from the validated baseline and
retain a mode that reproduces its behavior exactly.

## License and provenance

The project is Apache-2.0 licensed. See [LICENSE](LICENSE), [NOTICE](NOTICE),
and [docs/design.md](docs/design.md) for upstream provenance and reference
details.
