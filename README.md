# TinyMistral PyTorch Reference

A small, auditable PyTorch implementation of **`M4-ai/TinyMistral-248M-v3`** intended to be frozen as a vanilla reference point before any research modifications.

**This repository intentionally contains no MPTT, latent feedback, MemoryTape, cross-attention memory, special memory tokens, or multipass loss.** The purpose of this repo is to establish a trustworthy single-pass baseline that can later be forked for experiments.

## Target checkpoint

The pinned checkpoint is:

```text
M4-ai/TinyMistral-248M-v3
revision 5afbc96ddc964c68282cd970ef49e8d1a5e81c52
```

The checkpoint declares a Mistral causal LM with:

```text
layers                 12
hidden size            1024
intermediate size      4096
query heads            32
KV heads               8
head dimension         32
vocabulary size        32005
sliding window         32
max positions          32768
parameter count        248,024,064
transformers reference 4.45.2
```

The local module hierarchy preserves the original checkpoint parameter names, so `model.safetensors` is loaded with `strict=True` and no weight-renaming/conversion step.

## Sliding-window semantics

The repository follows the Transformers 4.45.2 Mistral convention exactly. For `sliding_window=W`, a key is visible when:

```text
key_position <= query_position
query_position - key_position < W
```

Thus `sliding_window=32` means the current position plus at most 31 previous positions are directly visible in each attention layer.

A per-layer window of 32 is **not** a 32-token end-to-end receptive field: information can propagate farther through successive decoder layers.

## Attention backends

The model has three attention implementations with the same mathematical semantics.

### `reference`

An explicit eager PyTorch implementation that materializes the dense `T x T` score matrix. It is intentionally simple and is the numerical correctness oracle. It is not intended for long-sequence training.

### `flex`

PyTorch FlexAttention with a block-sparse causal sliding-window mask. This is the preferred full-sequence backend on CUDA. It keeps K/V in grouped-query form and exploits the local attention structure instead of merely applying a dense mask.

### `local`

A PyTorch-only exact sliding-window implementation intended primarily for **Apple MPS**. It physically constructs only each query's local K/V window and forms a score tensor of shape approximately:

```text
[B, Hkv, query_groups, T, W]
```

rather than `[B, Hq, T, T]`. Its attention-score work/storage therefore scales as `O(T*W)` for fixed window `W`. GQA is handled without repeating the K/V sequence across all query heads.

The local implementation uses ordinary `pad`, `unfold`, `matmul`, softmax, and reshape/permute operations, making it usable on CPU and MPS without a custom CUDA/Triton kernel.

### `auto`

`auto` selects:

```text
CUDA full unpadded sequence -> flex
MPS  full unpadded sequence -> local
CPU                        -> reference
cached decoding            -> reference over the bounded rolling KV cache
padded/nonstandard positions -> reference
```

Cached autoregressive decoding is already cheap: each layer retains only the `sliding_window - 1` previous K/V states needed for the next token.

---

# Installation

The project supports Python 3.10 through 3.13 and pins Python 3.13 for local development. Install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync
uv run pytest -q
```

`uv sync` reads `.python-version`, installs Python 3.13 if needed, creates `.venv`, and installs the
locked project and development dependencies. You do not need to activate the virtual environment.
For a minimal environment without test and comparison tools, use:

```bash
uv sync --no-dev
```

To add the optional Hugging Face download/tokenizer support to that minimal environment, use
`uv sync --no-dev --extra io`.

## Apple Silicon / MPS check

On a supported Mac:

```bash
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY
```

Then run the hardware-only micro-model smoke test:

```bash
uv run python scripts/mps_smoke.py
```

This requires no TinyMistral checkpoint. It runs a small FP16 Mistral model on MPS, compares `local` with the reference attention path, and performs a backward pass.

---

# Validation sequence

The recommended order is deliberately conservative.

## 1. Offline unit tests

```bash
uv run pytest -q
```

These use tiny randomly initialized models and require no checkpoint download. They cover:

- target configuration and parameter count
- RMSNorm and RoPE
- exact sliding-window boundary semantics
- grouped-query attention
- reference vs FlexAttention
- reference vs MPS-oriented local-window attention
- local-attention gradients vs reference gradients
- strict safetensors round-trip loading
- full-forward vs rolling-cache decoding
- causal-LM loss/backward/optimizer plumbing
- generation plumbing
- device/dtype resolution
- an MPS hardware test that is automatically skipped when MPS is unavailable

## 2. Download the pinned checkpoint

```bash
uv run python scripts/download_checkpoint.py
```

The weights are not included in the repository. They are written under:

```text
checkpoints/TinyMistral-248M-v3/
```

## 3. Verify checkpoint structure

```bash
uv run python scripts/verify_checkpoint.py
```

This checks tensor names, tensor shapes, the complete recognized config against the pinned TinyMistral-248M-v3 config, the exact 248,024,064 parameter count, and the pinned safetensors SHA-256 before the full model is moved to an accelerator. The library also retains a generic structural verifier for internally consistent non-target checkpoints used by tests.

Optional tensor inspection:

```bash
uv run python scripts/inspect_checkpoint.py
```

## 4. Compare against Transformers 4.45.2

Start with the short CPU FP32 logits smoke test:

```bash
uv run python scripts/compare_hf.py --device cpu --dtype float32
```

The mandatory architecture oracle compares every hidden channel and every vocabulary logit over 40 positions, deliberately crossing the 32-token sliding window:

```bash
uv run python scripts/compare_hf_layers.py --device cpu --dtype float32
```

The mandatory real-checkpoint rolling-cache parity gate compares 96 token-by-token logits with full recomputation and exercises multiple sliding-window evictions:

```bash
uv run python scripts/compare_cache.py --device cpu --dtype float32 --length 96
```

The mandatory 64-token greedy-generation oracle compares the rolling-cache output directly with Transformers 4.45.2 and asserts that all 64 tokens were produced, ensuring repeated window eviction:

```bash
uv run python scripts/compare_hf_generation.py --max-new-tokens 64
```

The Hugging Face oracle uses eager attention intentionally. These are correctness checks, not performance runs.

## 5. Check the optimized backend with real weights

### Apple MPS

```bash
uv run python scripts/compare_backends.py \
  --device mps \
  --dtype float16 \
  --optimized-backend local
```

### CUDA

```bash
uv run python scripts/compare_backends.py \
  --device cuda \
  --dtype bfloat16 \
  --optimized-backend flex
```

The default lengths straddle the 32-token attention-window boundary.

---

# Generation

## Apple MPS

```bash
uv run python scripts/generate.py \
  "The meaning of life is" \
  --device mps \
  --dtype float16 \
  --backend auto \
  --max-new-tokens 40
```

Because `auto` is used, prompt prefill uses the `local` O(TW) backend on MPS and subsequent token-by-token generation uses the rolling local KV cache.

The minimal `generate()` API intentionally supports one prompt at a time (batch size 1). Generate
batched prompts independently when they need separate EOS stopping behavior.

For deterministic greedy generation, leave `--temperature 0`. For sampling:

```bash
uv run python scripts/generate.py \
  "Once upon a time" \
  --device mps \
  --dtype float16 \
  --temperature 0.8 \
  --top-k 50 \
  --max-new-tokens 80
```

## CUDA

```bash
uv run python scripts/generate.py \
  "The meaning of life is" \
  --device cuda \
  --dtype bfloat16 \
  --backend auto
```

You can also use `--device auto --dtype auto`; this prefers CUDA, then MPS, then CPU. The automatic dtype is BF16/FP16 on CUDA depending on hardware support, FP16 on MPS, and FP32 on CPU.

---

# Baseline training checks

There are two different training scripts on purpose.

## One-step engineering smoke test

This uses random token IDs and proves that the **real 248M checkpoint** supports forward, backward, finite gradients, and an AdamW update in this implementation.

### MPS

```bash
uv run python scripts/train_smoke.py \
  --device mps \
  --dtype float32 \
  --backend auto \
  --seq-len 64
```

### CUDA

```bash
uv run python scripts/train_smoke.py \
  --device cuda \
  --dtype bfloat16 \
  --backend auto \
  --seq-len 64
```

The validated MPS training baseline uses FP32. Direct FP16 AdamW updates on the tested Mac/PyTorch stack produced a non-finite post-update loss, so FP16 is retained for inference and attention validation but is not claimed as a stable training mode. If MPS unified memory is tight, lower `--seq-len` to 32.

## Small ordinary continued-pretraining baseline

`scripts/train_baseline.py` performs **single-pass causal language-model training only**. It does not contain MPTT or any recurrent mechanism. An intentionally tiny local corpus is included under `examples/baseline_corpus.txt` so the pipeline can be exercised offline after downloading the model/tokenizer.

On MPS:

```bash
uv run python scripts/train_baseline.py \
  --device mps \
  --dtype float32 \
  --backend auto \
  --steps 10 \
  --seq-len 128 \
  --batch-size 1 \
  --lr 1e-6
```

This updates **all 248M parameters**. The bundled corpus and ten-step run are only a systems baseline; they are not a meaningful language-model training recipe or benchmark.

To use your own local text:

```bash
uv run python scripts/train_baseline.py \
  --text-file /path/to/corpus.txt \
  --device mps \
  --dtype float32 \
  --steps 100 \
  --seq-len 256
```

For MPS memory problems, reduce parameters in this order:

```text
batch size -> sequence length -> number of steps
```

The number of steps affects runtime rather than peak memory, but reducing it is useful while validating a new machine.

---

# Attention benchmark

On Apple MPS:

```bash
uv run python scripts/benchmark_attention.py \
  --device mps \
  --dtype float16 \
  --lengths 128 256 512 1024 2048
```

By default this compares `local` against the dense reference only through sequence length 512, while continuing the local path at longer lengths.

On CUDA:

```bash
uv run python scripts/benchmark_attention.py \
  --device cuda \
  --dtype bfloat16
```

The benchmark is intended to catch gross scaling regressions; exact timings depend strongly on the PyTorch version, GPU, compilation cache, and sequence length.

---

# Manual loading

```python
import torch
from tiny_mistral import load_model

model = load_model(
    "checkpoints/TinyMistral-248M-v3",
    attention_backend="auto",
    device="mps",
    dtype=torch.float16,
).eval()

ids = torch.tensor([[1, 42, 17]], device="mps")
with torch.no_grad():
    out = model(ids, use_cache=False)
print(out.logits.shape)
```

The loader constructs the model on the `meta` device, performs a strict `assign=True` safetensors load, materializes the nonpersistent RoPE buffers, casts on CPU when requested, and then transfers the model to the requested device. Casting before the MPS transfer avoids depending on native checkpoint BF16 support on the local Apple/PyTorch combination.

---

# Acceptance gate before MPTT work

Do not add experimental model architecture code until the intended Mac passes every applicable command below:

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/mps_smoke.py
uv run python scripts/verify_checkpoint.py
uv run python scripts/compare_hf_layers.py --device cpu --dtype float32
uv run python scripts/compare_cache.py --device cpu --dtype float32 --length 96
uv run python scripts/compare_hf_generation.py --max-new-tokens 64
uv run python scripts/compare_backends.py --device mps --dtype float16 --optimized-backend local
uv run python scripts/train_smoke.py --device mps --dtype float32 --backend auto --seq-len 64
uv run python scripts/generate.py "The meaning of life is" --device mps --dtype float16 --backend auto --max-new-tokens 40
uv run python scripts/train_baseline.py --device mps --dtype float32 --backend auto --steps 10 --seq-len 128 --batch-size 1 --lr 1e-6
uv run python scripts/benchmark_attention.py --device mps --dtype float16 --lengths 128 256 512 1024 2048
```

The short `compare_hf.py` command remains useful as a quick smoke test, but the full layer oracle above is the required gate. CUDA users should substitute the CUDA/BF16/Flex checks documented earlier.

Once green, record the exact environment and results in `TEST_REPORT.md`, tag the repository—for example `v0.2.0-vanilla-mps`—and branch future FBT/MemoryTape/MPTT work from that immutable reference point. A research branch must retain a mode that reproduces this vanilla baseline.

## Non-goals

- no MPTT/multipass controller
- no FBT cross operator
- no MemoryTape
- no cross-attention memory
- no memory tokens
- no distributed trainer
- no large-scale data pipeline
- no attempt to reproduce every Hugging Face generation option
- no bundled model weights

The austerity is deliberate. Later experimental variants should be able to point back to a small, understandable and tested vanilla implementation.
