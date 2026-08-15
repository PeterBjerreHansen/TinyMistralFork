# Build/test report

This repository was rebuilt and tested in the ChatGPT execution environment before the archive was produced.

## Environment used for offline checks

```text
Python:      3.13.5
PyTorch:     2.10.0+cpu
safetensors: 0.7.0
pytest:      9.0.2
CUDA:        unavailable
MPS:         build unavailable in this Linux execution environment
```

## Completed checks

- Python byte-compilation of `src/`, `scripts/`, and `tests/`.
- Editable-package installation/import with no dependency download.
- **33 offline tests passed; 1 MPS-hardware test skipped** because this environment is not macOS/Apple Silicon.
- Exact target architecture metadata: **111 state-dict tensors**, **248,024,064 parameters**.
- Reference attention versus FlexAttention on small CPU inputs.
- Reference attention versus the new `local` O(TW) attention backend over sequence lengths 1, 2, 3, 4, 5, 17, and 33.
- Full-model reference versus local-attention equivalence on a micro Mistral configuration.
- Local-attention backward gradients versus reference-attention gradients.
- Padded-input fallback from local/Flex paths to the reference implementation.
- Sliding-window boundary test for the Transformers 4.45.2 convention.
- Rolling local KV-cache output versus full recomputation.
- Safetensors strict-load round trip through the meta-device loader.
- `input_ids` versus `inputs_embeds` equivalence.
- Forward/loss/backward/finite-gradient/optimizer-step smoke test on a micro configuration.
- Greedy generation plumbing on a micro configuration.
- Device/dtype-resolution tests, including FP16 as the automatic MPS dtype.
- Every command-line script that accepts options parses successfully with `--help`.
- CPU attention scaling sanity check: at T=1024 the local backend completed without allocating the dense reference path; at T=512 it was materially faster than the dense reference in this CPU environment. Timing is not treated as a portable benchmark.

## MPS checks supplied but not executable here

The repository now contains `scripts/mps_smoke.py`. On an actual Apple Silicon/MPS machine it checks:

1. FP16 micro-model construction on MPS.
2. `local` attention output versus the dense reference path on the same MPS device.
3. Backpropagation through the local backend.

The MPS-specific test in `tests/test_mps.py` performs a similar forward/backward check and is automatically skipped when MPS is unavailable.

Because this execution environment is Linux/CPU-only, I cannot truthfully claim that the MPS Metal kernels themselves have been executed here. That is the first local-machine gate to run.

## Real-checkpoint integration checks intentionally left for the local machine

The sandbox used to construct this repository does not have the TinyMistral checkpoint/tokenizer available locally and does not provide general package/model downloads. Therefore the following real-weight checks are supplied but could not be executed here:

```bash
python scripts/download_checkpoint.py
python scripts/verify_checkpoint.py
python scripts/compare_hf.py --device cpu --dtype float32
python scripts/compare_hf_layers.py --device cpu --dtype float32
python scripts/compare_backends.py --device mps --dtype float16 --optimized-backend local
python scripts/train_smoke.py --device mps --dtype float16 --backend auto --seq-len 64
python scripts/generate.py "The meaning of life is" --device mps --dtype float16 --backend auto
python scripts/train_baseline.py --device mps --dtype float16 --backend auto --steps 10 --seq-len 128
python scripts/benchmark_attention.py --device mps --dtype float16
```

The repository should not be tagged as the immutable vanilla baseline until the applicable real-checkpoint and accelerator checks pass on the intended experiment machine.
