.PHONY: test compile download verify hf-check hf-layers hf-generation-check backend-check cache-check mps-smoke train-smoke baseline benchmark

test:
	uv run pytest -q

compile:
	uv run python -m compileall -q src scripts tests

download:
	uv run python scripts/download_checkpoint.py

verify:
	uv run python scripts/verify_checkpoint.py

hf-check:
	uv run python scripts/compare_hf.py --device cpu --dtype float32

hf-layers:
	uv run python scripts/compare_hf_layers.py --device cpu --dtype float32

hf-generation-check:
	uv run python scripts/compare_hf_generation.py --max-new-tokens 64

backend-check:
	uv run python scripts/compare_backends.py

cache-check:
	uv run python scripts/compare_cache.py --device cpu --dtype float32 --length 96

mps-smoke:
	uv run python scripts/mps_smoke.py

train-smoke:
	uv run python scripts/train_smoke.py --device auto --dtype auto --backend auto --seq-len 64

baseline:
	uv run python scripts/train_baseline.py --device auto --dtype auto --backend auto --steps 10 --seq-len 128

benchmark:
	uv run python scripts/benchmark_attention.py --device auto --dtype auto
