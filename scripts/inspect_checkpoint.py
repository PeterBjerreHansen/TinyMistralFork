#!/usr/bin/env python
from __future__ import annotations

import argparse

from tiny_mistral.config import MistralConfig
from tiny_mistral.loading import checkpoint_tensor_metadata, expected_state_metadata, math_prod


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", nargs="?", default="checkpoints/TinyMistral-248M-v3")
    args = p.parse_args()
    config = MistralConfig.from_json_file(f"{args.model_dir}/config.json")
    actual = checkpoint_tensor_metadata(f"{args.model_dir}/model.safetensors")
    expected = expected_state_metadata(config)
    print(f"checkpoint tensors: {len(actual)}")
    print(f"expected tensors:   {len(expected)}")
    print(f"expected params:    {sum(math_prod(s) for s in expected.values()):,}")
    for key in sorted(actual):
        shape, dtype = actual[key]
        print(f"{key:65s} {str(shape):24s} {dtype}")


if __name__ == "__main__":
    main()
