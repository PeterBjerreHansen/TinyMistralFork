import warnings

import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralAttention


def test_flex_matches_reference_small_cpu():
    cfg = micro_config(sliding_window=4)
    ref = MistralAttention(cfg, 0, attention_backend="reference")
    flex = MistralAttention(cfg, 0, attention_backend="flex", compile_flex=False, flex_block_size=16)
    flex.load_state_dict(ref.state_dict())
    ref.eval(); flex.eval()
    x = torch.randn(2, 17, cfg.hidden_size)
    pos = torch.arange(17)[None, :].expand(2, -1)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yr, _ = ref(x, attention_mask=None, position_ids=pos, use_cache=False)
        yf, _ = flex(x, attention_mask=None, position_ids=pos, use_cache=False)
    torch.testing.assert_close(yf, yr, atol=2e-5, rtol=2e-5)


def test_padding_forces_correct_reference_fallback():
    cfg = micro_config(sliding_window=4)
    a = MistralAttention(cfg, 0, attention_backend="reference")
    b = MistralAttention(cfg, 0, attention_backend="flex", compile_flex=False)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 8, cfg.hidden_size)
    pos = torch.arange(8)[None, :]
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        ya, _ = a(x, attention_mask=mask, position_ids=pos)
        yb, _ = b(x, attention_mask=mask, position_ids=pos)
    torch.testing.assert_close(ya, yb)
