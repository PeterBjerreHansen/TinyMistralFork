import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM


def test_incremental_cache_matches_full_forward_at_each_position():
    cfg = micro_config(num_hidden_layers=2, sliding_window=4)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 13))
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        cache = None
        pieces = []
        for t in range(ids.shape[1]):
            out = model(ids[:, t:t+1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            pieces.append(out.logits)
            assert cache is not None
            for layer_cache in cache:
                assert layer_cache.seq_len <= max(cfg.sliding_window - 1, 0)
        inc = torch.cat(pieces, dim=1)
    torch.testing.assert_close(inc, full, atol=3e-5, rtol=3e-5)


def test_cached_decoding_rejects_noncontiguous_position_ids():
    cfg = micro_config(num_hidden_layers=2, sliding_window=4)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        first = model(ids[:, :1], use_cache=True)
        with pytest.raises(ValueError, match="contiguous"):
            model(
                ids[:, 1:3],
                past_key_values=first.past_key_values,
                position_ids=torch.tensor([[1, 3]]),
                use_cache=True,
            )


@pytest.mark.parametrize(
    "position_ids",
    [
        torch.tensor([[10, 12, 13]]),
        torch.tensor([[10, 11, 12], [20, 21, 22]]),
    ],
)
def test_initial_cache_rejects_positions_it_cannot_represent(position_ids):
    cfg = micro_config(num_hidden_layers=2, sliding_window=4)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, position_ids.shape)
    with pytest.raises(ValueError, match="contiguous.*shared"):
        model(ids, position_ids=position_ids, use_cache=True)


def test_initial_cache_accepts_shared_contiguous_nonzero_positions():
    cfg = micro_config(num_hidden_layers=2, sliding_window=4)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 3))
    position_ids = torch.tensor([[10, 11, 12], [10, 11, 12]])
    with torch.no_grad():
        out = model(ids, position_ids=position_ids, use_cache=True)
    assert out.past_key_values is not None
    assert {cache.next_position for cache in out.past_key_values} == {13}


def test_noncontiguous_positions_remain_available_without_cache():
    cfg = micro_config(num_hidden_layers=2, sliding_window=4)
    model = MistralForCausalLM(cfg, attention_backend="reference").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        out = model(
            ids,
            position_ids=torch.tensor([[10, 12, 13]]),
            use_cache=False,
        )
    assert out.logits.shape == (1, 3, cfg.vocab_size)
    assert out.past_key_values is None
