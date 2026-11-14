"""Compares the Flax model against Hugging Face transformers on a small,
randomly initialised Qwen2 model (no download needed).

Run with: uv run python tests/test_parity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import numpy as np
import torch
from transformers import Qwen2Config as HFQwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from model_flax import Qwen2Config, Qwen2ForCausalLM
from torch_to_flax import convert_model_cpu


def make_models(tie_word_embeddings=False):
    hf_config = HFQwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=1000000.0,
        tie_word_embeddings=tie_word_embeddings,
    )
    torch.manual_seed(0)
    torch_model = HFQwen2ForCausalLM(hf_config).eval()
    flax_model = Qwen2ForCausalLM(Qwen2Config(**hf_config.to_dict()))
    params = {"params": convert_model_cpu(flax_model, torch_model)}
    return torch_model, flax_model, params


def test_logits_match():
    for tie in (False, True):
        torch_model, flax_model, params = make_models(tie)
        ids = np.random.RandomState(0).randint(0, 128, (1, 10))
        with torch.no_grad():
            expected = torch_model(torch.tensor(ids)).logits.numpy()
        actual = flax_model.apply(params, input_ids=jnp.array(ids))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_padded_batch_logits_match():
    torch_model, flax_model, params = make_models()
    ids = np.random.RandomState(0).randint(0, 128, (2, 6))
    for mask in ([[1] * 6, [1] * 4 + [0] * 2], [[1] * 6, [0] * 2 + [1] * 4]):
        mask = np.array(mask)
        position_ids = np.clip(mask.cumsum(-1) - 1, 0, None)
        with torch.no_grad():
            expected = torch_model(
                torch.tensor(ids),
                attention_mask=torch.tensor(mask),
                position_ids=torch.tensor(position_ids),
            ).logits.numpy()
        actual = flax_model.apply(
            params,
            input_ids=jnp.array(ids),
            attention_mask=jnp.array(mask),
            position_ids=jnp.array(position_ids),
        )[0]
        valid = mask.astype(bool)
        np.testing.assert_allclose(
            np.asarray(actual)[valid], expected[valid], atol=1e-4
        )


def test_greedy_generate_matches():
    torch_model, flax_model, params = make_models()
    ids = np.random.RandomState(0).randint(0, 128, (1, 5))
    expected = torch_model.generate(
        torch.tensor(ids), max_new_tokens=10, do_sample=False
    ).numpy()
    actual = flax_model.generate(params, jnp.array(ids), max_new_tokens=10)
    np.testing.assert_array_equal(np.asarray(actual), expected)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
