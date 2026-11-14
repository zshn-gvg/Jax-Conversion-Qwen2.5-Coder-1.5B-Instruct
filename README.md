# Qwen2.5-Coder-1.5B-Instruct in JAX/Flax

A JAX/Flax implementation of the Qwen2 architecture with a converter for the
[Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
PyTorch weights.

The Flax config is built from the Hugging Face config, so other Qwen2
checkpoints can be converted with `--model`. Sliding-window attention is
not implemented.

## Setup

```bash
uv sync
```

## Usage

Convert the PyTorch weights to `flax_params.msgpack`:

```bash
uv run main.py convert
```

Generate (greedy by default, stops at the end-of-sequence token):

```bash
uv run main.py generate "Write a Python function that checks if a number is prime."
```

Options: `--model`, `--params`, `--max-new-tokens`, `--temperature`,
`--top-k`, `--seed`, `--no-chat-template`. See `uv run main.py --help`.

From Python:

```python
from model_flax import Qwen2Config, Qwen2ForCausalLM, shard_params

model = Qwen2ForCausalLM(Qwen2Config())  # defaults match Qwen2.5-Coder-1.5B-Instruct
params = shard_params({"params": params})  # shard over all available devices
output_ids = model.generate(params, input_ids, max_new_tokens=100)
```

## Notes

- Weights are stored in float32 (~7 GB). `flax.serialization.to_bytes`
  holds several copies of the weights while serializing, so converting runs
  out of memory on a 15 GB machine.
- `generate` has no KV cache: every step reruns the full padded sequence.

## Colab

https://colab.research.google.com/drive/18rqkuTnvlYwnmh3GaMnXmFPwaCMiIhpA
