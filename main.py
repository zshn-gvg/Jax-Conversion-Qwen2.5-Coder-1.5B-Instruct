import argparse

from torch_to_flax import MODEL_NAME


def convert(args):
    from torch_to_flax import torch_to_flax

    torch_to_flax(model_name=args.model, output_path=args.params)
    print(f"Saved Flax parameters to {args.params}")


def generate(args):
    import mmap

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.serialization import msgpack_restore
    from transformers import AutoConfig, AutoTokenizer

    from model_flax import Qwen2Config, Qwen2ForCausalLM, shard_params

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = Qwen2Config(**AutoConfig.from_pretrained(args.model).to_dict())
    model = Qwen2ForCausalLM(config)

    # Memory-map the file so it is not held in memory twice while decoding.
    with open(args.params, "rb") as f, mmap.mmap(
        f.fileno(), 0, access=mmap.ACCESS_READ
    ) as buf:
        params = shard_params({"params": msgpack_restore(buf)})

    if args.no_chat_template:
        text = args.prompt
    else:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = jnp.array(tokenizer(text, return_tensors="np")["input_ids"])

    output = model.generate(
        params,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature or 1.0,
        do_sample=args.temperature > 0,
        top_k=args.top_k,
        prng_key=jax.random.PRNGKey(args.seed),
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = np.array(output[0, input_ids.shape[1] :])
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2 models (default: Qwen2.5-Coder-1.5B-Instruct) in JAX/Flax."
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model id")
    parser.add_argument(
        "--params", default="flax_params.msgpack", help="Flax parameter file"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("convert", help="convert PyTorch weights to Flax")

    gen = commands.add_parser("generate", help="generate text from a prompt")
    gen.add_argument("prompt")
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument(
        "--temperature", type=float, default=0.0, help="0 means greedy decoding"
    )
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument(
        "--no-chat-template",
        action="store_true",
        help="pass the prompt as raw text instead of a chat message",
    )

    args = parser.parse_args()
    {"convert": convert, "generate": generate}[args.command](args)


if __name__ == "__main__":
    main()
