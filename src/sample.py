"""Generate text from a trained checkpoint.

Free-form continuation:
    python src/sample.py --ckpt ~/mini_llm_runs/pretrain/best.pt --prompt "The history of the bicycle"

Interactive support chat (use a fine-tuned checkpoint):
    python src/sample.py --ckpt ~/mini_llm_runs/support/best.pt --chat
"""

from __future__ import annotations

import argparse

import tiktoken
import torch

from chat_format import NEXT_TURN, Turn, build_prompt
from model import GPT, GPTConfig
from train import load_checkpoint


def load_model(ckpt_path: str, device: torch.device) -> GPT:
    checkpoint = load_checkpoint(ckpt_path)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def generate_text(
    model: GPT,
    encoder: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    """Return only the newly generated text, cut at <|endoftext|>."""
    device = next(model.parameters()).device
    # An empty prompt starts from <|endoftext|>, i.e. "beginning of a new document".
    # Long prompts are fine: generate() only feeds the model the last block_size tokens.
    prompt_tokens = encoder.encode_ordinary(prompt) or [encoder.eot_token]
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, stop_token=encoder.eot_token)
    new_tokens = [t for t in out[0, len(prompt_tokens) :].tolist() if t < encoder.n_vocab]
    if encoder.eot_token in new_tokens:
        new_tokens = new_tokens[: new_tokens.index(encoder.eot_token)]
    return encoder.decode(new_tokens)


def extract_agent_reply(generated: str) -> str:
    """The model may keep writing further turns (customer or agent); keep only this reply."""
    return NEXT_TURN.split(generated, maxsplit=1)[0].strip()


def chat_loop(model: GPT, encoder: tiktoken.Encoding, args: argparse.Namespace) -> None:
    print("Support chat. Type a message as the customer; empty line or Ctrl-D to quit.")
    history: list[Turn] = []
    while True:
        try:
            message = input("\nCustomer: ").strip()
        except EOFError:
            break
        if not message:
            break
        generated = generate_text(
            model, encoder, build_prompt(history, message), args.max_new_tokens, args.temperature, args.top_k
        )
        reply = extract_agent_reply(generated)
        print(f"Agent: {reply}")
        history += [("customer", message), ("agent", reply)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt")
    mode.add_argument("--chat", action="store_true")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model = load_model(args.ckpt, torch.device(args.device))
    encoder = tiktoken.get_encoding("gpt2")
    if args.chat:
        chat_loop(model, encoder, args)
        return
    for i in range(args.num_samples):
        text = generate_text(model, encoder, args.prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(f"--- sample {i + 1} ---\n{args.prompt}{text}")


if __name__ == "__main__":
    main()
