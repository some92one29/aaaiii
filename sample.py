import argparse
import json
from pathlib import Path

import torch

from model import TinyLM


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(tokens: list[int], itos: dict[str, str]) -> str:
    return "".join(itos[str(token)] if str(token) in itos else itos[token] for token in tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained tiny language model.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--start", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))

    stoi = meta["stoi"]
    itos = meta["itos"]

    model = TinyLM(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt = args.start
    prompt_ids = encode(prompt, stoi)
    x = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    print(decode(y[0].tolist(), itos))


if __name__ == "__main__":
    main()
