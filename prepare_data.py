import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a character-level dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw text file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for processed data.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation split fraction.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(len(data) * (1 - args.val_frac))
    train_data = data[:split]
    val_data = data[split:]

    torch.save(train_data, out_dir / "train.pt")
    torch.save(val_data, out_dir / "val.pt")

    meta = {
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": itos,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Read {len(text):,} characters from {input_path}")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Wrote dataset to {out_dir}")


if __name__ == "__main__":
    main()
