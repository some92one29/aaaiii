# Tiny Language Model

This project trains a very small character-level language model in PyTorch. It is small enough to train on Google Colab in a reasonable amount of time, but large enough to generate interesting text.

## What it does

- Reads a plain text file
- Learns character-by-character next-token prediction
- Trains a tiny GPT-style transformer
- Generates new text samples

## Files

- `prepare_data.py`: converts raw text into train/validation tensors
- `model.py`: tiny transformer language model
- `train.py`: training loop
- `sample.py`: text generation
- `requirements.txt`: Python dependencies

## Quick start on Google Colab

Open a new Colab notebook and run these cells.

### 1. Install dependencies

```bash
!pip install -r requirements.txt
```

If you are starting from scratch in Colab, first create the files by uploading this folder to Colab or GitHub, then run commands from the project directory.

### 2. Check that GPU is enabled

In Colab:

`Runtime -> Change runtime type -> T4 GPU` if available.

Then verify:

```bash
!nvidia-smi
```

### 3. Get training data

You need one plain text file. The simplest option is a single `.txt` file called `input.txt`.

Good starter datasets:

- Tiny Shakespeare
- Your own notes, stories, or code
- Public-domain books from Project Gutenberg
- A cleaned text export from Wikipedia or similar sources

For a first run, Tiny Shakespeare is a good choice because it is small and trains quickly:

```bash
!mkdir -p data
!wget -O data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

If you have your own text file, upload it in Colab and place it at `data/input.txt`.

### 4. Prepare the dataset

```bash
!python prepare_data.py --input data/input.txt --out_dir data/shakespeare
```

This creates:

- `train.pt`
- `val.pt`
- `meta.json`

### 5. Train the model

```bash
!python train.py --data_dir data/shakespeare --out_dir out --steps 3000
```

For a slightly stronger model, try:

```bash
!python train.py --data_dir data/shakespeare --out_dir out --steps 5000 --n_embd 192 --n_head 6 --n_layer 6
```

### 6. Generate text

```bash
!python sample.py --model_dir out --start "ROMEO: " --max_new_tokens 400
```

## How to collect training data

Keep the first version simple. This model works best when the training text is fairly consistent in style.

### Good training data for this project

- One book
- A few books by the same author
- A folder of similar stories
- Dialogue transcripts in one format
- Code from one language

### Avoid at first

- Huge mixed datasets from very different sources
- Binary files, PDFs, or HTML dumps without cleaning
- Very tiny datasets under a few kilobytes

### Basic cleaning rules

Put everything into one UTF-8 text file:

- Remove obvious junk like navigation text or headers you do not want learned
- Keep line breaks if they matter to style
- Normalize weird encoding characters if needed
- Make sure the final file is plain text

If you have many text files, combine them:

```bash
!cat texts/*.txt > data/input.txt
```

## Expected training time on Colab

Approximate for the default model:

- Tiny Shakespeare, T4 GPU, 3000 steps: around 10 to 25 minutes

This varies with Colab availability and runtime settings.

## Suggested experiments

- Train on poems and sample poetry
- Train on your own writing and compare style transfer
- Train on Python files and generate toy code
- Increase `n_layer`, `n_head`, and `n_embd` slowly
- Increase `block_size` for longer context

## Notes

- This is a character-level model, so it learns text one character at a time.
- It is intentionally small and educational, not production-grade.
- If the loss is going down and samples are getting more coherent, training is working.
