# building_mini_LLM

Train a small **GPT decoder** for text generation and a separate **bidirectional
encoder** for masked language modeling on one NVIDIA GPU. Both use GPT-2 BPE token
files; the encoder is a transformer, separate from the tokenizer's text encoding step.

The original 8 GB RTX 5060 recipe still works. New `rtx4090` (24 GB) and `rtx5090`
(32 GB) profiles use a ~124M-parameter model, 1,024-token context, BF16, SDPA attention,
fused AdamW and memory-bounded chunked loss. Profiles are starting points to benchmark
on the actual lab card. A ~354M `large` model is also available.

**Full setup, encoder explanation, data quality, GPU tuning, and resume instructions:
[GPU lab runbook](docs/lab_training.md).**

## Setup

Linux/WSL2, Python 3.11 or 3.12. Keep data and runs on the Linux SSD under WSL.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.7,<3' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pytest -q
```

Stable CUDA 12.8 wheels support Blackwell; a nightly build is unnecessary
([PyTorch release notes](https://pytorch.org/blog/pytorch-2-7/)). Native Windows runs
without compilation; Linux/WSL supports `torch.compile`.

## Data and training

```bash
# Stream a 5B-token budget (~10 GB); normalize, dedupe, split, tokenize and hash.
python src/prepare_data.py fineweb --out ~/mini_llm_data/edu_5b --max-tokens 5e9
python src/prepare_data.py verify --data-dir ~/mini_llm_data/edu_5b

# Replace rtx4090 with rtx5090 on that card. Benchmark before the full run.
python src/benchmark.py --profile rtx4090 --objective mlm --out encoder_benchmark.json

# Encoder: dynamically masked tokens, bidirectional attention.
python src/train.py --profile rtx4090 --objective mlm --lr 3e-4 \
  --data-dir ~/mini_llm_data/edu_5b --out-dir ~/mini_llm_runs/encoder
python src/encode.py --ckpt ~/mini_llm_runs/encoder/best.pt \
  --text 'My order has not arrived.' --out order_embedding.npy

# Decoder: causal next-token prediction, separate weights/run directory.
python src/train.py --profile rtx4090 --objective causal \
  --data-dir ~/mini_llm_data/edu_5b --out-dir ~/mini_llm_runs/gpt
python src/sample.py --ckpt ~/mini_llm_runs/gpt/best.pt --prompt 'The water cycle is'
```

FineWeb preparation draws from the larger
[FineWeb-Edu sample-100BT pool](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
streams only the requested budget, and records source revision, quality counts,
split settings and SHA-256 checksums in `meta.json`. Existing corpora are protected
from overwrite. Both models can use the same corpus. Larger corpora replace smaller
overlapping samples; do not concatenate them.

The encoder exports contextual mean-pooled vectors. Useful sentence similarity or
retrieval still needs contrastive/task supervision and a held-out task evaluation.
The models train independently; this repository does not connect them into a
sequence-to-sequence encoder-decoder model.

## Customer support fine-tuning

```bash
python src/prepare_data.py support --out ~/mini_llm_data/support \
  --local-csv data/customer_support_data.csv
python src/train.py --profile rtx4090 --data-dir ~/mini_llm_data/support \
  --out-dir ~/mini_llm_runs/support --init-from ~/mini_llm_runs/gpt/best.pt \
  --lr 1e-4 --epochs 4 --dropout 0.1 --eval-interval 10 --patience 4
python src/sample.py --ckpt ~/mini_llm_runs/support/best.pt --chat
```

The local synthetic CSV contained mostly random suffix noise. Preparation strips
that suffix, caps template conversations, and deduplicates across the local source
and [Bitext support data](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

`run.json` records settings, runtime and corpus identity. `log.jsonl` records losses,
learning rate, gradient norm, input tokens/sec and GPU memory. Evaluation uses fixed
windows without consuming training RNG, including after the final update. Ctrl-C
or SIGTERM saves at a completed optimizer step; repeat the command with `--resume`.
Periodic saves protect against hard stops. `best.pt` follows validation loss;
`last.pt` preserves optimizer, RNG and sampler state for continuation.

## Layout

```text
src/model.py          Shared blocks, causal GPT, bidirectional Encoder, presets
src/data_loader.py    Memmap batches, dynamic MLM masking, sampler state
src/prepare_data.py   FineWeb/support preparation and corpus verification
src/train.py          GPU profiles, training, evaluation, checkpoint/resume
src/benchmark.py      GPU training throughput and memory measurement
src/encode.py         Encoder contextual embedding export
src/sample.py         Decoder generation and support chat
tests/                CPU regression and integration tests
docs/lab_training.md  Complete lab workflow and practical limits
```
