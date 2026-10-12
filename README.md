# building_mini_LLM

A small GPT trained from scratch on a single RTX 5060 (8 GB, Blackwell) under WSL2. The goal is a
customer-support chat model.

## Recipe

| Stage | Data | Why |
|---|---|---|
| 1. Pretrain | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`, first ~1B tokens) | General English, facts, and grammar from filtered educational web text |
| 2. Fine-tune | [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (27k Q/A) + `data/customer_support_data.csv`, cleaned | Teaches the `Customer:` / `Agent:` format and support tone |

> **About `customer_support_data.csv`:** it is synthetic. Every message is one of ~430 template sentences
> followed by 20+ random pseudo-words (`tyugwmtolj qripqglpz ...`), and that noise was **91% of all characters**.
> The original char-level model spent its capacity predicting random letters, which is why loss stalled
> around 2.7 and samples were gibberish. `prepare_data.py` strips the noise, dedupes the conversations, and caps
> how many go in, so the templates don't dominate.

Tokenizer: GPT-2 BPE (`tiktoken`), stored as `uint16`. Documents are separated by `<|endoftext|>`.

## Setup (WSL2)

```bash
conda activate ml-gpu-rtx50          # torch 2.10 nightly + cu128 (Blackwell needs CUDA 12.8+)
pip install -r requirements.txt
python -m pytest                      # unit + smoke tests, CPU-only, ~40 s
```

**Keep data and runs on the Linux filesystem** (`~/...`), not `/mnt/c/...`. WSL reaches Windows files over the 9P bridge,
and random reads there are much slower. `train.py` warns if your data is under `/mnt/`.

**Turn off sysmem fallback.** NVIDIA Control Panel → Manage 3D settings → *CUDA – Sysmem Fallback Policy* → *Prefer No
Sysmem Fallback*. With it on, a run that overflows VRAM keeps going but silently pages GPU memory into system RAM and
slows down by 10x or more (symptom: "100% utilization" at ~35 W). With it off, you get a clear out-of-memory error instead.

## Run

```bash
# 1. Build datasets (FineWeb streams; ~1B tokens -> ~2 GB on disk)
python src/prepare_data.py support --out ~/mini_llm_data/support --local-csv data/customer_support_data.csv
python src/prepare_data.py fineweb --out ~/mini_llm_data/fineweb_edu --max-tokens 1e9

# 2. Pretrain (one pass over the data; Ctrl-C saves last.pt, add --resume to continue)
python src/train.py --data-dir ~/mini_llm_data/fineweb_edu --out-dir ~/mini_llm_runs/pretrain

# 3. Fine-tune on support chats (~284 steps, ~5 min; small dataset, so evaluate often)
python src/train.py --data-dir ~/mini_llm_data/support --out-dir ~/mini_llm_runs/support \
    --init-from ~/mini_llm_runs/pretrain/best.pt --lr 1e-4 --epochs 4 --dropout 0.1 --eval-interval 25

# 4. Talk to it
python src/sample.py --ckpt ~/mini_llm_runs/support/best.pt --chat
python src/sample.py --ckpt ~/mini_llm_runs/pretrain/best.pt --prompt "The water cycle is" --num-samples 3
```

Metrics are appended to `<out-dir>/log.jsonl` (train loss, lr, tok/s, and eval losses). `best.pt` is the checkpoint
with the lowest validation loss; `last.pt` is the latest one (used by `--resume`). A new run won't start in an
`--out-dir` that already holds checkpoints: pass `--resume` to continue it, `--overwrite` to replace it, or use a new
directory.

## Model sizes for 8 GB

`--model-size` picks a preset. All presets use 64-dim attention heads (the flash-attention sweet spot), tied
embeddings, and a 512-token context.

| Preset | Layers × width | Params | Use |
|---|---|---|---|
| `tiny` | 6 × 384 | ~30M | Quick experiments |
| `small` (default) | 8 × 512 | ~51M | Best quality per hour on a 5060 |
| `medium` | 12 × 768 | ~124M (GPT-2 small) | Overnight runs with 2B+ tokens |

Rule of thumb (Chinchilla): train on about 20 tokens per parameter. That's ~1B tokens for `small` and ~2.5B for `medium`.

## What makes training fast on this card

- **bf16 autocast plus TF32 matmuls**: tensor-core math with no loss scaling needed.
- **`torch.compile`**: fuses kernels. Disable it with `--no-compile` when debugging.
- **Flash attention** via `F.scaled_dot_product_attention`. The old code ran one Python-level attention head at a
  time and built a full T×T mask for each.
- **Fused AdamW**, with weight decay applied to matrices only.
- **Small micro-batches with gradient accumulation.** The largest activation is the logits tensor
  (`micro_batch × block_size × 50304`), so the micro-batch is sized to fit in VRAM and accumulation brings the
  effective batch up to ~65k tokens.
- **Pinned-memory batches**: the next batch is copied to the GPU while the current backward pass runs.
- **Warmup plus cosine LR schedule.** The old `StepLR` cut the LR by 100x by step 4000, so most of a 12.5k-step
  run learned almost nothing.

## Layout

```
src/model.py         GPT model, presets, generation
src/data_loader.py   memmap random-window batch sampler
src/prepare_data.py  download, clean, tokenize -> train.bin / val.bin / meta.json
src/chat_format.py   the Customer:/Agent: template shared by prep and sampling
src/train.py         training loop (pretrain, fine-tune, resume)
src/sample.py        text generation and interactive chat
tests/               pytest suite
```
