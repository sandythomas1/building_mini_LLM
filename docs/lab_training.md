# Training in the GPU lab

The repository supports two separate models. A **causal GPT decoder** predicts the
next token and generates replies. A **bidirectional encoder** predicts hidden tokens
using both left and right context. They share model blocks, the GPT-2 BPE tokenizer,
and prepared corpora; they train separately and have incompatible objectives.
This is not a combined encoder-decoder sequence-to-sequence model.

## 1. Install on Linux or WSL2

Use Python 3.11 or 3.12. Keep the repository, corpus, and runs on the Linux SSD,
for example `~/building_mini_LLM`, rather than `/mnt/c/...` under WSL.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.7,<3' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
python -m pytest -q
```

The stable CUDA 12.8 wheels support Blackwell; a nightly build is unnecessary.
See the [PyTorch release notes](https://pytorch.org/blog/pytorch-2-7/).
The lab administrator must provide a compatible NVIDIA driver. Native Windows runs
eager mode; use Linux/WSL for Triton and `torch.compile`.

## 2. Prepare substantially more usable text

This workspace also has prepared data under `data/prepared/`: a 1B-token-budget
FineWeb-Edu corpus in `fineweb_edu_1b/`, plus 31,872 cleaned support conversations
in `support_v2/` (4,898,478 tokens including validation). These generated files are
ignored by Git, so cloning the repository does not transfer them. Copy the corpus
folder to the lab SSD and run `verify`, or regenerate it with the commands below.
The local preparation dependency versions are in
`data/prepared/preparation_environment.json`. A 1B-token corpus is sufficient for
the first encoder/decoder smoke experiments; the 5B recipe expands that starting set.

FineWeb-Edu is filtered educational English text. Use its `sample-100BT` pool;
the command streams only enough documents to meet the requested budget. It does
not download the entire 100B-token subset.

```bash
# Recommended first full corpus: ~10 GB for token files, plus temporary disk space.
python src/prepare_data.py fineweb --out ~/mini_llm_data/edu_5b \
  --max-tokens 5e9 --config sample-100BT --num-threads 8
python src/prepare_data.py verify --data-dir ~/mini_llm_data/edu_5b
```

Preparation resolves and records the dataset commit SHA, shuffles documents using a
bounded buffer and fixed seed, normalizes Unicode/line endings, filters empty,
short, overlong and NUL-containing text, and removes whitespace-normalized exact
duplicates using a disk-backed SQLite index. Duplicate detection covers both splits.
A normalized-text hash assigns about 0.5% of documents to validation independent
of batch size. The split and seed must remain fixed across comparison experiments.

Documents stay intact, end with EOT, and never cross train/validation. Random windows
within each split can span EOT-separated documents. This is packed-text pretraining;
the encoder also attends across document boundaries inside a window. There is no
additional near-duplicate or URL-family isolation beyond the upstream corpus's
deduplication. Validation is an internal development set, not an external benchmark.

`meta.json` records source, license, tokenizer, token counts, preparation settings,
filter statistics and SHA-256 checksums. `verify` scans with bounded RAM and checks
counts, token IDs, and hashes. Run it after copying data to a lab machine. Existing
datasets cannot be overwritten. Failed preparation does not publish completed token
files; an interrupted preparation restarts from the beginning in a new output
directory. The tokenizer is threaded on CPU; no GPU is needed for this step.

To reproduce a corpus, pass its recorded SHA with `--revision SHA`, plus the same
seed, config, filters, batch size and budget. Keep the preparation environment versions
as well. The sample shuffle is bounded, not a uniform shuffle of the entire corpus.
If expanding from 1B to 5B, rebuild the full 5B dataset and use that as a replacement;
do not concatenate the two corpora, which contain overlapping source documents.

The [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
describes the upstream filtering and ODC-BY license. Keep attribution and review
upstream content rights if redistributing the data.

## 3. Measure the actual lab GPU first

| Profile | Expected GPU | Model | Context | Micro-batch | Accumulation | Input tokens/update |
|---|---|---|---:|---:|---:|---:|
| `rtx4090` | 24 GB RTX 4090 | medium, ~124M | 1,024 | 8 | 8 | 65,536 |
| `rtx5090` | 32 GB RTX 5090 | medium, ~124M | 1,024 | 16 | 4 | 65,536 |

These are conservative starting configurations, **not measured fit or speed guarantees**.
Both retain FP32 master weights/Adam moments, use BF16 autocast when supported, TF32,
SDPA attention, tied embeddings, fused AdamW, and chunked loss. Chunked loss checkpoints
the vocabulary projection, recomputing it during backward to keep full-vocabulary
activations bounded. Encoder training projects only selected MLM positions in this path.
The transformer's graph is compiled; its variable-sized sparse MLM head stays eager
to avoid recompiling for each different number of masked positions.
Block checkpointing is optional; it trades compute for more free memory.

```bash
# Substitute rtx5090 on the 5090. Measure both objectives if training both.
python src/benchmark.py --profile rtx4090 --objective mlm --lr 3e-4 \
  --steps 20 --out ~/mini_llm_runs/4090_mlm_benchmark.json
python src/benchmark.py --profile rtx4090 --objective causal \
  --steps 20 --out ~/mini_llm_runs/4090_gpt_benchmark.json
```

The benchmark performs forward, backward and optimizer updates, excludes warmup and
compilation time, and reports input tokens/sec plus allocated/reserved VRAM. It uses
synthetic resident batches; actual corpus I/O, evaluation, and checkpointing add time.
It does not measure model quality. If it runs out of memory, halve `--micro-batch` and
double `--grad-accum` to preserve the effective batch, then try
`--gradient-checkpointing` or a smaller `--loss-chunk-size`. Keep headroom for compilation
and evaluation. On Windows, disable NVIDIA CUDA sysmem fallback to expose VRAM overflow
instead of silently paging to host RAM.

Estimate training time from measured input tokens/sec: `input token budget / tok_per_s`.
For example, 5B tokens at a measured 100k tok/s is about 13.9 compute hours, before
evaluation and checkpoints. This is an illustration, not a throughput prediction.

## 4. Train the encoder

```bash
# Short real-data check first; inspect loss and memory before committing a long session.
python src/train.py --profile rtx4090 --objective mlm \
  --data-dir ~/mini_llm_data/edu_5b --out-dir ~/mini_llm_runs/encoder_smoke \
  --lr 3e-4 --max-steps 100 --eval-interval 50 --eval-iters 10

# Full initial experiment; keep a separate output directory.
python src/train.py --profile rtx4090 --objective mlm \
  --data-dir ~/mini_llm_data/edu_5b --out-dir ~/mini_llm_runs/encoder \
  --lr 3e-4 --epochs 1 --eval-interval 250 --save-interval 250
```

Masking is generated anew on each batch: select 15% of tokens, replace 80% of those
with MASK, 10% with random tokens, and leave 10% unchanged. EOT is excluded. Only
selected tokens contribute to loss. MASK occupies padded vocabulary slot 50,257;
the original GPT-2 tokenizer and token files remain unchanged.

One 5B-token exposure budget has roughly 750M MLM target predictions. The logs count
**input** tokens, not targets. Start with one pass-equivalent budget, then compare a
separate longer run if validation is still improving. `--epochs` sets the number of
randomly sampled tokens; it does not guarantee visiting every document once. Decoder
token-per-parameter scaling heuristics do not directly determine MLM training length.

Extract an embedding from the best checkpoint:

```bash
python src/encode.py --ckpt ~/mini_llm_runs/encoder/best.pt \
  --text 'My order has not arrived.' --out ~/mini_llm_runs/order_embedding.npy
```

This returns a normalized mean of contextual token vectors. Inputs beyond the context
limit fail clearly. MLM pretraining alone does not establish useful retrieval/similarity
quality; add a contrastive or task-specific supervised objective and a held-out task
evaluation before using these vectors for search, ranking, or classification.

## 5. Train the decoder and retain support fine-tuning

```bash
python src/train.py --profile rtx4090 --objective causal \
  --data-dir ~/mini_llm_data/edu_5b --out-dir ~/mini_llm_runs/gpt

python src/prepare_data.py support --out ~/mini_llm_data/support \
  --local-csv data/customer_support_data.csv
python src/train.py --profile rtx4090 --data-dir ~/mini_llm_data/support \
  --out-dir ~/mini_llm_runs/support --init-from ~/mini_llm_runs/gpt/best.pt \
  --lr 1e-4 --epochs 4 --dropout 0.1 --eval-interval 10 --patience 4
python src/sample.py --ckpt ~/mini_llm_runs/support/best.pt --chat
```

The local CSV is mostly random suffix noise. The support preparer strips that suffix,
caps local template conversations, combines Bitext support examples, and deduplicates
across sources. General language pretraining remains separate from support fine-tuning.
An encoder checkpoint cannot be used for autoregressive chat or initialized as a decoder.

For more capacity after a successful medium baseline, benchmark `--model-size large`
(24 layers x 1,024 width, ~354M parameters). Start with micro-batch 4, accumulation 16,
and `--gradient-checkpointing`. A 10B-token general-text corpus needs about 20 GB for
tokens. Increase capacity only when validation and the available lab time justify it.

## 6. Monitor, stop, resume, and compare

Each run writes `run.json` with runtime versions, model settings and corpus identity;
`log.jsonl` contains train loss, learning rate, gradient norm, input throughput, peak
allocated memory, and fixed-window evaluation losses. `best.pt` follows validation;
`last.pt` contains the latest completed saved optimizer step. Evaluation runs after
the final update as well as at configured intervals. `--patience N` enables early
stopping after N evaluations without improvement.

Ctrl-C/SIGTERM finishes the current optimizer step, then saves `last.pt`. Repeat the
same command with `--resume` to continue. Checkpoints preserve the optimizer, sampler,
CPU/CUDA RNG and masking state. Exact CPU replay is regression-tested; CUDA kernels
and compiled graphs may still be nondeterministic. Hard process kills cannot save;
at most the work since the last periodic save is recoverable only by rerunning it.
An unexpected training error preserves the previous checkpoint rather than saving a
partially updated state.

Keep profile, micro-batch, accumulation, seed, masking probability and corpus fixed
on resume. The checkpoint's architecture and objective win. Record any deliberate
learning-rate/schedule changes: extending `--max-steps` also changes cosine decay.
For a new corpus or experiment use `--init-from` and a new output directory.
Old checkpoints without RNG state can continue with a warning, but cannot replay
exactly. Old root-level `model_checkpoint.pt`/`training_data.bin` artifacts may be from
the character-level experiment; they are not the new GPT-2 corpus or encoder weights.

Compare fixed-data runs using validation loss and task examples. Encoder masked-token
loss and decoder next-token loss have different targets and are not comparable scores.
For encoder use cases, create a separate held-out downstream test before selecting a
model for deployment. No long lab training or 4090/5090 throughput claim is established
by the local CPU tests.
