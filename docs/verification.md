# Verification - 2026-10-26

## Prepared data

Both final corpora passed `python src/prepare_data.py verify --data-dir ...`,
which scans token counts, GPT-2 vocabulary bounds and SHA-256 content hashes.

| Corpus under `data/prepared/` | Documents | Train tokens | Validation tokens |
|---|---:|---:|---:|
| `fineweb_edu_1b` | 1,013,323 | 995,091,684 | 4,908,275 |
| `support_v2` | 31,872 | 4,652,294 | 246,184 |

FineWeb source: `HuggingFaceFW/fineweb-edu`, `sample-100BT`, revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The preparation inspected 1,018,923
documents, discarded 4,296 normalized exact duplicates and filtered 867 documents;
remaining excess documents were excluded by the token budget. Token files total
1,999,999,918 bytes. Per-file hashes and the full preparation recipe are in its
`meta.json`. The corpus is a fresh sample, not a guaranteed disjoint addition to
the earlier WSL corpus. The new default/lab recipe builds 5B tokens; that larger
corpus has not been downloaded here.

The initial 1B preparation process encountered a Windows SQLite temporary-directory
cleanup error after publishing the token files. The connection-close bug was fixed,
covered by the preparation test, and the final preparation code completed a separate
small live-download smoke check. The published 1B corpus passed the full verifier,
and its leftover temporary database was removed after the process exited.

Local preparation versions are recorded in the generated
`data/prepared/preparation_environment.json`. Existing root-level legacy artifacts
and `/home/sandy/mini_llm_data` / earlier training runs were preserved. Generated
data and runs are not transferred by cloning the Git repository.

## CPU regression suite

- `pytest -q`: **68 passed**, 53.65 seconds on the final behavioral code.
- Ruff lint: passed for all `src` and `tests`.
- Ruff formatting: passed for changed/new Python files.
- Python bytecode compilation and `git diff --check`: passed.
- CPU environment: Python 3.11.14, existing PyTorch 2.6.0.dev20241112; CUDA hidden
  for the final CPU suite. Recommended lab installation is a stable CUDA 12.8
  PyTorch build, as documented in the runbook.

Coverage includes causal attention, bidirectional future-token visibility, MLM
masking/EOT preservation, optimized-vs-full loss and parameter-gradient agreement,
learnability, encoder export, output-directory protection, final-update evaluation,
fixed evaluation sampling, corrupted corpus detection, changed-corpus resume
rejection, and checkpoint preservation after a failed update. Both objectives
produce bitwise-equal CPU weights for uninterrupted vs SIGTERM/resumed training
with dropout. A compiler-backend regression verifies that changing the number of
masked targets adds no compiled graphs after the first call.

## Local GPU compatibility checks

Hardware: RTX 5060, 8 GB, WSL2, PyTorch 2.10.0.dev20251203+cu128, CUDA 12.8,
Python 3.10.19. These checks establish the new GPU paths work on this setup;
they do not validate fit or performance on a 4090/5090.

- Encoder synthetic optimizer steps passed in BF16, both eager and compiled, with
  chunked loss and optional block checkpointing.
- A real-data encoder run initially exposed per-mask-count recompilation reaching
  PyTorch's limit of eight specializations. The final code keeps its sparse MLM
  loss eager while compiling the transformer. A subsequent 20-step real-data run
  passed without hitting that limit, including final evaluation and checkpointing.
- That run used `tiny`, context 128, micro-batch 2, accumulation 1, loss chunks 64,
  and block checkpointing. Peak allocated GPU memory was about 0.54 GiB. Its
  measured validation loss changed from 10.8482 to 9.8498. This tiny smoke experiment
  is not evidence of downstream embedding quality or a full pretraining result.
- A 3-step GPU resume followed by a 20-step continuation also completed during
  diagnosis, exercising safe checkpoint loading and CUDA RNG restoration.
- `encode.py` exported a 384-dimensional vector from the final smoke checkpoint.
- The causal decoder's compiled BF16 chunked-loss benchmark also completed forward,
  backward and optimizer updates.

GPU reports/checkpoints are in the new WSL directory
`/home/sandy/mini_llm_runs/pipeline_validation_20261026/`. Short synthetic benchmark
rates exclude data I/O and warmup; they should not be used to predict lab performance.
Real-data smoke batches were read from `/mnt/c/`, with the documented I/O warning.

Before a long lab run, execute `benchmark.py` with the chosen lab profile, then a
short real-data run. Actual 4090/5090 memory headroom, throughput, long-run stability,
and downstream task quality remain unmeasured.
