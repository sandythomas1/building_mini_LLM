# Decoder and encoder training decision

Date: 2026-10-26

The existing project was decoder-only GPT. The requested expansion adds a real
bidirectional encoder and scales data/training for a single RTX 4090/5090 while
preserving decoder generation and support fine-tuning.

Both models use the existing pre-norm transformer blocks with learned positions,
GELU MLPs, tied token/output embeddings and GPT-2 tokenization. `GPTConfig.causal`
selects SDPA attention direction. `Encoder` requires `causal=False`, exposes
contextual token vectors and rejects generation. The decoder retains its causal
mask. Old config dictionaries default to `causal=True`.

The encoder objective is BERT-style dynamic MLM with token-level corruption.
MASK uses vocabulary slot 50,257, already allocated by the padded model vocabulary
but never emitted by the original tokenizer. Existing corpora need no retokenization.
Loss excludes unselected positions and EOT. Encoder and decoder checkpoints retain
their objective; they cannot silently initialize each other. There is no cross
attention, shared trained weights between separate runs, or sequence-to-sequence
decoder. Those would require a separate task definition and paired-data objective.

Data preparation streams a pinned FineWeb-Edu snapshot into uint16 files. SQLite
stores normalized-text SHA-256 fingerprints to bound host RAM. Hash-based document
splitting is independent of encoding batch size. Files are published with metadata
and content hashes, and a standalone bounded-memory verifier checks copies. Exact
deduplication does not establish near-duplicate or downstream benchmark isolation.

Training retains PyTorch-native SDPA, BF16, compilation and AdamW. Optional block
checkpointing saves activation memory. Chunked cross entropy checkpoints the output
projection to avoid retaining every chunk's vocabulary-sized softmax activations.
MLM additionally selects target positions before projecting vocabulary logits.
The sparse MLM head explicitly stays outside the compiled graph: a real GPU run
exposed per-mask-count specialization reaching the compiler's recompile limit.
Keeping the transformer compiled and the variable-shaped loss eager avoids that
startup penalty. A CPU compiler-backend regression checks that changing mask counts
does not increase the number of compiled graphs.
The tradeoff is recomputing those chunks in backward. CPU tests compare losses and
parameter gradients against the full-logit path for both objectives.

Fixed evaluation samplers prevent evaluation from consuming training RNG and make
validation checkpoints comparable. Resume persists optimizer, sampler, masking and
CPU/CUDA RNG state; SIGINT/SIGTERM stops at a completed update boundary. CPU dropout
and MLM replay are tested for bitwise-equal weights. GPU determinism and actual
4090/5090 throughput require lab validation, so hardware presets are conservative
starting points and a real optimizer-step benchmark is supplied.
