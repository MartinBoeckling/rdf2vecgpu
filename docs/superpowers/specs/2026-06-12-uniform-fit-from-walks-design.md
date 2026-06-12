# Uniform `fit_from_walks` across backends + GPU-resident single-GPU feed

**Status:** Design approved 2026-06-12.
**Scope:** Generalise the decoupled `prepare_corpus` → `fit_from_walks` seam (today
torchrec-only) to all four training backends, and restore the in-memory
GPU-resident data feed as the default for the single-GPU pytorch path.
**Out of scope:** changing walk generation; the legacy `walk_generation` /
`_fit_from_tensors` / `ParquetSkipGramDataModule` paths (kept for `fit()`
back-compat, not extended); CBOW/bfs under the new seam.

## 1. Problem

Two coupled gaps, both surfaced while benchmarking the embedding backends:

1. **No uniform "train from a prepared corpus" entry.** Only `backend="torchrec"`
   exposes `fit_from_walks(walks_path, vocab_path)`. gensim and pytorch generate
   walks *inside* `fit()`, so a fair backend comparison — and the decoupled API
   the project already commits to (the walk corpus on disk is the
   backend-agnostic contract) — isn't available for them. A whole-pipeline
   benchmark also conflates **walk generation** with **embedding training**: on
   FB15k-237 the Dask/cuGraph multi-GPU walk generation took >10 min while the
   actual training is seconds, making the comparison meaningless.

2. **The single-GPU pytorch path lost its GPU-resident feed.** The throwaway
   spike commit `7b6172f` ("hard-wire pytorch fast path… No config flags /
   legacy preservation / tests — that's Plan 1") re-routed single-GPU
   skip-gram from the in-memory `_fit_from_tensors` path (cuDF →
   `to_dlpack()` → GPU tensors, no disk) to `_fit_via_pytorch_sequences`,
   which **unconditionally** writes the walks to a temp parquet dir and reads
   them back through `WalkSequenceDataModule` (pyarrow→NumPy on **CPU**) — even
   on one GPU. The clean version that preserved the in-memory default never
   landed, so single-GPU silently round-trips cuDF(GPU) → parquet(disk) →
   NumPy(CPU) → torch → GPU.

## 2. Architecture overview

- **Canonical corpus (already produced by `prepare_corpus`):** `out_dir/walks/`
  (walks-as-sequences: `vertices` / `predicates` list columns) + `vocab.parquet`
  (`token_id` int64, `entity` string). This single format feeds every backend.
- **`fit_from_walks(walks_path, vocab_path)` becomes backend-dispatched** (today
  it is torchrec-only). It loads the vocab from disk and routes to each
  backend's training, with **no walk generation**.
- **Single-GPU pytorch is GPU-resident** via a new in-memory DataModule that
  yields GPU tensors straight from a cuDF frame (DLPack), used by both `fit()`
  (in-memory walks) and `fit_from_walks` (`cudf.read_parquet`). Multi-GPU
  pytorch streams the parquet per rank under DDP (correct — a multi-GPU corpus
  can't sit in one process's GPU memory). gensim and torchrec are unchanged.
- **Launch models are inherent and unchanged:** gensim / pytorch-single = plain
  `python`; pytorch-multi = plain `python` (Lightning spawns DDP — needs
  `NCCL_P2P_DISABLE=1` on PCIe boxes without NVLink); torchrec = `torchrun`.

## 3. Components

### New

- **`InMemoryWalkSeqDataModule(cudf_df, *, emit_batch_size, shuffle, seed)`** in
  `embedders/word2vec_loader.py` — the GPU-native sibling of
  `WalkSequenceDataModule`. Batches a cuDF walks-as-sequences frame into
  `(vertices, predicates)` int tensors **on GPU** via DLPack (no parquet, no
  pyarrow/NumPy hop). Yields the same `(vertices, predicates)` shape the
  `SkipGram` on-GPU pair expansion (`_expand_skipgram`) already consumes, so the
  model/training is unchanged.
- **`GPU_RDF2Vec._load_word2idx(vocab_path) -> cudf.DataFrame`** — reads
  `vocab.parquet` and returns/sets `self.word2idx` in the `(token, word)` layout
  the pytorch/gensim extract paths expect (rename `token_id`→`token`,
  `entity`→`word`). torchrec keeps its own `_load_corpus_vocab`.
- **`GPU_RDF2Vec._train_pytorch_sequences(datamodule, n_gpus)`** — extracted
  shared tail: build `SkipGram`, configure the `L.Trainer`
  (`devices=n_gpus`, `strategy="ddp" if n_gpus>1 else "auto"`), `fit`, stash the
  model. Called by both `fit()`'s sequences path and `fit_from_walks` (pytorch).

### Modified

- **`GPU_RDF2Vec.fit_from_walks(walks_path, vocab_path)`** — generalise from the
  torchrec-only body to a backend dispatch:
  - `torchrec` → `_torchrec_preflight()` + `train_torchrec_word2vec` (unchanged).
  - `gensim` → `_load_word2idx`, `train_gensim_word2vec(walks_path, …)`.
  - `pytorch`, `multi_gpu=False` → `_load_word2idx`,
    `InMemoryWalkSeqDataModule(cudf.read_parquet(walks_path))`,
    `_train_pytorch_sequences(dm, n_gpus=1)`.
  - `pytorch`, `multi_gpu=True` → `_load_word2idx`,
    `WalkSequenceDataModule(walks_path)`, `_train_pytorch_sequences(dm, n_gpus)`.
  - Validates `walks_path` / `vocab_path` exist; torchrec additionally requires
    a torchrun launch (existing pre-flight).
- **`GPU_RDF2Vec._fit_via_pytorch_sequences(walk_sequences)`** — split by mode:
  - **single-GPU** (`walk_sequences` is cuDF): feed
    `InMemoryWalkSeqDataModule(walk_sequences)` directly to
    `_train_pytorch_sequences(dm, n_gpus=1)` — **no temp parquet, no CPU
    decode**. This is the restored default.
  - **multi-GPU** (`walk_sequences` is dask_cudf): unchanged — checkpoint to
    temp parquet, tear down Dask, DDP-train via `WalkSequenceDataModule`.

### Reused without changes

- `prepare_corpus`, `walk_generation_sequences`, `SkipGram` (+ `_expand_skipgram`),
  `WalkSequenceDataModule`, `train_gensim_word2vec`, `train_torchrec_word2vec`,
  `transform()` (already branches per backend), the `_prepare_corpus_fs` test
  helper.

## 4. Data flow

```
prepare_corpus(edge_df, out_dir)                     # once, outside torchrun
  └─> out_dir/walks/*.parquet  +  out_dir/vocab.parquet

fit_from_walks(walks_path, vocab_path)               # per backend
  ├─ _load_word2idx(vocab_path)                       # pytorch/gensim
  ├─ gensim       -> train_gensim_word2vec(walks_path)
  ├─ pytorch  1×  -> InMemoryWalkSeqDataModule(cudf.read_parquet(walks_path)) -> Trainer(devices=1)
  ├─ pytorch  N×  -> WalkSequenceDataModule(walks_path) -> Trainer(devices=N, ddp)   # NCCL_P2P_DISABLE=1
  └─ torchrec     -> train_torchrec_word2vec(walks_path, vocab_path)                  # torchrun

fit(edge_df)  single-GPU pytorch                      # end-to-end, GPU-resident
  └─ walk_generation_sequences -> cudf (GPU) -> InMemoryWalkSeqDataModule -> Trainer(devices=1)
```

The single-GPU `fit()` and `fit_from_walks` share `InMemoryWalkSeqDataModule`;
the only difference is the cuDF source (in-memory walks vs `cudf.read_parquet`).

## 5. Error handling

- `fit_from_walks`: raise a clear `FileNotFoundError` if `walks_path` or
  `vocab_path` is missing; torchrec keeps the existing torchrun pre-flight.
- `InMemoryWalkSeqDataModule`: require the cuDF frame to carry `vertices` /
  `predicates` list columns; raise if absent.
- Document (README + docstring) that pytorch-multi `fit_from_walks` uses
  Lightning DDP and needs `NCCL_P2P_DISABLE=1` on PCIe-only multi-GPU hosts —
  the same first-collective hazard as torchrec.

## 6. Testing

- **Unit**
  - `_load_word2idx` round-trips a `(token_id, entity)` vocab parquet into the
    `(token, word)` layout.
  - `InMemoryWalkSeqDataModule` yields `(vertices, predicates)` GPU tensors of
    the expected dtype/shape from a synthetic cuDF frame; honours
    `emit_batch_size`.
  - `fit_from_walks` dispatches to the right per-backend path (monkeypatched).
- **Integration** (gpu / multigpu marked)
  - Prepare one small corpus, then each of gensim / pytorch-1GPU / pytorch-mg /
    torchrec runs `fit_from_walks` + `transform` → finite, correctly-sized
    embeddings (reuse `_prepare_corpus_fs`).
  - Single-GPU `fit()` skip-gram takes the in-memory path: assert no temp
    parquet is written (e.g. patch `tempfile.mkdtemp` / `cudf.read_parquet` and
    assert the in-memory DataModule was used), embeddings finite.

## 7. Benchmark harness

`performance/bench_backends.py`: `prepare_corpus` once (shared corpus, timed
once), then every backend runs **`fit_from_walks` only** (the timed, compared
step) + `transform`. This isolates embedding training from walk generation —
removing the confound that made pytorch-mg look ~70× worse on FB15k-237.

## 8. Follow-ups (deferred)

- Generalise `fit_from_walks` to CBOW / order-aware skip-gram and bfs walks.
- A spawn-safe, picklable streaming loader so multi-GPU `fit_from_walks` can use
  DataLoader workers (currently `num_workers=0` on the torchrec path).
- Retire the legacy `walk_generation` / `_fit_from_tensors` /
  `ParquetSkipGramDataModule` paths once the sequences path covers every
  embedding model.
