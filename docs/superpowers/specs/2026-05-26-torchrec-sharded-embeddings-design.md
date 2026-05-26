# Torchrec-backed sharded embeddings for Word2Vec training

**Status:** Design approved 2026-05-26. Implementation plan pending.
**Scope:** v1 of a new `backend="torchrec"` training backend.
**Out of scope:** CBOW + order-aware skip-gram under sharding; CPU/UVM offload; custom sharding plans; backwards-compatibility with `pl.Trainer` on the sharded path; resume-from-checkpoint.

## 1. Problem

The existing `backend="pytorch"` trainer allocates two full `nn.Embedding(vocab_size, dim)` tables (in + output) and replicates them on every DDP rank. For Wikidata-scale graphs the embedding parameters alone are ≈312 GB (390 M tokens × 100 dims × 4 B × 2 tables), exceeding aggregate VRAM on any reasonable consumer-GPU cluster. The `backend="gensim"` path solves this on CPU but at CPU throughput.

This design adds a path that keeps training on GPUs while partitioning the embedding tables row-wise across ranks, so the per-rank VRAM footprint is `total_table_bytes / world_size` instead of `total_table_bytes`.

## 2. Architecture overview

- **New backend:** `backend="torchrec"`, joining the existing `"pytorch"` and `"gensim"` values on `RDF2VecConfig`.
- **Coverage:** skip-gram only in v1. CBOW + order-aware skip-gram raise `NotImplementedError` when combined with this backend.
- **Sharding library:** Meta's [torchrec](https://github.com/pytorch/torchrec) (`EmbeddingCollection`, `EmbeddingShardingPlanner`, `DistributedModelParallel`), backed by FBGEMM fused embedding+optimizer kernels.
- **Distributed runner:** Lightning Fabric (`lightning.fabric.Fabric`). Replaces `pl.Trainer(strategy="ddp")` on this path. Fabric and torchrec coexist cleanly; `pl.Trainer` and torchrec's `DistributedModelParallel` do not.
- **Optimizer:** torchrec fused row-wise. `sparseadam → PARTIAL_ROWWISE_ADAM`, `sgd → EXACT_SGD`. Documented difference: `PARTIAL_ROWWISE_ADAM` keeps per-row first/second-moment state (not per-cell as `torch.optim.SparseAdam`).
- **Negative sampling:** same global f^0.75 unigram (or uniform) as today; negative-row lookups go through the same sharded output table, so torchrec handles the all-to-all. Training semantics are preserved.
- **Output:** `fit_transform` / `transform` return a `dask_cudf.DataFrame` on this backend (one partition per rank). Single-rank `cudf.DataFrame` return is preserved for the `pytorch` and `gensim` backends.

## 3. Components

### New

**`src/rdf2vecgpu/embedders/torchrec_word2vec.py`** (est. 400–600 LOC inc. docstrings)
- `ShardedSkipGramModel(nn.Module)` — two `EmbeddingCollection`s (in + out) over a shared vocab. Forward signature `(center, context, negative) → loss` mirrors the existing `Word2VecSkipgram`.
- `_build_sharding_plan(model, world_size, …)` — defers to `torchrec.distributed.planner.EmbeddingShardingPlanner`. No knobs exposed in v1.
- `_sample_negatives(...)` — port of the existing f^0.75 unigram sampler; emits a flat token-id tensor consumed by the sharded output table.
- `train_torchrec_word2vec(config, vocab_size, token_counts, walk_loader)` — top-level entry point. Builds Fabric, wraps the model in `DistributedModelParallel`, attaches the fused optimizer via `apply_optimizer_in_backward`, runs the epoch loop, hooks into `tracker`.
- `extract_embeddings_to_dask_cudf(model, vocab_df) → dask_cudf.DataFrame` — per-rank shard extraction; one dask partition per rank, no rank-0 gather.

### Modified

**`src/rdf2vecgpu/config.py`**
- `backend: Literal["pytorch", "gensim", "torchrec"]`.
- Docstring spells out: torchrec backend requires `multi_gpu=True`, skip-gram only, launched via `torchrun`.

**`src/rdf2vecgpu/gpu_rdf2vec.py`**
- New `_fit_via_torchrec(...)` method, mirroring `_fit_via_gensim(...)`.
- `fit()` dispatch on `backend == "torchrec"`.
- Pre-flight checks (see §5).
- `transform()` / `fit_transform()` branch on backend to choose return type.

**`pyproject.toml`**
- New optional group `[project.optional-dependencies].torchrec` (`torchrec`, `fbgemm-gpu`, version-pinned).
- New `multigpu` pytest marker in `[tool.pytest.ini_options]`.

**`README.md`**
- Install snippet: `pip install rdf2vecgpu[torchrec]`.
- New "Sharded embeddings (torchrec backend)" subsection under Implementation Details, covering `torchrun` launch and the `dask_cudf` return type.

### Reused without changes

- `corpus/walk_corpus.py` — walk generation (multi-GPU path already exists and scales).
- `helper/functions.py` — vocab build (`_generate_vocab` scales to 390 M tokens).
- `embedders/word2vec_loader.py` — cuDF→tensor loader. A thin adapter wraps it with `Fabric.setup_dataloaders`; the loader itself does not change.

## 4. Data flow

### `fit(edge_df, walk_vertices=None)` — torchrec path

1. Pre-flight checks (§5).
2. Walk generation via `MultiGPUWalkCorpus.random_walk_sequences(...)` → dask_cudf walks parquet on disk. **Backend-agnostic; works for any walk strategy.**
3. Vocab build via `_generate_vocab(...)` → `vocab_df` (dask_cudf), `token_counts` (cudf.Series).
4. `fabric = Fabric(accelerator="gpu", devices="auto", precision=…)`; `fabric.launch()`.
5. Build `ShardedSkipGramModel`; build plan via `EmbeddingShardingPlanner.collective_plan(...)`; wrap in `DistributedModelParallel(model, plan=plan, device=fabric.device, init_data_parallel=False)`.
6. Attach fused optimizer via `apply_optimizer_in_backward(PartialRowWiseAdam | ExactSGD, embedding_params, optimizer_kwargs)`.
7. Build DataLoader from the walks parquet; wrap via `fabric.setup_dataloaders(loader, use_distributed_sampler=True)`.
8. Training loop (~50 LOC, Fabric-driven):
   ```
   for epoch in range(config.epochs):
       for batch in loader:
           center, context = batch
           negative = _sample_negatives(...)
           loss = model(center, context, negative)
           fabric.backward(loss)
           # optimizer step fused into backward
           tracker.log({"train/loss": loss.item(), ...})
   ```
9. Stash the wrapped model on `self._torchrec_model` for `transform()`.

### `transform(...)` — torchrec path

- Each rank owns rows `[rank * shard_size, (rank+1) * shard_size)` of the in-table.
- Per-rank: read local shard weight tensor → `cudf.DataFrame` with `token_id` + `dim` columns → join (shard-local) with the vocab slice to attach the entity string → wrap as one dask_cudf partition.
- Return `dask_cudf.DataFrame` with one partition per rank.
- User calls `.to_parquet("dir/")` → partitioned parquet dataset on disk.

### Walked-mask + un-walked-row zeroing (spec §7.5)

Each rank computes the mask for its local shard and zeros locally — no comms required. Same observable semantics as the pytorch backend, executed shard-local.

### Distributed launch

```bash
torchrun --standalone --nproc_per_node=2 your_script.py
```

Single-process invocations of `backend="torchrec"` raise with the launch hint (see §5).

## 5. Error handling

Principle: **fail fast at `fit()` entry with an actionable message; never let the user discover the problem mid-epoch.**

### Pre-flight (raised before walk generation)

| Condition | Exception | Message |
|---|---|---|
| `embedding_model != "skipgram"` | `NotImplementedError` | `"backend='torchrec' currently supports embedding_model='skipgram' only (got {model!r})."` |
| `multi_gpu=False` | `ValueError` | `"backend='torchrec' requires multi_gpu=True."` |
| `torchrec` or `fbgemm_gpu` not importable | `ImportError` | `"backend='torchrec' requires the torchrec extra. Install with: pip install rdf2vecgpu[torchrec]"` |
| `torch.distributed` not initialised | `RuntimeError` | `"backend='torchrec' requires a distributed launch. Run via: torchrun --standalone --nproc_per_node=N your_script.py"` |

Notably **absent:** any `walk_strategy` restriction. Walks are sequences of token IDs by the time training sees them; the backend has no business knowing how they were generated.

### During fit

- **Sharding-plan OOM** → catch `PlannerError`, re-raise with explicit budget info: `"Sharding plan could not fit vocab={N} dim={D} across world_size={W} (per-rank budget exceeded). Either increase --nproc_per_node, reduce vector_size, or wait for CPU/UVM offload support."`
- **Rank desynchronisation** → rely on Fabric's barrier-on-error; no custom watchdog.
- **Loss NaN** → log + raise after one batch, matching pytorch backend behaviour.

### During transform

- `transform()` called outside the same distributed launch as `fit()` → `RuntimeError("transform() on backend='torchrec' must be called within the same distributed launch as fit(). The sharded model is rank-local and cannot be reconstructed after the process group is destroyed.")`
- Backend mismatch between `fit()` and `transform()` → `RuntimeError` spelling it out.

### Deliberate non-handling

- **Cross-version torchrec/FBGEMM drift** — pinned dep, patches via `_compat.py` if needed.
- **Resume from checkpoint** — out of scope for v1.
- **Silent fallback to replicated embeddings** — never. The user picked this backend for a reason.

## 6. Testing

### New marker

`@pytest.mark.multigpu`, registered in `pyproject.toml`. Tests under it skip unless `WORLD_SIZE >= 2`. Local invocation: `torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu`.

### `test/embedders/torchrec_word2vec_test.py`

**Layer 1 — Import & config guards (no GPU, no torchrec needed)**
- `backend="torchrec"` + `embedding_model="cbow"` → `NotImplementedError`.
- `backend="torchrec"` + `multi_gpu=False` → `ValueError`.
- Module skip-at-import if `torchrec` extra missing.

**Layer 2 — Single-process / degenerate-shard (`gpu` marker, 1 GPU)**
- `ShardedSkipGramModel` with `world_size=1` (planner degenerates to table-wise): forward on a tiny vocab (1024 × 16), assert finite loss.
- `_sample_negatives` produces token ids in `[0, vocab_size)`, no NaNs.
- `extract_embeddings_to_dask_cudf` returns a `dask_cudf.DataFrame` with the right schema and row count.

**Layer 3 — True multi-GPU integration (`multigpu` marker, skipped in CI by default)**
- Tiny graph (~10 K triples), `world_size=2`, 1 epoch. Asserts: training completes; each rank owns ~half the vocab IDs; extracted dask_cudf covers `[0, vocab_size)` exactly once with no gaps or duplicates (mirrors the row-drift guard in the existing vocab tests); loss decreases over a multi-epoch run on a synthetic two-cluster dataset.

**Layer 4 — Cross-backend equivalence (`multigpu` marker, optional)**
- Same tiny graph + seed; train via `backend="pytorch"` and `backend="torchrec"`. Assert top-K nearest-neighbours overlap > 70 %. Not bit-exact (`PARTIAL_ROWWISE_ADAM` ≠ `SparseAdam`); a quality-regression canary.

### CI integration

- Layers 1 + 2 run on the existing GPU CI runner.
- Layers 3 + 4 are opt-in; contributors with a 2-GPU box run them via `torchrun … pytest -m multigpu`. Release process for any version touching torchrec code requires a manual multi-GPU run.

### Not tested

- FBGEMM kernel correctness (assumed; flag upstream if broken).
- Torchrec planner internals.
- Resume-from-checkpoint (out of scope).
- Scaling beyond 2 GPUs (algorithm is the same; correctness at world_size=2 covers it).

## 7. Follow-ups (explicitly deferred)

- **`fit_from_walks(walks_path, vocab_path)` entry point**, applied across all three backends, exposing the on-disk walks parquet as the natural seam between extraction and training. Deferred to its own design + PR; bundling here would muddy the scope.
- **Removing the existing `backend="gensim" + walk_strategy="bfs"` restriction** (same decoupling principle); separate PR.
- **CBOW and order-aware skip-gram** under sharding.
- **CPU/UVM offload** for tables exceeding aggregate VRAM.
- **Custom `ParameterConstraints`** if torchrec's planner picks suboptimally at some specific scale we haven't observed yet.
