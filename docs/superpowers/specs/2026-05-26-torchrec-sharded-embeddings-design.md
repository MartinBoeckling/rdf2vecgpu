# Torchrec-backed sharded embeddings for Word2Vec training

**Status:** Implemented and verified end-to-end on 2× RTX A6000 (2026-06-12). v0.5.0.
**Scope:** v1 of a new `backend="torchrec"` training backend.
**Out of scope:** CBOW + order-aware skip-gram under sharding; CPU/UVM offload; custom sharding plans; backwards-compatibility with `pl.Trainer` on the sharded path.

---

## Update (2026-06-12) — implemented architecture & verification findings

The sections below are the original approved design. Four things changed during
implementation and first end-to-end verification; **where they conflict, this
update wins**:

1. **Decoupled data-prep from training (supersedes §3 `_fit_via_torchrec` and
   the §4 `fit(edge_df)` path).** The cuGraph + dask-cudf walk generation runs
   its own GPU cluster and **cannot share a process with the `torchrun`/NCCL
   job** (each paradigm wants to own all GPUs; symptoms were
   `cudaErrorInvalidDevice` + a dtype merge crash in `_generate_vocab`). The
   backend is therefore split into two entry points:
   - `GPU_RDF2Vec.prepare_corpus(edge_df, out_dir, walk_vertices=None)` — runs
     the single-GPU data path **outside** `torchrun`; writes a backend-agnostic
     corpus (`walks/` parquet + `vocab.parquet`). Any `walk_strategy` is valid;
     the on-disk corpus is the contract.
   - `GPU_RDF2Vec.fit_from_walks(walks_path, vocab_path)` — the training entry,
     launched under `torchrun`; loads vocab + walks and trains. It does **no**
     walk generation. `_fit_via_torchrec(edge_df)` now raises with guidance
     pointing here.

2. **NCCL P2P deadlock on PCIe boxes.** On consumer multi-GPU hosts without
   NVLink, NCCL peer-to-peer over PCIe deadlocks the **first** collective —
   the failure looks like a `DistributedModelParallel`-init hang in object
   collectives (the `_get_pg_default_device will be deprecated` warning is a
   red herring, not a version problem). **Mitigation: export
   `NCCL_P2P_DISABLE=1`.** Do not bake it into the library (NVLink systems want
   P2P); document it as a deployment note.

3. **Fused-optimizer gradient clipping (correctness, not perf).** The skip-gram
   loss is a SUM over the pair batch, so the default `"sgd"` optimizer diverges
   to all-NaN without clipping. The pytorch backend uses an explicit
   `clip_grad`; the fused optimizer-in-backward has no such hook, so
   `_fused_optimizer_from_config` passes FBGEMM `gradient_clipping=True,
   max_gradient=grad_clip_value` (element-wise; matches
   `grad_clip_mode="element"`). The optimizer classes are
   `torchrec.optim.SGD` / `PartialRowWiseAdam` (not the `EXACT_SGD` /
   `PARTIAL_ROWWISE_ADAM` enum names guessed in §2).

4. **Batch-count join + row-striped loader (correctness).** torchrec's sharded
   forward all-to-alls every step, so every rank must run the same number of
   `forward()`s. The pair loader row-stripes each walk batch across ranks (not
   per-file sharding, which starved ranks), runs with `num_workers=0` (forked
   CUDA workers abort), and the training loop gates each step on a min-reduce
   "join" so all ranks stop together.

**Verified stack:** torch 2.9.1+cu128 · torchrec 1.4.0+cu128 · fbgemm-gpu
1.4.1+cu128 · RAPIDS 25.06. **Tests green:** single-GPU 22 passed; multi-GPU
5 passed (smoke, planted-community-structure, DCP save/reload bit-equal,
overwrite, keep-last). The cross-backend top-K overlap test (§6) was replaced
by `test_torchrec_recovers_planted_community_structure` (finite + intra/inter
cosine gap), because the original was ill-posed on a structureless graph with
independent vocabularies.

---

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
- **Checkpointing:** `torch.distributed.checkpoint` (DCP) — PyTorch-stdlib sharded checkpoints. Per-rank shards under a single directory; supports resharding across runs (save at `world_size=4`, resume at `world_size=2`). Saves model + optimizer + RNG + epoch/step metadata. Schedule: every epoch boundary **and** every `checkpoint_interval_minutes`. API: single `checkpoint_dir` field with auto-resume on subsequent invocations.

## 3. Components

### New

**`src/rdf2vecgpu/embedders/torchrec_word2vec.py`** (est. 550–800 LOC inc. docstrings, with checkpointing)
- `ShardedSkipGramModel(nn.Module)` — two `EmbeddingCollection`s (in + out) over a shared vocab. Forward signature `(center, context, negative) → loss` mirrors the existing `Word2VecSkipgram`.
- `_build_sharding_plan(model, world_size, …)` — defers to `torchrec.distributed.planner.EmbeddingShardingPlanner`. No knobs exposed in v1.
- `_sample_negatives(...)` — port of the existing f^0.75 unigram sampler; emits a flat token-id tensor consumed by the sharded output table.
- `train_torchrec_word2vec(config, vocab_size, token_counts, walk_loader)` — top-level entry point. Builds Fabric, wraps the model in `DistributedModelParallel`, attaches the fused optimizer via `apply_optimizer_in_backward`, runs the epoch loop, hooks into `tracker`, integrates checkpoint save/resume (§4.4).
- `extract_embeddings_to_dask_cudf(model, vocab_df) → dask_cudf.DataFrame` — per-rank shard extraction; one dask partition per rank, no rank-0 gather.
- `_save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir, keep_last=3)` — writes a DCP shard set to `{checkpoint_dir}/step_{global_step}/`, saves rank-0-only metadata (`epoch`, `global_step`, RNG state, vocab path) as a sibling `metadata.pt`, atomically updates `{checkpoint_dir}/latest` symlink, and prunes all but the most recent `keep_last` checkpoints.
- `_maybe_resume_checkpoint(model, optimizer, checkpoint_dir, overwrite) → (start_epoch, start_step)` — if `checkpoint_dir/latest` exists and `overwrite=False`, `DCP.load(...)` into the already-sharded model + optimizer (DCP handles resharding when current world_size differs from the saved one), restore metadata, return resume point; otherwise return `(0, 0)`. If `overwrite=True`, wipe the directory first.

### Modified

**`src/rdf2vecgpu/config.py`**
- `backend: Literal["pytorch", "gensim", "torchrec"]`.
- New fields (only consumed by the torchrec backend; ignored elsewhere with a doc note):
  - `checkpoint_dir: str | None = None` — when set, enables checkpointing for the torchrec backend. If the directory already contains a valid DCP checkpoint, training auto-resumes; if empty or missing, a fresh run starts and writes into it.
  - `checkpoint_interval_minutes: int = 30` — wall-clock interval between time-based saves. Epoch-boundary saves always happen on top of this.
  - `checkpoint_overwrite: bool = False` — escape hatch: if `True`, wipe `checkpoint_dir` at the start of `fit()` and start fresh.
  - `checkpoint_keep_last: int = 3` — how many step-directories to retain; older ones are pruned after each save.
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
7. **Maybe resume from checkpoint** — `start_epoch, start_step = _maybe_resume_checkpoint(model, optimizer, config.checkpoint_dir, config.checkpoint_overwrite)`. Must happen *after* the model is wrapped in `DistributedModelParallel` so DCP loads into the already-sharded parameters (this is what enables resharding).
8. Build DataLoader from the walks parquet; wrap via `fabric.setup_dataloaders(loader, use_distributed_sampler=True)`.
9. Training loop (~80 LOC, Fabric-driven):
   ```
   last_ckpt_time = time.monotonic()
   for epoch in range(start_epoch, config.epochs):
       for step, batch in enumerate(loader, start=start_step if epoch == start_epoch else 0):
           center, context = batch
           negative = _sample_negatives(...)
           loss = model(center, context, negative)
           fabric.backward(loss)
           # optimizer step fused into backward
           tracker.log({"train/loss": loss.item(), ...})

           # time-based checkpoint
           if config.checkpoint_dir and \
              (time.monotonic() - last_ckpt_time) >= config.checkpoint_interval_minutes * 60:
               _save_checkpoint(model, optimizer, epoch, global_step, config.checkpoint_dir,
                                keep_last=config.checkpoint_keep_last)
               last_ckpt_time = time.monotonic()

       # epoch-boundary checkpoint
       if config.checkpoint_dir:
           _save_checkpoint(model, optimizer, epoch + 1, global_step, config.checkpoint_dir,
                            keep_last=config.checkpoint_keep_last)
   ```
10. Stash the wrapped model on `self._torchrec_model` for `transform()`.

### `transform(...)` — torchrec path

- Each rank owns rows `[rank * shard_size, (rank+1) * shard_size)` of the in-table.
- Per-rank: read local shard weight tensor → `cudf.DataFrame` with `token_id` + `dim` columns → join (shard-local) with the vocab slice to attach the entity string → wrap as one dask_cudf partition.
- Return `dask_cudf.DataFrame` with one partition per rank.
- User calls `.to_parquet("dir/")` → partitioned parquet dataset on disk.

### Walked-mask + un-walked-row zeroing (spec §7.5)

Each rank computes the mask for its local shard and zeros locally — no comms required. Same observable semantics as the pytorch backend, executed shard-local.

### Checkpointing (§4.4)

**On-disk layout** under a user-supplied `checkpoint_dir`:

```
{checkpoint_dir}/
├── step_00001500/              # DCP shard set written by all ranks
│   ├── __0_0.distcp            #   (per-rank shard files; DCP's internal naming)
│   ├── __1_0.distcp
│   └── .metadata               #   DCP planner metadata
├── step_00003000/
│   └── …
├── metadata.pt                 # rank-0 only: epoch, global_step, RNG, vocab path, config snapshot
└── latest -> step_00003000     # symlink to the most recent checkpoint
```

**Save protocol** (executed by `_save_checkpoint`):

1. All ranks call `DCP.save(state_dict, storage_writer=FileSystemWriter(f"{ckpt_dir}/step_{global_step}"))`. `state_dict` contains `model.state_dict()` (sharded tensors) and `optimizer.state_dict()` (sharded optimizer state).
2. Rank 0 writes `metadata.pt` (epoch, global_step, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, vocab parquet path, the `RDF2VecConfig` snapshot).
3. Rank 0 atomically updates the `latest` symlink (`tmp = f"{ckpt_dir}/latest.tmp"; os.symlink(...); os.replace(tmp, f"{ckpt_dir}/latest")`).
4. Rank 0 prunes `step_*` directories older than the most recent `checkpoint_keep_last`.
5. Barrier before continuing training.

**Resume protocol** (executed by `_maybe_resume_checkpoint`):

- `checkpoint_dir` does not exist → create it, return `(0, 0)`.
- `checkpoint_dir` exists but `latest` does not (empty dir, or never-completed save) → start fresh in the existing directory, return `(0, 0)`. (No error — this is the common case where a user pre-created the directory.)
- `checkpoint_dir/latest` resolves to a valid `step_*` directory and `checkpoint_overwrite=False` → call `DCP.load(state_dict, storage_reader=FileSystemReader(...))`. **DCP redistributes the saved shards to match the current sharding plan**, so a checkpoint saved at world_size=4 loads transparently at world_size=2. Load `metadata.pt`, restore RNG state, return `(epoch, global_step)`.
- `checkpoint_overwrite=True` → recursively delete `checkpoint_dir` contents, recreate empty, return `(0, 0)`.
- `checkpoint_dir/latest` resolves to a missing or corrupted step directory → raise `RuntimeError` with the broken path. Do not silently drop the checkpoint.

**Resharding caveat.** DCP handles model + optimizer tensor resharding, but the *vocab itself* must be unchanged across save and resume. If the user changes `min_count` or the input graph between runs, vocab IDs shift and the checkpoint is no longer valid. We guard against this by saving the vocab parquet path in `metadata.pt` and refusing to resume if `config.checkpoint_dir` is reused with a different vocab path. Override via `checkpoint_overwrite=True`.

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

### Checkpoint-specific failures

| Condition | Exception | Message |
|---|---|---|
| `checkpoint_dir/latest` resolves to a missing/corrupted directory | `RuntimeError` | `"Checkpoint at {path} is missing or corrupt. Pass checkpoint_overwrite=True to start fresh, or restore the checkpoint contents."` |
| Resuming with a different vocab parquet path than the saved one | `RuntimeError` | `"Cannot resume: saved checkpoint was built against vocab at {saved_path}, but current run resolved vocab to {current_path}. Vocab IDs would no longer match. Pass checkpoint_overwrite=True to start fresh."` |
| Disk write failure during `_save_checkpoint` | original `OSError` propagated | DCP's writer raises directly; we add no wrapping. Training does not silently continue on save failure. |
| `checkpoint_dir` is on a non-shared filesystem in a multi-node setup | not detectable in v1; documented as a known limitation | README note: single-node multi-GPU is fully supported; multi-node requires a shared filesystem visible at the same path on every node. |

### During transform

- `transform()` called outside the same distributed launch as `fit()` → `RuntimeError("transform() on backend='torchrec' must be called within the same distributed launch as fit(). The sharded model is rank-local and cannot be reconstructed after the process group is destroyed.")`
- Backend mismatch between `fit()` and `transform()` → `RuntimeError` spelling it out.

### Deliberate non-handling

- **Cross-version torchrec/FBGEMM drift** — pinned dep, patches via `_compat.py` if needed.
- **Silent fallback to replicated embeddings** — never. The user picked this backend for a reason.
- **Async / overlapped checkpoint writes** — synchronous saves only in v1. DCP's async API is a follow-up.

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

**Layer 5 — Checkpoint round-trip (`multigpu` marker)**
- *Save → reload preserves embeddings exactly.* Train 2 epochs with `checkpoint_dir=tmp`, capture extracted embeddings, simulate restart by tearing down + re-launching with the same `checkpoint_dir`, run 0 additional epochs, re-extract embeddings; assert bit-equal across the round trip.
- *Auto-resume continues from saved step.* Train 2 epochs writing to `tmp`, then re-invoke `fit()` with `epochs=4` and the same `checkpoint_dir`; assert training resumed from epoch 2 (loss curve length, logged epoch counter), and final embeddings match a single 4-epoch run within numerical noise.
- *Resharding.* Save at `world_size=2`, reload at `world_size=1` (single-GPU degenerate sharding); assert load succeeds and extracted embeddings match the 2-GPU output.
- *Overwrite escape hatch.* `checkpoint_overwrite=True` against a populated `checkpoint_dir` wipes it and starts from epoch 0.
- *Vocab-mismatch guard.* Save against vocab A, attempt resume against vocab B; assert `RuntimeError` with the documented message and no partial state mutation.
- *`keep_last` pruning.* Run with `checkpoint_interval_minutes` low enough to trigger multiple time-based saves; assert exactly `checkpoint_keep_last` step directories remain after training.

### CI integration

- Layers 1 + 2 run on the existing GPU CI runner.
- Layers 3 + 4 are opt-in; contributors with a 2-GPU box run them via `torchrun … pytest -m multigpu`. Release process for any version touching torchrec code requires a manual multi-GPU run.

### Not tested

- FBGEMM kernel correctness (assumed; flag upstream if broken).
- Torchrec planner internals.
- DCP shard-format internals (assumed; flag upstream if broken).
- Scaling beyond 2 GPUs (algorithm is the same; correctness at world_size=2 covers it).
- Async checkpoint writes (deferred follow-up).

## 7. Follow-ups (explicitly deferred)

- ~~**`fit_from_walks(walks_path, vocab_path)` entry point**~~ — **DONE** (see the
  2026-06-12 update). The torchrec backend required the decoupling, so
  `prepare_corpus` + `fit_from_walks` landed here rather than as a separate PR.
  Generalising the same seam across the `pytorch`/`gensim` backends is still
  open.
- **Removing the existing `backend="gensim" + walk_strategy="bfs"` restriction** (same decoupling principle); separate PR.
- **`grad_clip_mode="row"` parity** for the torchrec path — FBGEMM's fused clip
  is element-wise (`max_gradient`), so it matches `grad_clip_mode="element"`
  (the default) but not the L2-per-row mode. Add row-norm clipping if/when a
  user needs it.
- **Async / overlapped data loading** for the torchrec pair loader — currently
  `num_workers=0` (forked CUDA workers abort, and `spawn` can't pickle the
  nested dataset). A picklable, spawn-safe loader would restore I/O overlap.
- **CBOW and order-aware skip-gram** under sharding.
- **CPU/UVM offload** for tables exceeding aggregate VRAM.
- **Custom `ParameterConstraints`** if torchrec's planner picks suboptimally at some specific scale we haven't observed yet.
- **Async checkpoint writes** via `DCP.async_save` — overlap save IO with subsequent training steps. Mature follow-up once the synchronous path is proven.
- **Cross-backend checkpoints** — current design checkpoints are torchrec-specific. Sharing checkpoint format with the pytorch / gensim backends would require a round-trip through a dense `(vocab × dim)` representation; not worth the complexity in v1.
