# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-04-27

This release bundles a four-PR upstream-port sequence that ports
Wikidata-scale fixes and a new optional gensim training backend back into
the library. The pipeline now scales to the 1.38 B-edge / 390 M-vertex
Wikidata graph end-to-end on 2× RTX A6000.

### Added

- **Optional gensim Word2Vec backend.** New
  `backend="gensim"` config field on `RDF2VecConfig` (alongside the
  default `"pytorch"`); `GPU_RDF2Vec.fit()` dispatches on it. The gensim
  path consumes walks-as-sequences via streaming parquet
  (`pyarrow.parquet.ParquetFile.iter_batches`) and trains via
  `gensim.models.Word2Vec`'s C-level negative-sampling loop — typically
  5–10× faster on CPU than the equivalent PyTorch skip-gram on the same
  corpus, with constant-memory streaming on multi-billion-walk corpora.
  Opt in via `pip install rdf2vecgpu[gensim]`.
- `embedders/gensim_word2vec.py` — new module with three exports:
  - `WalkCorpus`: streaming parquet iterator over walk sequences (trims
    cuGraph `-1` sentinels, skips walks with <2 valid vertices).
  - `load_id_to_word`: scatters `_generate_vocab`'s contiguous token range
    into a numpy object array indexed by token (avoids the ~45 GB
    Python-dict overhead the original gensim recipe measured at Wikidata
    scale).
  - `train_gensim_word2vec`: thin wrapper around `gensim.models.Word2Vec`
    forwarding the relevant `RDF2VecConfig` fields.
- `_walks_to_lists` helper in `corpus/walk_corpus.py` — sister to
  `_build_walk_steps` (added in this release) that emits per-walk rows
  with `vertices: list<int>` + `predicates: list<int>` columns. Used by
  the new `random_walk_sequences()` methods on both `SingleGPUWalkCorpus`
  and `MultiGPUWalkCorpus`.
- `GPU_RDF2Vec.walk_generation_sequences()` and
  `GPU_RDF2Vec._fit_via_gensim()` orchestration methods on the main
  pipeline class.
- `_build_walk_steps(vertices_s, edge_attrs_s, walk_length, walk_id_offset)`
  in `rdf2vecgpu.corpus.walk_corpus` — a per-partition reshape helper that
  consumes cuGraph's flat `(vertices, edge_attrs)` walk output and emits
  the `[src, predicate, dst, walk_id, step]` schema downstream code already
  consumes. Drops cuGraph's `-1` sentinel rows for early-terminating walks.
- `_assign_ids(partition, offset)` private helper in
  `rdf2vecgpu.helper.functions` — per-partition `cupy.arange + offset` used
  by the new multi-GPU vocab builder.
- `test/corpus/walk_corpus_test.py` — regression tests for the reshape
  helper layout, walk_id offsetting, sentinel handling, stride validation,
  and an integration smoke that asserts emitted predicates exist between
  the (src, dst) pair the walker actually stepped on.
- `test/helper/functions_test.py` —
  `test_generate_vocab_multi_gpu_unique_tokens_contiguous_range` (catches
  off-by-one in cumsum offsets) and
  `test_generate_vocab_multi_gpu_round_trip_decodes_to_original_strings`
  (catches row drift from broadcast merges or unstable shuffles).
- `gpu` pytest marker registered in `pyproject.toml`'s
  `[tool.pytest.ini_options]`.
- `test/embedders/gensim_word2vec_test.py` — CPU-only tests for the
  new gensim backend (streaming `WalkCorpus` semantics, sentinel
  trimming, contiguous-token guard in `load_id_to_word`, end-to-end
  smoke through `train_gensim_word2vec`). Skipped at module import if
  the `gensim` extra isn't installed.
- `[project.optional-dependencies].gensim` group in `pyproject.toml`
  (`gensim>=4.3.0`, `pyarrow>=15`).

### Documentation

- New "Distributed correctness — when to `persist`" subsection under
  Implementation Details in the README. Captures the two
  contributor-facing dask patterns that produce silent data drift if
  ignored (persist-before-second-consumer; `broadcast=True` on
  small-table merges), with concrete cite-points in `walk_corpus.py`
  and `helper/functions.py`. Also documents the two `_compat` patches
  (`_patch_convert_to_cudf` and `_patch_dask_cudf_from_cudf`) and notes
  that the long-term fix for both is upstream in cuGraph / dask-cudf.
- Roadmap entry for the optional gensim Word2Vec trainer backend
  (now shipped in this same `0.4.0` — see "Optional gensim Word2Vec
  backend" under Added).

### Changed

- **Multi-GPU vocab build: hash-partition rewrite (replaces categorize
  funnel).** The previous `dd.concat(s, p, o).unique()` followed by
  `vocabulary_df.categorize(columns=["word"])` funneled the full
  deduplicated vocabulary onto a single worker for `sort_values()`. On
  Wikidata-scale graphs (~390 M unique tokens) the single-worker sort
  exceeded 40 GB of intermediate state and either OOM'd the GPU or hung
  the worker. The replacement is embarrassingly parallel:
  `shuffle(on="word", shuffle_method="tasks") → drop_duplicates() →
  persist+wait → cumsum offsets → dask.delayed _assign_ids` per partition
  with `cupy.arange + offset`. Token ids remain globally unique and form a
  contiguous `[0, n)` range without any cross-partition shuffle. Result on
  2× RTX A6000: ~13 min for the 390 M-token Wikidata vocab vs.
  never-completes with the previous code path.
- **Multi-GPU vocab build: persist+wait stabilizes the shuffle output.**
  Without persist, the hash-shuffle re-rolls every time `sizes.compute()`
  or the per-partition id assignment reads from it, and partition
  assignments drift between rolls (the standalone tool that pioneered this
  pattern observed 23 M → 15 M row loss between sizes.compute and the
  eventual write). Persist + `dask.distributed.wait()` pins the shuffle
  output before downstream consumers read it.
- **Encode-side merges now use `broadcast=True`.** The three `.merge()`
  calls that join `word2idx` into the edge table didn't specify
  `broadcast=True`, so dask defaulted them to hash-shuffle joins. With a
  small word2idx partition count, the 1.38 B-row edge side got funneled
  through one worker — observed 6+ hours at 100 % single-GPU utilization
  with no progress on Wikidata-scale runs. `broadcast=True` replicates the
  small word2idx to every worker so each edge partition does a local hash
  join with no shuffle of the large side; one hash-table build per worker
  per merge instead of one per partition.

### Fixed

- **Walk corpus: predicates now match the edge cuGraph actually traversed.**
  Both `SingleGPUWalkCorpus.random_walk` and `MultiGPUWalkCorpus.random_walk`
  used to discard the `edge_attrs` return from `cugraph.uniform_random_walks`
  / `biased_random_walks` and re-derive the per-step predicate via a
  `(src, dst)` merge against the edge table. On `cugraph.MultiGraph`
  instances with parallel edges (multiple predicates between the same
  vertex pair) the merge was non-unique on the right side — cuDF picked an
  arbitrary matching predicate per row, so walks could end up annotated
  with a predicate the walker never traversed. Now the `edge_attrs` return
  is consumed directly via the new `_build_walk_steps` reshape helper.
- **Multi-GPU walks: no more silent row loss at dask-partition boundaries.**
  The previous `_build_walk_df` closure ran inside `map_partitions` and did
  `df["dst"] = df["src"].shift(-1); df = df.iloc[:-1]`, dropping the last
  row of every dask partition rather than the last row of every walk.
  Because partitions don't respect walk boundaries, a walk that started in
  partition N and ended in partition N+1 lost its final step or terminal
  vertex. The new reshape works on stride-aligned (vertex, edge_attr)
  pairs directly, so no partition-boundary trim is needed.
- **Multi-GPU walks: walk_ids are now globally unique across partitions.**
  The previous `cp.arange(len(df)) // max_len` ran per-partition and
  produced walk_ids `0..n_walks_in_partition`, which collided across
  partitions. The downstream skip-gram pair builder does a dask merge on
  `(walk_id, pos)`, so collisions silently created spurious cross-walk
  pairs. The reshape helper now takes a `walk_id_offset`, and the multi-GPU
  caller computes per-partition cumulative-walk offsets so ids are globally
  unique.

### Performance

- **Multi-GPU walks: removed a multi-billion-row merge.** The
  `walks.merge(edge_ddf, ...)` predicate-recovery step is gone. On
  Wikidata-scale graphs (1.38 B edges, 390 M vertices, walks_per_vertex=10,
  walk_length=8) the merge was the pipeline's worst-scaling stage after
  vocab generation. The reshape pattern is O(n_walks).
