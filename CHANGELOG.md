# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-10-30

Walk-generation fixes and a chunked multi-GPU walk writer. 0.4.0 was not
published to PyPI, so this is the first PyPI release that also contains the
0.4.0 changes below.

### Added

- `MultiGPUWalkCorpus.write_walk_sequences(...)` — multi-GPU random walks
  written straight to a walks-as-sequences parquet directory in chunks, for
  corpora that do not fit in GPU memory at once. Start vertices are split
  across the GPUs once; each chunk is one task per GPU that walks, builds the
  list columns and writes its own parquet files in the background while the
  next chunk walks, so only walk counts reach the client.
  `walk_generation_sequences(..., out_dir=...)` uses it in multi-GPU mode and
  returns a `WalkParquet(path, n_walks)`, and the multi-GPU gensim `fit()`
  trains from the files the GPUs wrote.

### Changed

- Walk list columns (`vertices`, `predicates`) are assembled on the GPU from
  an offsets column instead of `cudf.Series(matrix.tolist())`, which
  round-tripped every element through a Python object (~36 s per 25 M walks
  of depth 4, now ~5 ms). The parquet schema is unchanged.
- Walk predicates are int64 vocabulary tokens, weighted and unweighted
  (previously float, from cuGraph's weight slot).
- `walk_weighted=True` rejects missing, negative or non-finite edge weights
  and converts integer weights to float32, and a graph whose weights are all
  0. A weight-0 edge is never walked; a vertex whose out-edges all weigh 0
  ends the walk. BFS walks ignore the weights.
- Triples with a missing subject, predicate or object are dropped before the
  vocabulary is built.
- Multi-GPU CBOW with `backend="pytorch"` is rejected by `RDF2VecConfig`: its
  training pairs lost the context windows, so CBOW trained on one context
  word per sample. Use `backend="gensim"` or a single GPU.
- `write_walk_sequences` refuses an `out_dir` that already holds parquet
  files, which readers would otherwise mix with the new walks.
- Multi-GPU gensim training checks that the walk files it reads hold every
  walk written. The Dask workers write them, so all workers must run on the
  client's node.
- `generate_artifact=True` also writes `vector/word2idx.parquet`, as
  documented, and creates `vector/`.

### Fixed

- Walk predicates were rounded for vocabularies above 2^24 tokens: cuGraph
  carries the edge attribute in a float32 weight slot, so token 16,777,217
  came back as 16,777,216. The graph is now built on compact predicate codes
  1..N (`encode_predicates`) and walk output is decoded back to tokens.
- Weighted walks (`walk_weighted=True`) emitted each step's edge weight as
  its predicate, so training failed on float token indices (since 0.4.0,
  which stopped looking predicates up in the edge list as 0.3.0 did).
  cuGraph's biased walks return only the weight, so the predicate is now
  recovered from the edge list by (subject, object, weight), drawn uniformly
  among parallel edges of equal weight. This matches the walker's edge
  choice exactly, including on parallel edges where 0.3.0 picked an
  arbitrary predicate. Multi-GPU skip-gram/CBOW walks look predicates up in
  the distributed edge list; the multi-GPU walk-sequence paths (gensim) keep
  a copy of the edge list on every GPU, so that list has to fit on one GPU.
- BFS walks (`walk_strategy="bfs"`) failed with a `TypeError`: cuGraph's
  `bfs()` only accepts a `cugraph.Graph`, not the `MultiGraph` the walks use.
  BFS now runs on a `Graph` of the distinct (subject, object) pairs.
- BFS walks were also out of order and duplicated steps over parallel
  edges. Each walk's edges were sorted by subject token rather
  than by depth, so a path whose vertex ids do not increase came out
  scrambled and did not start at its root; and the predicate was merged back
  by (subject, object), repeating a step once per parallel edge. Walks now
  follow BFS-tree depth, and over parallel edges one predicate is drawn
  uniformly (seeded by `random_state`), one walk per root-to-leaf path.
- Single-GPU weighted walks crashed (`overall_num_elements must be > 0`,
  raised by cuGraph's `biased_random_walks`) whenever every walk of a call
  ended two or more steps before `walk_depth`, e.g. on graphs with many dead
  ends. Such calls are now walked one step at a time, which gives the same
  walk distribution. The multi-GPU walker is not affected.
- Multi-GPU weighted walks dropped the `weights` column during vocabulary
  encoding and could not build the graph.
- Multi-GPU random walks started from only a fraction of the vertices:
  cuGraph's MG `.nodes()` on a `renumber=False` graph returns a subset of the
  source vertices, and a `dask_cudf.Series` of start vertices made cuGraph
  walk only ~1/n_workers of them. Walks now start from the subjects of the
  edge list (`construct_graph` stores them), passed to cuGraph as one
  `cudf.Series`.
- Multi-GPU `random_walk_sequences` failed dask's meta check on newer cuDF
  (list vs object dtype); the meta is now the empty head of the first
  persisted partition.
  With pyarrow strings enabled (dask's default) it also failed in
  `from_delayed` (`issubclass() arg 1 must be a class`), which is now
  called with string conversion off.
- Multi-GPU PyTorch training could hang: `ParquetSkipGramDataModule` gave
  each DDP rank whole files, so ranks ran different numbers of batches and
  the gradient all-reduce blocked. Each rank now reads a disjoint block of
  `total_rows // world_size` rows. On one node, Lightning now spawns the
  per-GPU workers itself; inside a SLURM allocation with `--ntasks=1` it
  previously ran DDP as a single rank. When SLURM starts one task per GPU,
  the launch stays with SLURM.
- The single-GPU vocabulary paired tokens with the wrong words after a
  missing word (`factorize` codes it -1 but leaves it out of the uniques),
  so every later embedding was labelled with the next word.
- Seeded predicate draws among parallel edges (weighted walks, BFS) depended
  on the row order of a cuDF merge and were not reproducible.
- Multi-GPU N-Triples input failed: `dask.dataframe` has no `read_text`; it
  now uses `dask_cudf.read_text`.
- Parquet input with a `col_map` other than the identity raised `KeyError`:
  the reader took the map's values as file columns, the rename its keys.
  `col_map` maps file columns to `subject`/`predicate`/`object`/`weights`,
  as documented.
- `save()` failed with `backend="gensim"` (gensim models have no
  `state_dict`); gensim models are now saved as `model.gensim` and restored
  by `load()`.
- `transform()` with `generate_artifact=True` failed when `./vector/` did
  not exist.
- Multi-GPU graph construction recomputed the edge list (file read and
  vocabulary merges) for each of its steps; it is now persisted once.
- Exporting gensim vectors copied them one key at a time in Python; the copy
  is now one array assignment. With word keys (`int_tokens=False`), keys
  that look like integers (e.g. the literal "1990") are looked up as words.

### Known issues

- Multi-GPU PyTorch training on one node: Lightning starts the user's script
  again for every further GPU, so each process repeats data loading and walk
  generation and trains on its own walks.
- Before training, multi-GPU `fit()` closes the Dask client but not the
  cluster, so the workers keep their GPU memory (e.g. an RMM pool).

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
