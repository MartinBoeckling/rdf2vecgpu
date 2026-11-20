# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-11-20

Faster training and closer parity with jRDF2Vec. PyTorch skip-gram trains
from the walks as sequences on the GPU instead of expanding them into pairs
up front, trains like word2vec (SGD, linearly decaying learning rate,
subsampling) and gains CBOW and order-aware skip-gram on walk sequences.
Walks can be generated once and trained on later (`prepare_corpus` /
`fit_from_walks`). On FB15k-237 (100 walks, depth 4, 5 epochs) PyTorch on
one RTX A6000 now takes 43 s end to end, 3.1x faster than jRDF2Vec with 16
CPU threads on the same machine; 0.5.0 took 594 s, 4.6x slower than
jRDF2Vec. Across FB15k-237 runs of 100–1000 walks and depth 4–12, 0.6.0 is
2.9–3.6x faster than jRDF2Vec and 40–90x faster than pyRDF2Vec (see the
performance page). Walk corpora larger than cuDF's 2^31-element column limit
now train on one GPU.

### Added

- `prepare_corpus(edge_df, out_dir)` writes the walks (Parquet, one row per
  walk with `vertices` / `predicates` token lists) and `word2idx.parquet`;
  `fit_from_walks(walks_path, vocab_path)` trains on them without a graph or
  `load_data`, with PyTorch on one GPU or DDP across the node's GPUs, or with
  gensim. This is jRDF2Vec's `-onlyWalks` / `-onlyTraining`: walks can be
  generated once and trained on several times, and multi-GPU training needs
  no Dask client, since the processes Lightning starts per GPU only read the
  files. `GPU_RDF2Vec(multi_gpu=True)` no longer requires a client at
  construction; `read_data` / `load_data` raise without one.
- `embedding_model="order_aware_skipgram"` (RDF2Vec_oa, Ling et al. 2015):
  one output embedding per position in the context window. PyTorch and random
  walks only.
- PyTorch CBOW with negative sampling on walk sequences (gensim's `sg=0`,
  `cbow_mean=1`), on one GPU, with DDP and via `fit_from_walks`.
- New `RDF2VecConfig` options:
  - `sample` (default 0.001): frequent-token subsampling as gensim's
    `sample`; also passed to the gensim backend.
  - `min_learning_rate` (default 0.0001): the learning rate decays linearly
    to it over training (gensim's `min_alpha`).
  - `grad_clip_value` (default 1.0, 0 turns it off): per-component gradient
    clip of the PyTorch models.
  - `shrink_window` (default False): draw each centre token's window
    uniformly from 1..`window_size`, as word2vec and gensim do. About a third
    fewer pairs and 30-36% less training time on FB15k-237, but lower
    embedding quality in our checks, so it is off by default.
- BFS walks on one GPU are walk sequences like random walks (one
  root-to-leaf path per leaf of each root's BFS tree), so they train with the
  same sequence trainers and also work with gensim and `prepare_corpus`.

### Changed

- **`batch_size` means real training pairs per step.** It defaults to 32768
  (Lightning's batch-size tuner and its 8192 cap are gone). On one process,
  steps are sized from the pairs training actually builds after walks that
  end early and `sample` subsampling, estimated before training, so a step
  holds about `batch_size` pairs at every walk depth. Before, the count
  assumed full-length walks that kept every token, so a step held ~15k pairs
  at depth 4 and ~8k at depth 12, and deeper walks took more steps: the same
  `batch_size` now gives fewer, larger steps. DDP keeps the full-walk bound,
  because all ranks must agree on the step count. A step uses at most 1/100
  of the training pairs per GPU (at least 100 steps per epoch); an explicit
  `batch_size` reduced by this rule is logged as a warning.
- PyTorch skip-gram, CBOW and order-aware skip-gram train like word2vec /
  gensim: summed loss, sparse SGD with a linearly decaying learning rate,
  unigram^0.75 negatives, output tables initialised to zero, and negatives
  equal to the target left out of the loss. `learning_rate` defaults to
  0.025 for every model; only PyTorch CBOW on BFS walks keeps Adam with
  0.0001.
- Tokens that occur in no walk are not trained and come back from
  `transform()` as zeros, also after `save()` / `load()`
  (`walked_tokens.npy`).
- `cpu_count` (gensim threads) defaults to the CPU cores available to the
  process instead of 4.
- Single-GPU walks are generated in chunks of at most 25 M walks per cuGraph
  call and kept as int32 matrices. PyTorch trains on them in GPU memory; gensim
  and `prepare_corpus` write each chunk straight to Parquet.
- Multi-GPU PyTorch CBOW is only refused on BFS walks now (it trains on
  random walks).

### Deprecated

- `tune_batch_size` is ignored and raises a `FutureWarning`; it will be
  removed in 0.7.0. Set `batch_size` instead.

### Removed

- The looped `OrderAwareSkipgram` / `OrderAwareCBOW` models and their
  DataModules, which the config could not reach.
- The ISWC 2025 benchmark scripts and their pinned environment files from
  `performance/`, replaced by the end-to-end harness (`benchmark.py`,
  `benchmark_charts.py`). They remain in the `v0.5.0` tag.

### Fixed

- The PyTorch gradient clip of 0.1 per component cut the updates of frequent
  tokens (predicates, hub entities) by orders of magnitude, so embedding
  quality dropped below gensim's. With 1.0, on the KGrEAT tasks on DBpedia
  2016 (RDF2Vec Light, 3 seeds), skip-gram matches or beats gensim on all ten,
  and CBOW and order-aware skip-gram are on par with gensim and wang2vec's
  structured skip-gram (see the performance page of the docs).
- Single-GPU random walks on large graphs ran out of GPU memory within
  seconds, for both trainers, or exceeded cuDF's 2^31-element column limit.
- Single-GPU BFS walks ran out of GPU memory on FB15k-237 and
  random_graph_10000 (depth 4).
- A DDP batch whose pairs were all subsampled away detached the loss from
  the embeddings, which left that rank out of the gradient all-reduce and hung
  the others until the NCCL timeout.
- Start vertices passed to `fit()` are cast to the graph's vertex dtype
  (token ids of a small vocabulary are int8, which cuGraph rejects).

### Performance

End to end on one RTX A6000, median of 3 runs, 5 epochs, 16 CPU threads for
the CPU trainers, seconds, all on the same machine (0.5.0 ran out of GPU
memory at 500 walks of depth 8 and at 1000 walks of depth 4):

| FB15k-237, random walks | 0.6.0 PyTorch | 0.6.0 gensim | 0.5.0 PyTorch | jRDF2Vec | pyRDF2Vec |
|---|---|---|---|---|---|
| 100 walks, depth 4 | 42.6 | 82.1 | 593.6 | 130.2 | 1,728 |
| 100 walks, depth 12 | 71.1 | 131.4 | 1,291.3 | 217.1 | 3,811 |
| 500 walks, depth 4 | 160.1 | 355.8 | 2,955.6 | 574.0 | 12,226 |
| 1000 walks, depth 8 | 491.6 | 985.5 | out of memory | 1,624.7 | 35,500 |

- Training pairs are shuffled and batched on the GPU instead of through a
  Python index list per epoch (pair-based paths: ~95 s on FB15k-237 whatever
  the batch size).
- Steps sized by real pairs (see `batch_size` above): skip-gram training on
  FB15k-237 30.6 → 18.0 s at depth 4 and 108.4 → 36.4 s at depth 12.
- The PyTorch models log the loss and redraw the progress bar only on
  logging steps, and a batch's pairs are filtered with one index, so the host
  waits for the GPU once per batch (~20% less training time).
- The gensim backend reads walk batches as numpy token matrices instead of
  building strings walk by walk in Python: one pass over 1.45 M walks
  14.0 → 1.1 s at depth 4.
- Unigram negatives are drawn with `torch.searchsorted` on a CDF built once
  instead of `torch.multinomial` over the whole vocabulary every step.
- BFS on FB15k-237 depth 4: 1,586 s end to end.

## [0.5.0] — 2026-11-09

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
- The `gensim` extra requires `gensim>=4.4.0`, the first release that
  supports numpy 2.
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
- N-Quads input (`.nq`) kept the graph term in the object column
  (`<a> <b> <c> <g> .` gave the object `c> <g`); it is now dropped.
- `fit()` without `load_data()` (for example after `read_data()`, which only
  reads the file) failed with a misleading "call fit() first"; it now says to
  call `load_data()`, which builds the vocabulary and the graph.
- `read_kwargs={"columns": ...}` for parquet input raised a duplicate-keyword
  `TypeError`; it is now rejected with a message pointing to `col_map`.
- Multi-GPU graph construction recomputed the edge list (file read and
  vocabulary merges) for each of its steps; it is now persisted once.
- Exporting gensim vectors copied them one key at a time in Python; the copy
  is now one array assignment. With word keys (`int_tokens=False`), keys
  that look like integers (e.g. the literal "1990") are looked up as words.
- `fit_transform(edge_df)` raised a `TypeError` because `walk_vertices` had no
  default; it now defaults to `None` like `fit`.
- `backend="gensim"` passed the PyTorch default `learning_rate=0.0001` to
  gensim as `alpha`, 250x below gensim's own default of 0.025, so the
  vectors barely moved from their random start. `learning_rate` now defaults
  to `None` and resolves per backend: 0.0001 for `"pytorch"`, 0.025 for
  `"gensim"`. An explicitly set value is used unchanged.
- `backend="gensim"` crashed in `fit()` on a single GPU with
  `AttributeError: DataFrame object has no attribute persist`: a cuDF frame was
  mistaken for a Dask one when the walks were written to Parquet.
- Single-GPU training ran Lightning's batch-size tuner whenever
  `tune_batch_size=True` (the default) and replaced an explicit
  `batch_size` with its result. The tuner now only runs when `batch_size` is
  `None`.

### Documentation

- All examples (README, getting started, training, data loading, embedding
  retrieval) now call `load_data()`, which builds the vocabulary and the
  graph, instead of `read_data()`, which only reads the file.
- Installation: a CUDA GPU and Linux (or WSL2) are required, Python >= 3.12;
  RAPIDS and PyTorch are installed as dependencies (the README's RAPIDS 25.4
  instructions conflicted with the `cugraph-cu12==25.6.*` pin); new
  `gensim` extra and multi-GPU setup notes.
- `RDF2VecConfig` reference: corrected defaults (`learning_rate=0.0001`,
  `reproducible=False`), documented `backend`, `tracker`, `tracker_kwargs`
  and `tracker_run_name`, and what `batch_size`, `cpu_count`, `min_count`
  and `reproducible` actually do.
- `GPU_RDF2Vec` docstrings rewritten to match the code (`read_data`,
  `load_data` formats, `fit`, `transform`, `save`/`load`).
- Data loading: N-Triples/N-Quads use the cuDF reader only, CSV needs a
  header, `literal_predicates` are matched without angle brackets for `.nt`
  input; tracking and training pages list the actual stage names.
- The API reference is linked from the navigation; the stale `docs/source/api/`
  pages of the old `src.*` layout are removed, and the gensim backend module
  is documented. `walk_corpus` and `gpu_rdf2vec` failed to import under the
  Sphinx mocks, so their API pages were empty; the docs now build with
  `sphinx-build -W`.
- README links and the header image use absolute GitHub URLs so they work on
  PyPI.
- The README's "Implementation Details" section and the list on the docs landing
  page are replaced by a short "How it works" section based on the code. It drops claims the code does not support
  (a GPU-memory batch-size rule, all-reduce at fixed intervals, linear scaling)
  and figures without a source.
- The contributor notes (the Dask `persist` / `broadcast=True` patterns and the
  `_compat` patches) moved from the README to a new `CONTRIBUTING.md`, which
  also covers the development setup, the tests and the docs build.

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
