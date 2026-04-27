# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] — 2026-04-27

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

### Added

- `_build_walk_steps(vertices_s, edge_attrs_s, walk_length, walk_id_offset)`
  in `rdf2vecgpu.corpus.walk_corpus` — a per-partition reshape helper that
  consumes cuGraph's flat `(vertices, edge_attrs)` walk output and emits
  the `[src, predicate, dst, walk_id, step]` schema downstream code already
  consumes. Drops cuGraph's `-1` sentinel rows for early-terminating walks.
- `test/corpus/walk_corpus_test.py` — regression tests for the reshape
  helper layout, walk_id offsetting, sentinel handling, stride validation,
  and an integration smoke that asserts emitted predicates exist between
  the (src, dst) pair the walker actually stepped on.
- `gpu` pytest marker registered in `pyproject.toml`'s
  `[tool.pytest.ini_options]`.
