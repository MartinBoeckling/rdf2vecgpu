# Contributing to gpuRDF2vec

Thanks for helping improve gpuRDF2vec. This page covers the development setup, the tests and
docs, and two Dask patterns the multi-GPU code depends on.

## Workflow

1. Open an issue first for bugs and larger changes. The README's
   [Report issues and bugs](README.md#report-issues-and-bugs) section lists what to include.
2. Fork the repository and branch off `main`.
3. Make the change with tests, and add an entry under `## [Unreleased]` in
   [`CHANGELOG.md`](CHANGELOG.md).
4. Open a pull request that references the issue.

## Development setup

The package needs Linux (or WSL2), Python 3.12 or newer and an NVIDIA GPU with a CUDA 12.x driver.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test,gensim]"
```

With `uv`: `uv sync --extra test --extra gensim`.

## Tests

Tests that need a GPU are marked `gpu` and only run with `--gpu`:

```bash
pytest test                                     # CPU-only tests
pytest --gpu test --ignore-glob="test/e2e_*"    # unit tests on the GPU
pytest --gpu test                               # everything, including end-to-end tests
```

The multi-GPU end-to-end tests need at least two GPUs.

## Documentation

The docs are built with Sphinx and published on Read the Docs. Warnings fail the build:

```bash
python -m pip install sphinx -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Sphinx mocks the GPU libraries (`autodoc_mock_imports` in `docs/source/conf.py`), so the build
runs without a GPU. Type hints such as `cudf.DataFrame | dask_cudf.DataFrame` fail under the
mocks unless the module starts with `from __future__ import annotations`.

## Distributed correctness: when to `persist`

The multi-GPU code paths use `dask_cudf` for the vocabulary build, the broadcast-encode of edge
triples and the random-walk reshape. Two patterns are easy to get wrong and produce silent data
drift rather than visible crashes. Both are also documented inline at the call sites.

**(a) Any non-deterministic Dask output that is read more than once must be `.persist()`-ed
before the second consumer.** Examples:

- `MultiGPUWalkCorpus.random_walk` and `MultiGPUWalkCorpus.random_walk_sequences` persist the
  cuGraph walk output before they compute per-partition sizes for the reshape (see the
  `walks_s.persist()` calls in `corpus/walk_corpus.py`).
- `_generate_vocab` persists the post-shuffle deduplicated vocabulary before reading partition
  sizes for the `cumsum` offsets that drive the per-partition `cupy.arange` token assignment (see
  the `vocabulary_df.persist()` call in `helper/functions.py`).

Without persist, every consumer triggers a re-roll of the underlying hash-shuffle or random-walk
generator. Because `drop_duplicates` over a non-deterministic shuffle is itself
non-deterministic, partition assignments drift between rolls, and rows are silently lost between
a partition-sizes pass and the eventual write.

**(b) When using `.merge()` against a small table, prefer `broadcast=True`.** Without it, Dask
defaults to a hash-shuffle join. With a small right-side partition count, the large left side
gets funneled through a single worker, which turns the vocabulary build of a large graph into
hours of single-GPU work. `broadcast=True` replicates the small side to every worker so each
large-side partition does a local hash join with no shuffle.

## Compatibility patches

`_compat.apply_patches()` (called from `GPU_RDF2Vec.__init__` when `multi_gpu=True`) fixes two
upstream issues that are still present in the cuGraph 25.6 / dask >= 2025.5 versions the package
pins:

- `_patch_convert_to_cudf`: `cugraph.dask`'s random-walk `convert_to_cudf` helper crashes on
  `renumber=False` graphs because it unconditionally dereferences
  `number_map.implementation.numbered`.
- `_patch_dask_cudf_from_cudf`: works around a dask-expr backend breakage in dask >= 2025.5 that
  affects `dask_cudf.from_cudf`.

Both patches are self-contained workarounds; the long-term fix is upstream in cuGraph and
dask-cudf. Track those there and remove the patches once they land.
