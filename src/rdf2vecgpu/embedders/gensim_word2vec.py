"""Optional gensim Word2Vec backend for the RDF2Vec training stage.

Used by `GPU_RDF2Vec` when ``RDF2VecConfig(backend="gensim")`` is set. The
walks are still generated on the GPU; gensim then trains on the CPU.

Why this exists alongside the default PyTorch trainer:

* gensim's C implementation of negative-sampling Word2Vec is a mature,
  widely used CPU trainer.
* `pyarrow.parquet.ParquetFile.iter_batches` streams the walk corpus from
  Parquet with constant memory; the PyTorch path holds the whole set of
  training pairs in GPU memory.
* On machines with many CPU cores and few or small GPUs, the GPU is only
  needed for walk generation and is free again during training.

This module needs the ``gensim`` optional-dependency group::

    pip install "rdf2vecgpu[gensim]"
"""
from __future__ import annotations

import glob
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# gensim and pyarrow are optional dependencies. Import lazily so users on the
# default pytorch backend don't need them installed; raise a friendly error
# only when the gensim path is actually invoked.
try:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from gensim.models import Word2Vec

    _GENSIM_AVAILABLE = True
    _GENSIM_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - optional dep
    _GENSIM_AVAILABLE = False
    _GENSIM_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from gensim.models import KeyedVectors  # noqa: F401


def _require_gensim() -> None:
    if not _GENSIM_AVAILABLE:
        raise ImportError(
            "The gensim training backend requires `gensim` and `pyarrow`. "
            "Install with: pip install rdf2vecgpu[gensim] "
            f"(original import error: {_GENSIM_IMPORT_ERROR!r})"
        )


def load_id_to_word(word2idx_df) -> np.ndarray:
    """Convert word2idx → numpy object array indexed by int32 token.

    `_generate_vocab` produces tokens as a contiguous `[0, n)` range, so we
    scatter the word column into a dense numpy object array indexed by token.
    Avoids the ~45 GB Python-dict overhead the standalone tool's first-version
    used; numpy fancy-indexing (`arr[[1,2,3]]`) is one vectorised call per walk.
    """
    if hasattr(word2idx_df, "compute"):
        # dask_cudf.DataFrame
        word2idx_df = word2idx_df.compute()
    if hasattr(word2idx_df, "to_pandas"):
        # cudf.DataFrame
        word2idx_df = word2idx_df.to_pandas()
    tokens = word2idx_df["token"].to_numpy()
    n = int(tokens.max()) + 1
    if n != len(tokens):
        raise ValueError(
            f"word2idx tokens are not contiguous [0, n) "
            f"(rows={len(tokens):,}, max={n - 1:,})"
        )
    words = np.empty(n, dtype=object)
    words[tokens] = word2idx_df["word"].to_numpy()
    return words


def _list_matrix(column) -> np.ndarray | None:
    """A list<number> column as an int64 ``[rows, list length]`` matrix, or
    None when its lists differ in length or one is null. Null and NaN values
    become -1; floats are truncated like ``int()``."""
    n = len(column)
    if column.null_count:
        return None
    if n == 0:
        return np.empty((0, 0), dtype=np.int64)
    bounds = pc.min_max(pc.list_value_length(column))
    width = bounds["min"].as_py()
    if width != bounds["max"].as_py():
        return None
    values = column.flatten()
    if values.null_count:
        values = pc.fill_null(values, -1)
    flat = values.to_numpy(zero_copy_only=False)
    if flat.dtype.kind == "f":
        flat = np.where(np.isnan(flat), -1, flat)
    return flat.astype(np.int64, copy=False).reshape(n, width)


class WalkCorpus:
    """Stream walks-parquet → interleaved URI sequences for gensim.

    Each parquet file under `walks_dir` has the columns

    - ``vertices``: ``list<int>`` of length ``walk_depth + 1``
    - ``predicates``: ``list<int>`` of length ``walk_depth``

    as written by `MultiGPUWalkCorpus.write_walk_sequences` or
    `MultiGPUWalkCorpus.random_walk_sequences`. This iterator emits interleaved sequences
    `[v0, p0, v1, p1, v2, ..., v_n]` per walk — the format gensim's
    `Word2Vec(sentences=...)` constructor consumes.

    cuGraph emits `-1` for vertex slots past the walk's actual end (dead-end
    hit before max_depth). We trim each walk at the first `-1` and keep the
    matching predicate prefix; walks with fewer than 2 valid vertices are
    skipped (need at least one edge for skip-gram).

    Each parquet batch of ``batch_size`` walks is converted with numpy as one
    ``[walks, tokens]`` matrix (walk lists of one fixed length, as the GPU
    writes them), and the token strings come from a table built once. A batch
    that doesn't fit that shape (walk lists of different lengths, null
    walks, negative tokens inside a walk) is read walk by walk instead, with
    the same output.
    """

    def __init__(
        self,
        walks_dir: str,
        id_to_word: np.ndarray | None,
        batch_size: int = 20_000,
        int_tokens: bool = False,
    ):
        _require_gensim()
        self.walks_dir = walks_dir
        self.id_to_word = id_to_word
        self.batch_size = batch_size
        self.int_tokens = int_tokens
        # Both the flat layout (`*.parquet`) and any nested layout
        # (`**/*.parquet`) are supported.
        files = sorted(
            glob.glob(f"{walks_dir}/*.parquet")
            + glob.glob(f"{walks_dir}/**/*.parquet", recursive=True)
        )
        self.files = sorted(set(files))
        if not self.files:
            raise FileNotFoundError(f"no parquet files under {walks_dir}")
        if id_to_word is None and not int_tokens:
            raise ValueError(
                "id_to_word is required when int_tokens=False; pass the "
                "word2idx mapping or set int_tokens=True"
            )

        self._int_strings = np.empty(0, dtype=object)

    def __iter__(self):
        for f in self.files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(
                columns=["vertices", "predicates"], batch_size=self.batch_size
            ):
                walks = self._walks_of_batch(batch)
                if walks is None:
                    walks = self._walks_one_by_one(batch)
                yield from walks

    def _walks_of_batch(self, batch):
        """The batch's walks via one token matrix, or None if the batch
        doesn't have that shape (see the class docstring)."""
        vertices = _list_matrix(batch.column("vertices"))
        predicates = _list_matrix(batch.column("predicates"))
        if vertices is None or predicates is None:
            return None
        n, n_vertices = vertices.shape
        if n == 0 or n_vertices < 2 or predicates.shape[1] < n_vertices - 1:
            return None
        width = 2 * n_vertices - 1
        tokens = np.empty((n, width), dtype=np.int64)
        tokens[:, 0::2] = vertices
        tokens[:, 1::2] = predicates[:, : n_vertices - 1]
        negative = vertices < 0
        ends = np.where(negative.any(axis=1), negative.argmax(axis=1), n_vertices)
        lengths = 2 * ends - 1
        tokens[np.arange(width) >= lengths[:, None]] = 0   # past the walk's end
        if (tokens < 0).any():
            return None
        words = self._strings_for(int(tokens.max()))[tokens].tolist()
        return (
            row if length == width else row[:length]
            for row, length, end in zip(words, lengths.tolist(), ends.tolist())
            if end >= 2
        )

    def _strings_for(self, largest_token: int) -> np.ndarray:
        """Token → word table covering ``largest_token``: `id_to_word`, or the
        tokens as strings, built once and extended when a larger token turns up."""
        if not self.int_tokens:
            return self.id_to_word
        if largest_token >= len(self._int_strings):
            size = max(largest_token + 1, 2 * len(self._int_strings))
            self._int_strings = np.array([str(t) for t in range(size)], dtype=object)
        return self._int_strings

    def _walks_one_by_one(self, batch):
        id_to_word = self.id_to_word
        int_tokens = self.int_tokens
        # Convert list<int> columns once per batch, not per walk.
        vs = batch.column("vertices").to_pylist()
        ps = batch.column("predicates").to_pylist()
        for walk_v, walk_p in zip(vs, ps):
            end = next(
                (i for i, v in enumerate(walk_v) if v < 0),
                len(walk_v),
            )
            if end < 2:
                continue
            walk_v = walk_v[:end]
            walk_p = walk_p[: end - 1]
            if int_tokens:
                v_words = [str(int(v)) for v in walk_v]
                p_words = [str(int(p)) for p in walk_p]
            else:
                v_words = id_to_word[walk_v].tolist()
                # Predicates may come back as float (cuGraph edge-weight
                # slot is float-typed); cast to int for fancy-indexing.
                p_words = id_to_word[[int(p) for p in walk_p]].tolist()
            out: list[Any] = [None] * (len(walk_v) + len(walk_p))
            out[0::2] = v_words
            out[1::2] = p_words
            yield out


def kv_to_token_aligned_matrix(kv, word2idx_df, int_tokens: bool | None = None) -> np.ndarray:
    """Reshape a `gensim.models.KeyedVectors` into a token-indexed matrix.

    Used by `gpu_rdf2vec._extract_embeddings_gensim` to materialize the
    `[token, word, embedding_*]` schema the PyTorch path produces. Built as
    a separate pure-numpy helper so its row-alignment invariants can be
    unit-tested without a GPU (the cudf concat that follows is
    GPU-dependent).

    Behaviour:

    * Each row `i` of the returned matrix is the embedding for token id `i`.
      Tokens are assumed contiguous `[0, n)` (guaranteed by `_generate_vocab`);
      `n_tokens` is derived from `max(token) + 1`.
    * If gensim trained with `int_tokens=True` (default in the library), the
      KV's `.index_to_key` entries are stringified ints and we use them
      directly. If `int_tokens=False`, keys are words (URIs, literals) and
      are looked up in `word2idx_df`. With `int_tokens=None` the keys are
      taken as tokens when every one of them is an integer.
    * Tokens that gensim's `min_count` filter dropped from the KV vocab get
      a zero row in the output. This is intentional but lossy at
      `min_count > 1`; callers may want to log a warning when
      `len(kv.index_to_key) < n_tokens`.
    """
    if hasattr(word2idx_df, "compute"):
        word2idx_df = word2idx_df.compute()
    if hasattr(word2idx_df, "to_pandas"):
        word2idx_df = word2idx_df.to_pandas()
    n_tokens = int(word2idx_df["token"].max()) + 1
    vectors = np.zeros((n_tokens, kv.vector_size), dtype=np.float32)
    keys = list(kv.index_to_key)
    if not keys:
        return vectors
    toks = None
    if int_tokens is not False:
        try:
            toks = np.asarray(keys).astype(np.int64)
        except (TypeError, ValueError):
            if int_tokens:
                raise
    if toks is None:
        lookup = pd.Series(word2idx_df["token"].to_numpy(), index=word2idx_df["word"].to_numpy())
        toks = lookup.reindex(keys).fillna(-1).to_numpy(np.int64)
    kv_vectors = getattr(kv, "vectors", None)
    if kv_vectors is None:
        kv_vectors = np.stack([kv[k] for k in keys])
    keep = (toks >= 0) & (toks < n_tokens)
    vectors[toks[keep]] = np.asarray(kv_vectors, dtype=np.float32)[keep]
    return vectors


def train_gensim_word2vec(
    walks_dir: str,
    word2idx_df,
    *,
    vector_size: int,
    window: int,
    min_count: int,
    embedding_model: str = "skipgram",
    negative: int = 5,
    epochs: int = 5,
    workers: int = 4,
    learning_rate: float = 0.025,
    min_learning_rate: float = 0.0001,
    sample: float = 0.001,
    random_state: int = 42,
    batch_size: int = 20_000,
    int_tokens: bool = True,
):
    """Fit a gensim `Word2Vec` model on a walks-parquet directory.

    Returns the fitted `gensim.models.Word2Vec` model. Caller can pull
    `.wv` (KeyedVectors) for downstream `transform()`.

    Parameters
    ----------
    walks_dir : str
        Path to a parquet directory written by
        `MultiGPUWalkCorpus.random_walk_sequences` (or the equivalent
        single-GPU output materialized to parquet).
    word2idx_df : cudf.DataFrame | dask_cudf.DataFrame | pandas.DataFrame
        The word2idx mapping (`word`, `token` columns). Required even when
        `int_tokens=True` so the resulting embeddings can be joined back to
        URIs in `transform()`.
    vector_size, window, min_count, negative, epochs, workers, learning_rate, min_learning_rate, sample, random_state : int or float
        Forwarded to `gensim.models.Word2Vec` (``learning_rate`` as
        ``alpha``, ``min_learning_rate`` as ``min_alpha``, ``random_state``
        as ``seed``).
    embedding_model : {"skipgram", "cbow"}, default "skipgram"
        Mapped to gensim's `sg=1` / `sg=0`.
    int_tokens : bool, default True
        Train on stringified int tokens instead of decoded URI strings. Saves
        a lot of RAM on large vocabularies at the cost of one post-training
        join with `word2idx_df` to recover URI-keyed embeddings (handled in
        `gpu_rdf2vec._fit_via_gensim`).
    """
    _require_gensim()
    sg = 1 if embedding_model == "skipgram" else 0
    id_to_word = None if int_tokens else load_id_to_word(word2idx_df)
    corpus = WalkCorpus(
        walks_dir,
        id_to_word,
        batch_size=batch_size,
        int_tokens=int_tokens,
    )
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        workers=workers,
        epochs=epochs,
        alpha=learning_rate,
        min_alpha=min_learning_rate,
        sample=sample,
        seed=random_state,
    )
    return model
