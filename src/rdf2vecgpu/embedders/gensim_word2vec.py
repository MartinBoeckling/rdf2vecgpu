"""Optional gensim Word2Vec backend for the RDF2Vec training stage.

Mirrors `tools/train_gensim.py` from the upstream Wikidata calibration runs
but lives inside the library so `GPU_RDF2Vec(config=RDF2VecConfig(
backend="gensim")).fit(...)` works end-to-end.

Why this exists alongside the default PyTorch trainer:

* gensim's C-level negative-sampling Word2Vec is typically 5-10× faster on CPU
  than the equivalent PyTorch skip-gram loop on the same corpus.
* `pyarrow.parquet.ParquetFile.iter_batches` keeps memory constant while
  streaming multi-billion-walk corpora; the PyTorch path materializes the
  whole corpus to GPU.
* For users running CPU-rich + GPU-light boxes (e.g. a single A6000 + a
  large NUMA box for training), the gensim path frees the GPU after walk
  generation and lets the CPU do the training work.

This module is gated behind the `gensim` optional-dependency group:
    pip install rdf2vecgpu[gensim]
"""
from __future__ import annotations

import glob
from typing import TYPE_CHECKING, Any

import numpy as np

# gensim and pyarrow are optional dependencies. Import lazily so users on the
# default pytorch backend don't need them installed; raise a friendly error
# only when the gensim path is actually invoked.
try:
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


class WalkCorpus:
    """Stream walks-parquet → interleaved URI sequences for gensim.

    Each parquet file under `walks_dir` has columns
        vertices   : list<int>   length = walk_depth + 1
        predicates : list<int>   length = walk_depth
    written by `MultiGPUWalkCorpus.random_walk_sequences` (or by the standalone
    `tools/walk_gen.py`). This iterator emits interleaved sequences
    `[v0, p0, v1, p1, v2, ..., v_n]` per walk — the format gensim's
    `Word2Vec(sentences=...)` constructor consumes.

    cuGraph emits `-1` for vertex slots past the walk's actual end (dead-end
    hit before max_depth). We trim each walk at the first `-1` and keep the
    matching predicate prefix; walks with fewer than 2 valid vertices are
    skipped (need at least one edge for skip-gram).
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

    def __iter__(self):
        id_to_word = self.id_to_word
        int_tokens = self.int_tokens
        for f in self.files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(
                columns=["vertices", "predicates"], batch_size=self.batch_size
            ):
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


def kv_to_token_aligned_matrix(kv, word2idx_df) -> np.ndarray:
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
      directly. If `int_tokens=False`, keys are URI strings and we fall back
      to looking each one up in `word2idx_df` — slower but correct.
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
    for key in kv.index_to_key:
        try:
            tok = int(key)
        except (TypeError, ValueError):
            row = word2idx_df[word2idx_df["word"] == key]
            if row.empty:
                continue
            tok = int(row["token"].iloc[0])
        if 0 <= tok < n_tokens:
            vectors[tok] = kv[key]
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
    vector_size, window, min_count, negative, epochs, workers, learning_rate,
    random_state : int / float
        Forwarded to `gensim.models.Word2Vec`.
    embedding_model : {"skipgram", "cbow"}, default "skipgram"
        Mapped to gensim's `sg=1` / `sg=0`.
    int_tokens : bool, default True
        Train on stringified int tokens instead of decoded URI strings. Saves
        ~20 GB RAM on Wikidata-scale corpora at the cost of one post-training
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
        seed=random_state,
    )
    return model
