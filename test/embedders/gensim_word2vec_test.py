"""Tests for the optional gensim Word2Vec backend.

These run on CPU only — gensim is a CPU library, and the streaming
`WalkCorpus` iterator reads parquet files directly without going through
cuDF. Tests are skipped at import time if the `gensim` extra isn't
installed, mirroring the user-facing import gate in
`embedders.gensim_word2vec`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip the whole module if gensim isn't installed (matches the optional
# dependency gate). pyarrow is technically optional alongside gensim, but
# in practice it ships with the cudf install in our environments.
gensim = pytest.importorskip("gensim")
pq = pytest.importorskip("pyarrow.parquet")
import pyarrow as pa  # noqa: E402

from src.rdf2vecgpu.embedders.gensim_word2vec import (  # noqa: E402
    WalkCorpus,
    kv_to_token_aligned_matrix,
    load_id_to_word,
    train_gensim_word2vec,
)


class _FakeKeyedVectors:
    """Minimal stand-in for `gensim.models.KeyedVectors` for unit-testing the
    row-alignment helper without spinning up a real Word2Vec training run.

    Implements just the surface `kv_to_token_aligned_matrix` consumes:
    `vector_size`, `index_to_key`, and `__getitem__`.
    """

    def __init__(self, key_to_vector: dict[str, np.ndarray]):
        if not key_to_vector:
            raise ValueError("at least one key required")
        first = next(iter(key_to_vector.values()))
        self.vector_size = int(first.shape[0])
        self._kv = {k: v.astype(np.float32) for k, v in key_to_vector.items()}
        self.index_to_key = list(key_to_vector.keys())

    def __getitem__(self, key):
        return self._kv[key]


@pytest.fixture
def synthetic_walks(tmp_path: Path) -> Path:
    """Write a tiny parquet directory shaped like cuGraph's walk output.

    Schema matches `MultiGPUWalkCorpus.random_walk_sequences` /
    `tools/walk_gen.py`:
        vertices   : list<int32>   length walk_depth + 1
        predicates : list<int32>   length walk_depth

    Includes one walk that hits cuGraph's `-1` sentinel partway through to
    exercise the trim-at-first-sentinel logic.
    """
    walks = pa.table(
        {
            "vertices": [
                [0, 1, 2, 3],     # full-length walk
                [4, 5, 6, 7],     # full-length walk
                [0, 2, -1, -1],   # walk truncates at index 2
                [1, 3, 5, 7],     # full-length walk
            ],
            "predicates": [
                [10, 11, 12],
                [13, 14, 15],
                [10, -1, -1],
                [11, 13, 15],
            ],
        }
    )
    path = tmp_path / "walks"
    path.mkdir()
    pq.write_table(walks, path / "part.0.parquet")
    return path


@pytest.fixture
def synthetic_word2idx() -> pd.DataFrame:
    """word2idx mapping covering the synthetic walk vocabulary.

    Tokens 0..7 for vertices, 10..15 for predicates. Contiguous tokens are
    not required for this fixture (gaps are fine for `int_tokens=True`
    training), but `load_id_to_word` requires contiguous so we test that
    separately below.
    """
    rows = (
        [(t, f"v{t}") for t in range(8)]
        + [(t, f"p{t}") for t in range(10, 16)]
    )
    return pd.DataFrame(rows, columns=["token", "word"])


def test_walk_corpus_streams_interleaved_int_tokens(synthetic_walks):
    """`int_tokens=True` emits stringified ints; `-1` sentinel trims the walk."""
    corpus = WalkCorpus(
        str(synthetic_walks), id_to_word=None, batch_size=2, int_tokens=True
    )
    walks = list(iter(corpus))
    # 4 input walks; the 3rd has end<2 after trim? No — vertices=[0,2,-1,-1]
    # trims at index 2 → walk_v=[0,2], walk_p=[10] → length 3 (interleaved).
    assert len(walks) == 4
    # First walk: [0, 1, 2, 3] vertices + [10, 11, 12] predicates → interleaved
    # [v0, p10, v1, p11, v2, p12, v3] = ["0", "10", "1", "11", "2", "12", "3"]
    assert walks[0] == ["0", "10", "1", "11", "2", "12", "3"]
    # Truncated walk: [0, 2] vertices + [10] predicate → ["0", "10", "2"]
    assert walks[2] == ["0", "10", "2"]


def test_walk_corpus_streams_decoded_uris(synthetic_walks, synthetic_word2idx):
    """`int_tokens=False` decodes via the id_to_word numpy array."""
    # Build a dense object array indexed by token. Tokens go up to 15 here,
    # so size 16. Unset slots stay None (not used by these walks).
    id_to_word = np.empty(16, dtype=object)
    for _, row in synthetic_word2idx.iterrows():
        id_to_word[int(row["token"])] = str(row["word"])

    corpus = WalkCorpus(
        str(synthetic_walks), id_to_word=id_to_word, batch_size=10
    )
    walks = list(iter(corpus))
    # First walk decoded: vertices [0,1,2,3] → v0,v1,v2,v3; predicates
    # [10,11,12] → p10,p11,p12; interleaved.
    assert walks[0] == ["v0", "p10", "v1", "p11", "v2", "p12", "v3"]


def test_walk_corpus_drops_walks_below_min_length(tmp_path: Path):
    """A walk with `-1` at vertex index 1 has fewer than 2 valid vertices and is dropped."""
    walks = pa.table(
        {
            "vertices": [[0, -1, -1, -1], [0, 1, 2, 3]],
            "predicates": [[-1, -1, -1], [10, 11, 12]],
        }
    )
    walks_dir = tmp_path / "walks"
    walks_dir.mkdir()
    pq.write_table(walks, walks_dir / "part.0.parquet")
    corpus = WalkCorpus(str(walks_dir), id_to_word=None, int_tokens=True)
    out = list(iter(corpus))
    # Only the second walk survives.
    assert len(out) == 1
    assert out[0] == ["0", "10", "1", "11", "2", "12", "3"]


def test_load_id_to_word_requires_contiguous_tokens():
    """Non-contiguous tokens raise — `_generate_vocab` always produces [0, n)."""
    bad = pd.DataFrame({"token": [0, 1, 5], "word": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="not contiguous"):
        load_id_to_word(bad)


def test_load_id_to_word_returns_index_aligned_array():
    """token-indexed lookup: `arr[t]` returns the word for token t."""
    df = pd.DataFrame(
        {"token": [2, 0, 1], "word": ["c", "a", "b"]}  # arbitrary order
    )
    arr = load_id_to_word(df)
    assert len(arr) == 3
    assert arr[0] == "a" and arr[1] == "b" and arr[2] == "c"


def test_train_gensim_word2vec_end_to_end(synthetic_walks, synthetic_word2idx):
    """Smoke test: gensim trains and KV holds a vector per seen token.

    Note: `synthetic_word2idx` has non-contiguous tokens (gap at 8, 9). This
    works for the test because `int_tokens=True` skips `load_id_to_word`'s
    contiguous-token guard — gensim trains directly on stringified int
    tokens from the walks file, not on URI lookups.
    """
    model = train_gensim_word2vec(
        walks_dir=str(synthetic_walks),
        word2idx_df=synthetic_word2idx,
        vector_size=8,
        window=2,
        min_count=1,
        embedding_model="skipgram",
        negative=2,
        epochs=2,
        workers=1,
        learning_rate=0.025,
        random_state=42,
        int_tokens=True,
    )
    kv = model.wv
    # gensim only sees tokens that actually appear in the synthetic walks
    # (after sentinel trimming): vertices {0,1,2,3,4,5,6,7}, predicates
    # {10,11,12,13,14,15}. Total 14 unique stringified-int keys.
    assert kv.vector_size == 8
    assert "0" in kv  # vertex token
    assert "10" in kv  # predicate token
    # Sanity: pulling a vector returns a float32 array of vector_size.
    v = kv["0"]
    assert v.shape == (8,) and v.dtype == np.float32


def test_kv_to_token_aligned_matrix_int_tokens_alignment():
    """Stringified-int KV keys map directly to token rows (the default path)."""
    word2idx = pd.DataFrame(
        {"token": [0, 1, 2, 3], "word": ["a", "b", "c", "d"]}
    )
    # KV trained with int_tokens=True: keys are str(int_token).
    fake_kv = _FakeKeyedVectors(
        {
            "0": np.array([1.0, 0.0, 0.0]),
            "1": np.array([0.0, 1.0, 0.0]),
            "2": np.array([0.0, 0.0, 1.0]),
            "3": np.array([1.0, 1.0, 1.0]),
        }
    )
    matrix = kv_to_token_aligned_matrix(fake_kv, word2idx)
    assert matrix.shape == (4, 3)
    assert np.array_equal(matrix[0], [1.0, 0.0, 0.0])
    assert np.array_equal(matrix[1], [0.0, 1.0, 0.0])
    assert np.array_equal(matrix[2], [0.0, 0.0, 1.0])
    assert np.array_equal(matrix[3], [1.0, 1.0, 1.0])


def test_kv_to_token_aligned_matrix_min_count_filtered_rows_are_zero():
    """Tokens dropped by gensim's min_count filter get zero rows.

    This is the documented (lossy) behaviour at min_count > 1: the user's
    word2idx contains 4 tokens, but gensim only kept 2; the missing two
    must be zero-filled in the output so the row-index-to-token contract
    holds.
    """
    word2idx = pd.DataFrame(
        {"token": [0, 1, 2, 3], "word": ["a", "b", "c", "d"]}
    )
    fake_kv = _FakeKeyedVectors(
        {
            # Tokens 1 and 2 are absent — simulates min_count filtering.
            "0": np.array([1.0, 2.0]),
            "3": np.array([3.0, 4.0]),
        }
    )
    matrix = kv_to_token_aligned_matrix(fake_kv, word2idx)
    assert matrix.shape == (4, 2)
    assert np.array_equal(matrix[0], [1.0, 2.0])
    assert np.array_equal(matrix[1], [0.0, 0.0])  # filtered → zero
    assert np.array_equal(matrix[2], [0.0, 0.0])  # filtered → zero
    assert np.array_equal(matrix[3], [3.0, 4.0])


def test_kv_to_token_aligned_matrix_uri_keys_fallback_lookup():
    """Non-int KV keys fall back to looking up the token via word2idx['word']."""
    word2idx = pd.DataFrame(
        {"token": [0, 1, 2], "word": ["http://a", "http://b", "http://c"]}
    )
    # KV trained with int_tokens=False: keys are URIs.
    fake_kv = _FakeKeyedVectors(
        {
            "http://a": np.array([1.0]),
            "http://c": np.array([3.0]),
        }
    )
    matrix = kv_to_token_aligned_matrix(fake_kv, word2idx)
    assert matrix.shape == (3, 1)
    assert matrix[0, 0] == 1.0  # token 0 → http://a → fake_kv["http://a"]
    assert matrix[1, 0] == 0.0  # not in KV
    assert matrix[2, 0] == 3.0  # token 2 → http://c


def test_kv_to_token_aligned_matrix_dtype_is_float32():
    """Output matrix is float32, regardless of fake KV's input dtype."""
    word2idx = pd.DataFrame({"token": [0], "word": ["a"]})
    fake_kv = _FakeKeyedVectors({"0": np.array([1.0], dtype=np.float64)})
    matrix = kv_to_token_aligned_matrix(fake_kv, word2idx)
    assert matrix.dtype == np.float32
