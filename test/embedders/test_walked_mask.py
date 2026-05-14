"""Unit tests for the walked_mask coverage-fix feature (spec §7.5, plan Task 4-6).

The bitmask is derived from `_token_counts_from_walk_parquet`. These tests
exercise the derivation and downstream zero-row application in transform.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import pytest


def _write_synth_walks(tmpdir, walks):
    """walks: list of (vertices: List[int], predicates: List[int]) tuples.
    Writes a single parquet file under `<tmpdir>/walks/part.0.parquet`."""
    os.makedirs(f"{tmpdir}/walks", exist_ok=True)
    tbl = pa.table({
        "vertices": pa.array([v for v, _ in walks], type=pa.list_(pa.int32())),
        "predicates": pa.array([p for _, p in walks], type=pa.list_(pa.int32())),
    })
    pq.write_table(tbl, f"{tmpdir}/walks/part.0.parquet")


def test_walked_mask_from_synthetic_walks(tmp_path):
    """walked_mask[i] iff token i appears as a vertex or predicate in any walk."""
    from src.rdf2vecgpu.gpu_rdf2vec import _token_counts_from_walk_parquet

    walks = [
        ([0, 5, 2], [7, 9]),       # tokens touched: 0, 2, 5, 7, 9
        ([5, 2], [9]),             # tokens touched: 2, 5, 9
        ([3], []),                 # token touched: 3
    ]
    _write_synth_walks(str(tmp_path), walks)
    counts = _token_counts_from_walk_parquet(f"{tmp_path}/walks", vocab_size=10)
    mask = counts > 0
    expected_touched = {0, 2, 3, 5, 7, 9}
    expected_untouched = {1, 4, 6, 8}
    assert set(np.flatnonzero(mask).tolist()) == expected_touched
    assert set(np.flatnonzero(~mask).tolist()) == expected_untouched


def test_walked_mask_attribute_is_initialised_to_none():
    """`_walked_mask` must be initialised to None in GpuRdf2Vec.__init__ so legacy/
    unfitted instances skip the masking in transform (spec §7.5 backward-compat).

    Source-level check rather than constructing the class (which requires a full
    config + GPU). The string literal must appear in the module file."""
    import importlib
    rdf_mod = importlib.import_module("src.rdf2vecgpu.gpu_rdf2vec")
    src = open(rdf_mod.__file__).read()
    assert "self._walked_mask = None" in src, (
        "_walked_mask must be initialised to None in GpuRdf2Vec.__init__ "
        "(spec §7.5 backward-compat — legacy/unfitted instances skip masking)"
    )


class _StubWord2VecModel:
    """Mock for self.word2vec_model.get_embeddings() — returns a fixed tensor."""
    def __init__(self, embeddings):
        self._E = embeddings
    def get_embeddings(self):
        return self._E


def _make_rdf2vec_stub(vocab_size, dim, walked_mask=None):
    """Build a minimally populated GpuRdf2Vec instance bypassing __init__'s
    GPU-dependent initialisation. Only what transform's pytorch branch needs."""
    import cudf
    import torch
    from src.rdf2vecgpu.gpu_rdf2vec import GpuRdf2Vec

    inst = GpuRdf2Vec.__new__(GpuRdf2Vec)
    # Minimal config carrier — _extract_embeddings_pytorch only reads
    # self.config.backend (already branched) and self.config.generate_artifact.
    inst.config = type("C", (), {"backend": "pytorch", "generate_artifact": False})()
    inst.word2idx = cudf.DataFrame({
        "token": list(range(vocab_size)),
        "word":  [f"w{i}" for i in range(vocab_size)],
    })
    embeddings = torch.ones(vocab_size, dim, dtype=torch.float32) * 0.5  # all 0.5
    inst.word2vec_model = _StubWord2VecModel(embeddings)
    inst._walked_mask = walked_mask
    inst.tracker = type("T", (), {"stage": lambda self, name: __import__("contextlib").nullcontext()})()
    return inst


@pytest.mark.gpu
def test_extract_embeddings_pytorch_zeros_unwalked_rows():
    """transform → pytorch branch zeros rows where walked_mask is False."""
    import numpy as np
    vocab_size, dim = 6, 4
    mask = np.array([True, False, True, False, True, False])
    inst = _make_rdf2vec_stub(vocab_size, dim, walked_mask=mask)
    df = inst._extract_embeddings_pytorch().to_pandas()

    embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
    arr = df[embedding_cols].to_numpy()
    # Walked rows: all 0.5; un-walked rows: all 0.0
    assert np.allclose(arr[mask], 0.5)
    assert np.allclose(arr[~mask], 0.0)


@pytest.mark.gpu
def test_extract_embeddings_pytorch_legacy_no_mask_unchanged():
    """`_walked_mask is None` → no zeroing (legacy / unfitted behaviour)."""
    import numpy as np
    vocab_size, dim = 6, 4
    inst = _make_rdf2vec_stub(vocab_size, dim, walked_mask=None)
    df = inst._extract_embeddings_pytorch().to_pandas()

    embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
    arr = df[embedding_cols].to_numpy()
    assert np.allclose(arr, 0.5), "no mask → no zeroing"


def test_gensim_assertion_catches_mismatch(monkeypatch):
    """The gensim transform sanity check fires when KV emits a token not in walked_mask."""
    import numpy as np
    import pandas as pd
    import importlib
    rdf_mod = importlib.import_module("src.rdf2vecgpu.gpu_rdf2vec")
    GPU_RDF2Vec = rdf_mod.GPU_RDF2Vec

    inst = GPU_RDF2Vec.__new__(GPU_RDF2Vec)
    inst.config = type("C", (), {"backend": "gensim", "generate_artifact": False, "vector_size": 4})()
    # Use a plain pandas DataFrame so no GPU is needed for the assertion path.
    inst.word2idx = pd.DataFrame({
        "token": [0, 1, 2, 3, 4],
        "word":  [f"w{i}" for i in range(5)],
    })
    # Mock gensim KV with index_to_key = ["w0", "w2", "w4"] — token ids 0, 2, 4.
    class _StubKV:
        vector_size = 4
        index_to_key = ["w0", "w2", "w4"]
        def get_vector(self, key):
            return np.full(4, 0.5, dtype=np.float32)
        def __getitem__(self, key):
            return np.full(4, 0.5, dtype=np.float32)
    inst.word2vec_model = type("M", (), {"wv": _StubKV()})()
    # walked_mask says only tokens {0, 4} are walked → gensim KV emits token 2 too
    # → mismatch → expect AssertionError containing "walked_mask".
    inst._walked_mask = np.array([True, False, False, False, True])
    inst.tracker = type("T", (), {"stage": lambda self, name: __import__("contextlib").nullcontext()})()
    with pytest.raises(AssertionError, match=r"walked_mask"):
        inst._extract_embeddings_gensim()
