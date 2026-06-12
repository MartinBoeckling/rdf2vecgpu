import pytest


@pytest.mark.gpu
def test_token_counts_from_cudf_walks_skips_sentinels():
    import cudf
    from rdf2vecgpu.gpu_rdf2vec import _token_counts_from_cudf_walks

    df = cudf.DataFrame({
        "vertices":   [[0, 1, 2, -1], [1, 1, -1, -1]],
        "predicates": [[3, 3, -1], [3, -1, -1]],
    })
    counts = _token_counts_from_cudf_walks(df, vocab_size=4)
    # token 0 →1, token 1 →3 (two in vertices + one), token 2 →1, token 3 →3; -1 skipped.
    assert counts.tolist() == [1, 3, 1, 3]
    assert counts.dtype.name == "int64"


@pytest.mark.gpu
def test_load_word2idx_renames_corpus_vocab(tmp_path):
    import cudf
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    vocab_path = str(tmp_path / "vocab.parquet")
    cudf.DataFrame({"token_id": [0, 1, 2], "entity": ["a", "b", "c"]}).to_parquet(vocab_path)

    m = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", embedding_model="skipgram"))
    out = m._load_word2idx(vocab_path)

    assert list(out.columns) == ["token", "word"]
    assert m.word2idx is out
    assert out["token"].to_arrow().to_pylist() == [0, 1, 2]
    assert out["word"].to_arrow().to_pylist() == ["a", "b", "c"]


@pytest.mark.gpu
def test_single_gpu_fit_uses_in_memory_feed_no_temp_parquet(tmp_path, monkeypatch):
    """Single-GPU skip-gram fit() must NOT round-trip walks through temp parquet;
    it uses the GPU-resident InMemoryWalkSeqDataModule."""
    import tempfile
    import cudf
    import numpy as np
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    # Fail loudly if the single-GPU path creates a temp dir (the parquet round-trip).
    def _boom(*a, **k):
        raise AssertionError("temp parquet used on the single-GPU path")
    monkeypatch.setattr(tempfile, "mkdtemp", _boom)

    triples = str(tmp_path / "t.parquet")
    cudf.DataFrame({
        "subject": [f"e{i%40}" for i in range(2000)],
        "predicate": [f"p{i%5}" for i in range(2000)],
        "object": [f"e{(i*7)%40}" for i in range(2000)],
    }).to_parquet(triples)

    cfg = RDF2VecConfig(backend="pytorch", embedding_model="skipgram",
                        walk_strategy="random", walk_depth=3, walk_number=5,
                        vector_size=8, epochs=1, multi_gpu=False)
    m = GPU_RDF2Vec(config=cfg)
    edges = m.load_data(triples)
    m.fit(edges)                       # must not raise (no mkdtemp on single-GPU)
    emb = m.transform().to_pandas()
    vec = [c for c in emb.columns if c not in ("token", "word")]
    assert np.isfinite(emb[vec].to_numpy()).all()
