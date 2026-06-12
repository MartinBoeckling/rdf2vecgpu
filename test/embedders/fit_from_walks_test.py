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


def test_fit_from_walks_missing_corpus_raises(tmp_path):
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec
    m = GPU_RDF2Vec(config=RDF2VecConfig(backend="gensim", embedding_model="skipgram"))
    with pytest.raises(FileNotFoundError):
        m.fit_from_walks(str(tmp_path / "missing_walks"), str(tmp_path / "missing.parquet"))


def test_fit_from_walks_dispatches_per_backend(tmp_path, monkeypatch):
    import cudf
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    walks = tmp_path / "walks"; walks.mkdir()
    (walks / "part.0.parquet").write_bytes(b"x")          # existence only
    vocab = str(tmp_path / "vocab.parquet")
    cudf.DataFrame({"token_id": [0, 1], "entity": ["a", "b"]}).to_parquet(vocab)

    called = {}
    m = GPU_RDF2Vec(config=RDF2VecConfig(backend="gensim", embedding_model="skipgram"))
    monkeypatch.setattr(m, "_load_word2idx", lambda vp: called.setdefault("vocab", vp))
    # train_gensim_word2vec is imported lazily from its source module — patch it THERE.
    monkeypatch.setattr("rdf2vecgpu.embedders.gensim_word2vec.train_gensim_word2vec",
                        lambda **kw: called.setdefault("gensim", kw) or "model")
    m.fit_from_walks(str(walks), vocab)
    assert called["gensim"]["walks_dir"] == str(walks)


@pytest.mark.gpu
def test_fit_from_walks_gensim_and_pytorch_single_from_shared_corpus(tmp_path):
    import cudf, numpy as np
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    triples = str(tmp_path / "t.parquet")
    cudf.DataFrame({
        "subject": [f"e{i%50}" for i in range(3000)],
        "predicate": [f"p{i%5}" for i in range(3000)],
        "object": [f"e{(i*3)%50}" for i in range(3000)],
    }).to_parquet(triples)

    common = dict(embedding_model="skipgram", walk_strategy="random",
                  walk_depth=3, walk_number=10, vector_size=16, epochs=1, multi_gpu=False)

    prep = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", **common))
    walks_path, vocab_path = prep.prepare_corpus(prep.load_data(triples), str(tmp_path / "corpus"))

    for backend in ("gensim", "pytorch"):
        m = GPU_RDF2Vec(config=RDF2VecConfig(backend=backend, **common))
        m.fit_from_walks(walks_path, vocab_path)
        emb = m.transform().to_pandas()
        vec = [c for c in emb.columns if c not in ("token", "word")]
        assert len(emb) > 0 and np.isfinite(emb[vec].to_numpy()).all()


@pytest.mark.gpu
def test_fit_from_walks_pytorch_mg_subprocess(tmp_path):
    import os, sys, subprocess, torch, pandas as pd, numpy as np
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    driver = os.path.join(os.path.dirname(__file__), "_drivers", "ffw_pytorch_mg_driver.py")
    out = str(tmp_path / "emb.parquet")
    env = {**os.environ, "NCCL_P2P_DISABLE": "1", "CUDA_VISIBLE_DEVICES": "0,1"}
    r = subprocess.run([sys.executable, driver, out], env=env, capture_output=True, text=True, timeout=600)
    assert "DRIVER_OK" in r.stdout, f"driver failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr[-3000:]}"
    emb = pd.read_parquet(out)
    vec = [c for c in emb.columns if c not in ("token", "word")]
    assert np.isfinite(emb[vec].to_numpy()).all()
