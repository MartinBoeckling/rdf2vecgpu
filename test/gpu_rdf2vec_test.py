"""Tests for GPU_RDF2Vec helpers: the DDP launch, the multi-GPU walk-file
check, and saving, loading and exporting a gensim model."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.rdf2vecgpu.gpu_rdf2vec import WalkParquet, _check_walk_files, _ddp_strategy


def test_ddp_strategy_spawns_on_one_node(monkeypatch):
    from lightning.pytorch.plugins.environments import LightningEnvironment

    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    assert _ddp_strategy(1, 1) == "auto"
    assert isinstance(_ddp_strategy(4, 1).cluster_environment, LightningEnvironment)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert isinstance(_ddp_strategy(4, 1).cluster_environment, LightningEnvironment)


def test_ddp_strategy_leaves_slurm_tasks_to_slurm(monkeypatch):
    """With one SLURM task per GPU, a second launch would start N x N workers."""
    monkeypatch.setenv("SLURM_NTASKS", "4")
    assert _ddp_strategy(4, 1) == "ddp"
    monkeypatch.delenv("SLURM_NTASKS")
    assert _ddp_strategy(4, 2) == "ddp"


def _write_walks(path: Path, n: int, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    table = pa.table({"vertices": [[0, 1]] * n, "predicates": [[2]] * n})
    pq.write_table(table, path / name)


def test_check_walk_files(tmp_path):
    _write_walks(tmp_path, 3, "part.0.parquet")
    _write_walks(tmp_path / "sub", 2, "part.1.parquet")
    _check_walk_files(WalkParquet(str(tmp_path), 5))
    with pytest.raises(RuntimeError, match="holds 5 of the 7 walks"):
        _check_walk_files(WalkParquet(str(tmp_path), 7))


@pytest.mark.gpu
def test_gensim_save_load_and_artifacts(tmp_path, monkeypatch):
    gensim = pytest.importorskip("gensim")
    import cudf

    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    sentences = [["0", "3", "1"], ["1", "3", "2"], ["2", "3", "0"]] * 20
    w2v = gensim.models.Word2Vec(sentences, vector_size=4, window=2, min_count=1, seed=1, workers=1)
    model = GPU_RDF2Vec(backend="gensim", vector_size=4, window_size=2, generate_artifact=True)
    model.word2vec_model = w2v
    model.word2idx = cudf.DataFrame({"token": [0, 1, 2, 3], "word": ["a", "b", "c", "p"]})

    model.save(str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "model.gensim").exists()
    loaded = GPU_RDF2Vec.load(str(tmp_path / "saved"))
    assert np.array_equal(loaded.word2vec_model.wv["3"], w2v.wv["3"])

    monkeypatch.chdir(tmp_path)
    emb = loaded.transform()
    assert os.path.exists("vector/embeddings_skipgram.parquet")
    assert cudf.read_parquet("vector/word2idx.parquet").shape == (4, 2)
    row = emb[emb["token"] == 3]
    got = np.array([float(row[f"embedding_{i}"].iloc[0]) for i in range(4)])
    assert np.allclose(got, w2v.wv["3"])


@pytest.mark.gpu
def test_fit_before_load_data_says_so(tmp_path):
    """`read_data` only reads the file; `fit` needs the graph `load_data` builds."""
    import cudf

    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    path = tmp_path / "kg.parquet"
    cudf.DataFrame({"subject": ["a", "b"], "predicate": ["p", "p"], "object": ["b", "a"]}).to_parquet(path)
    model = GPU_RDF2Vec(epochs=1, vector_size=4, window_size=2, tune_batch_size=False)
    edges = model.read_data(str(path))
    with pytest.raises(RuntimeError, match="call load_data"):
        model.fit(edges)
