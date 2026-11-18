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


def test_fit_transform_walk_vertices_defaults_to_none():
    """The docs call `fit_transform(edges)` without start vertices."""
    import inspect

    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    assert inspect.signature(GPU_RDF2Vec.fit_transform).parameters["walk_vertices"].default is None


def test_pairs_in_sequence_matches_the_window_template():
    from src.rdf2vecgpu.embedders.word2vec_loader import skipgram_pairs_per_walk
    from src.rdf2vecgpu.gpu_rdf2vec import _pairs_in_sequence

    for depth in (1, 2, 4, 8):
        for window in (1, 2, 5, 20):
            assert _pairs_in_sequence(2 * depth + 1, window) == skipgram_pairs_per_walk(depth, window)
    assert _pairs_in_sequence(1, 5) == 0


def test_walk_statistics_count_tokens_and_pairs_of_walks_as_they_end():
    import torch

    from src.rdf2vecgpu.gpu_rdf2vec import _pairs_in_sequence, _walk_statistics

    # 1 -p5-> 2 -p6-> 3; 4 -p7-> 0 then ended (padding predicate 9 is not counted); 8 alone
    vertices = torch.tensor([[1, 2, 3], [4, 0, -1], [8, -1, -1]])
    predicates = torch.tensor([[5, 6], [7, 9], [9, 9]])
    counts, n_pairs = _walk_statistics([(vertices, predicates)], vocab_size=10, window=2)
    assert counts.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert n_pairs == _pairs_in_sequence(5, 2) + _pairs_in_sequence(3, 2)


def test_walk_statistics_count_large_int32_corpora_in_row_slices():
    """The whole single-GPU corpus is one `(vertices, predicates)` pair of int32
    matrices; it is counted a slice of rows at a time (an int64 copy of all
    of it ran out of GPU memory on Wikidata5M), with the same result."""
    import torch

    from src.rdf2vecgpu.gpu_rdf2vec import _walk_statistics

    g = torch.Generator().manual_seed(0)
    vertices = torch.randint(0, 50, (1000, 5), generator=g, dtype=torch.int32)
    predicates = torch.randint(50, 60, (1000, 4), generator=g, dtype=torch.int32)
    vertices[::3, 2:] = -1
    whole = _walk_statistics([(vertices.long(), predicates.long())], vocab_size=60, window=3)
    sliced = _walk_statistics([(vertices, predicates)], vocab_size=60, window=3, rows_per_slice=7)
    assert sliced[0].tolist() == whole[0].tolist() and sliced[1] == whole[1]


@pytest.mark.gpu
def test_batch_size_is_capped_to_one_hundredth_of_the_pairs():
    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    default = GPU_RDF2Vec()
    assert default._effective_batch_size(100_000_000) == 32_768
    assert default._effective_batch_size(1_000_000) == 10_000
    assert default._effective_batch_size(150) == 2
    assert default._effective_batch_size(0) == 1
    assert GPU_RDF2Vec(batch_size=64)._effective_batch_size(1_000_000) == 64


@pytest.mark.gpu
def test_single_gpu_gensim_fit(tmp_path):
    """The whole single-GPU gensim path: a cuDF walk frame is not a Dask one."""
    pytest.importorskip("gensim")
    import cudf

    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    path = tmp_path / "kg.parquet"
    cudf.DataFrame(
        {
            "subject": ["a", "b", "c", "a", "d"],
            "predicate": ["p", "p", "q", "q", "p"],
            "object": ["b", "c", "a", "c", "a"],
        }
    ).to_parquet(path, index=False)
    model = GPU_RDF2Vec(backend="gensim", walk_depth=2, walk_number=5, epochs=1, vector_size=4, window_size=2)
    edges = model.load_data(str(path))
    emb = model.fit_transform(edges)
    assert emb.shape == (model.word2idx.shape[0], 4 + 2)
    assert model.word2vec_model.alpha == 0.025


@pytest.mark.gpu
def test_pytorch_skipgram_zeroes_unwalked_tokens_and_keeps_them_zero_after_load(tmp_path):
    """Walk only from `a`: the tokens of the other component never occur in a
    walk, are never trained, and come back as zeros, also after save/load."""
    import cudf

    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    path = tmp_path / "kg.parquet"
    cudf.DataFrame(
        {"subject": ["a", "b", "c"], "predicate": ["p", "p", "q"], "object": ["b", "a", "d"]}
    ).to_parquet(path)
    model = GPU_RDF2Vec(epochs=2, vector_size=4, window_size=2, walk_number=5, batch_size=64)
    edges = model.load_data(str(path))
    words = model.word2idx.to_pandas().set_index("word")["token"]
    # the token ids of this tiny vocabulary are int8; fit must cast them for cuGraph
    model.fit(edges, walk_vertices=cudf.Series([int(words["a"])], dtype=words.dtype))

    def vectors(emb):
        frame = emb.to_pandas().set_index("word")
        return frame[[f"embedding_{i}" for i in range(4)]]

    emb = vectors(model.transform())
    for word in ("c", "q", "d"):
        assert (emb.loc[word] == 0).all(), word
    for word in ("a", "b", "p"):
        assert (emb.loc[word] != 0).any(), word

    model.save(str(tmp_path / "saved"))
    assert (tmp_path / "saved" / "walked_tokens.npy").exists()
    loaded = vectors(GPU_RDF2Vec.load(str(tmp_path / "saved")).transform())
    assert np.allclose(loaded.loc[emb.index].to_numpy(), emb.to_numpy())


def _small_kg(tmp_path):
    import cudf

    path = tmp_path / "kg.parquet"
    subjects = [f"e{i}" for i in range(20)]
    cudf.DataFrame(
        {
            "subject": subjects,
            "predicate": ["p" if i % 2 else "q" for i in range(20)],
            "object": [f"e{(i + 1) % 20}" for i in range(20)],
        }
    ).to_parquet(path)
    return str(path)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend, model",
    [("pytorch", "skipgram"), ("pytorch", "cbow"), ("pytorch", "order_aware_skipgram"), ("gensim", "skipgram")],
)
def test_prepare_corpus_then_fit_from_walks(tmp_path, backend, model):
    if backend == "gensim":
        pytest.importorskip("gensim")
    import cudf

    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    settings = dict(
        backend=backend, embedding_model=model, epochs=1, vector_size=4, window_size=2, walk_number=3, walk_depth=3,
        grad_clip_value=0.3,
    )
    preparer = GPU_RDF2Vec(**settings)
    edges = preparer.load_data(_small_kg(tmp_path))
    walks_path, vocab_path = preparer.prepare_corpus(edges, str(tmp_path / "corpus"))
    walks = cudf.read_parquet(walks_path)
    assert len(walks) == 20 * 3
    assert walks["vertices"].list.len().max() == 4  # walk_depth + 1
    assert set(cudf.read_parquet(vocab_path).columns) == {"token", "word"}

    trainer = GPU_RDF2Vec(**settings)  # a fresh instance: no graph, no load_data
    trainer.fit_from_walks(walks_path, vocab_path)
    if backend == "pytorch":
        assert trainer.word2vec_model.grad_clip_value == 0.3
    emb = trainer.transform()
    assert len(emb) == len(preparer.word2idx)
    vectors = emb[[f"embedding_{i}" for i in range(4)]].to_pandas().to_numpy()
    assert np.isfinite(vectors).all() and np.abs(vectors).sum() > 0


@pytest.mark.gpu
def test_fit_from_walks_needs_both_files(tmp_path):
    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    with pytest.raises(FileNotFoundError):
        GPU_RDF2Vec().fit_from_walks(str(tmp_path / "missing"), str(tmp_path / "missing.parquet"))


@pytest.mark.gpu
def test_multi_gpu_without_client_only_fails_when_reading(tmp_path):
    """fit_from_walks needs no Dask client, so construction must not ask for one."""
    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    model = GPU_RDF2Vec(multi_gpu=True)
    assert model.client is None
    with pytest.raises(ValueError, match="needs a Dask client"):
        model.read_data(_small_kg(tmp_path))


@pytest.mark.gpu
def test_pytorch_progress_bar_refreshes_when_the_loss_is_logged(tmp_path):
    """The progress bar redraws only on the steps the loss is logged (every
    `log_every_n_steps`); redrawing it on every step costs a sizeable share
    of a small training step."""
    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    model = GPU_RDF2Vec(epochs=1, vector_size=4, window_size=2, walk_number=3, walk_depth=3)
    model.fit(model.load_data(_small_kg(tmp_path)))

    trainer = model.word2vec_model.trainer
    assert trainer.progress_bar_callback.refresh_rate == trainer.log_every_n_steps > 1


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["pytorch", "gensim"])
def test_single_gpu_fit_and_prepare_corpus_walk_in_chunks(tmp_path, monkeypatch, backend):
    """With a small chunk size the single-GPU walks come in several chunks, and
    fit / prepare_corpus still see all of them (20 start vertices x 3 walks)."""
    if backend == "gensim":
        pytest.importorskip("gensim")
    import pyarrow.parquet as pq

    from src.rdf2vecgpu.corpus import walk_corpus
    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    monkeypatch.setattr(walk_corpus, "SINGLE_GPU_WALKS_PER_CHUNK", 7)  # 2 start vertices per chunk
    chunks = []
    original = walk_corpus.SingleGPUWalkCorpus.walk_chunks

    def counting(self, *args, **kwargs):
        for chunk in original(self, *args, **kwargs):
            chunks.append(len(chunk[0]))
            yield chunk

    monkeypatch.setattr(walk_corpus.SingleGPUWalkCorpus, "walk_chunks", counting)
    settings = dict(backend=backend, epochs=1, vector_size=4, window_size=2, walk_number=3, walk_depth=3)

    model = GPU_RDF2Vec(**settings)
    model.fit(model.load_data(_small_kg(tmp_path)))
    assert chunks == [6] * 10
    vectors = model.transform()[[f"embedding_{i}" for i in range(4)]].to_pandas().to_numpy()
    assert np.isfinite(vectors).all() and np.abs(vectors).sum() > 0

    chunks.clear()
    preparer = GPU_RDF2Vec(**settings)
    walks_path, _ = preparer.prepare_corpus(preparer.load_data(_small_kg(tmp_path)), str(tmp_path / "corpus"))
    assert chunks == [6] * 10
    assert pq.read_table(walks_path).num_rows == 60


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend, model", [("pytorch", "skipgram"), ("pytorch", "cbow"), ("gensim", "skipgram")]
)
def test_single_gpu_bfs_walks_train_like_random_walks(tmp_path, backend, model):
    """BFS walks are walk sequences like random walks (root-to-leaf paths,
    -1 padded): every trainer takes them, and prepare_corpus writes them."""
    if backend == "gensim":
        pytest.importorskip("gensim")
    import pyarrow.parquet as pq

    from src.rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    settings = dict(backend=backend, embedding_model=model, walk_strategy="bfs", epochs=1, vector_size=4,
                    window_size=2, walk_depth=3)
    rdf = GPU_RDF2Vec(**settings)
    rdf.fit(rdf.load_data(_small_kg(tmp_path)))
    vectors = rdf.transform()[[f"embedding_{i}" for i in range(4)]].to_pandas().to_numpy()
    assert np.isfinite(vectors).all() and np.abs(vectors).sum() > 0

    preparer = GPU_RDF2Vec(**settings)
    walks_path, _ = preparer.prepare_corpus(preparer.load_data(_small_kg(tmp_path)), str(tmp_path / "corpus"))
    walks = pq.read_table(walks_path)
    # the 20-vertex cycle: from each vertex one path of 3 steps
    assert walks.num_rows == 20
    assert all(len(v) == 4 and -1 not in v for v in walks.column("vertices").to_pylist())
