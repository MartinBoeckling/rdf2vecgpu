import pytest
from src.rdf2vecgpu.config import RDF2VecConfig


def test_default_config():
    config = RDF2VecConfig(
        walk_strategy="random",
        walk_depth=4,
        walk_number=100,
        embedding_model="skipgram",
        epochs=5,
        batch_size=None,
        vector_size=100,
        window_size=5,
        min_count=1,
        learning_rate=0.01,
        negative_samples=5,
        random_state=42,
        reproducible=False,
        multi_gpu=False,
        generate_artifact=False,
        cpu_count=20,
        backend="pytorch",
        tune_batch_size=True,
        num_nodes=1,
        tracker="none",
        tracker_kwargs={},
    )
    assert config.walk_depth == 4
    assert config.embedding_model == "skipgram"
    assert config.epochs == 5
    assert config.vector_size == 100
    assert config.learning_rate == 0.01
    assert config.random_state == 42
    assert config.cpu_count == 20


def test_config_reject_invalid_values():
    with pytest.raises(ValueError):
        RDF2VecConfig(walk_depth=0)
    with pytest.raises(ValueError):
        RDF2VecConfig(embedding_model="invalid_model")
    with pytest.raises(ValueError):
        RDF2VecConfig(learning_rate=-0.01)


def test_config_reject_invalid_strategy():
    with pytest.raises(ValueError) as exec_info:
        RDF2VecConfig(walk_strategy="invalid_strategy")

    msg = str(exec_info.value)
    assert "random" in msg
    assert "bfs" in msg


def test_config_reject_invalid_backend():
    with pytest.raises(ValueError) as exec_info:
        RDF2VecConfig(backend="invalid_backend")
    msg = str(exec_info.value)
    assert "pytorch" in msg
    assert "gensim" in msg


def test_config_reject_invalid_tracker():
    with pytest.raises(ValueError) as exec_info:
        rdf2vec_config = RDF2VecConfig(tracker="invalid_tracker")
    msg = str(exec_info.value)
    assert "mlflow" in msg
    assert "wandb" in msg
    assert "none" in msg
