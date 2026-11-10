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


def test_multi_gpu_cbow_needs_gensim():
    """Multi-GPU CBOW pairs on the PyTorch backend lose their context windows."""
    with pytest.raises(ValueError, match="multi-GPU CBOW"):
        RDF2VecConfig(multi_gpu=True, embedding_model="cbow", backend="pytorch")
    RDF2VecConfig(multi_gpu=True, embedding_model="cbow", backend="gensim")
    RDF2VecConfig(multi_gpu=False, embedding_model="cbow", backend="pytorch")


def test_learning_rate_defaults():
    # skip-gram: a step size per pair, gensim's alpha, for both trainers
    assert RDF2VecConfig().learning_rate == 0.025
    assert RDF2VecConfig(backend="pytorch").learning_rate == 0.025
    assert RDF2VecConfig(backend="gensim").learning_rate == 0.025
    assert RDF2VecConfig(backend="gensim", embedding_model="cbow").learning_rate == 0.025
    # PyTorch CBOW uses Adam on a mean loss
    assert RDF2VecConfig(embedding_model="cbow").learning_rate == 0.0001


def test_training_defaults_match_gensim():
    config = RDF2VecConfig()
    assert config.min_learning_rate == 0.0001
    assert config.sample == 0.001
    with pytest.raises(ValueError):
        RDF2VecConfig(sample=-0.1)


def test_batch_size_default_and_tuning_are_exclusive():
    assert RDF2VecConfig().batch_size == 32_768
    assert RDF2VecConfig().tune_batch_size is False
    assert RDF2VecConfig(batch_size=4096).batch_size == 4096
    tuned = RDF2VecConfig(tune_batch_size=True)
    assert tuned.batch_size is None
    with pytest.raises(ValueError, match="leave batch_size unset"):
        RDF2VecConfig(tune_batch_size=True, batch_size=4096)


def test_resolved_batch_size_round_trips():
    for config in (RDF2VecConfig(), RDF2VecConfig(tune_batch_size=True)):
        restored = RDF2VecConfig.model_validate_json(config.model_dump_json())
        assert restored.batch_size == config.batch_size


def test_explicit_learning_rate_is_kept():
    assert RDF2VecConfig(backend="gensim", learning_rate=0.01).learning_rate == 0.01
    assert RDF2VecConfig(backend="pytorch", learning_rate=0.01).learning_rate == 0.01


def test_resolved_learning_rate_round_trips():
    """`save()` writes the resolved value; `load()` must read it back unchanged."""
    config = RDF2VecConfig(backend="gensim")
    restored = RDF2VecConfig.model_validate_json(config.model_dump_json())
    assert restored.learning_rate == 0.025
