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


def test_multi_gpu_cbow_on_bfs_walks_needs_gensim():
    """BFS walks reach PyTorch CBOW as pairs, which lose their context windows."""
    with pytest.raises(ValueError, match="multi-GPU CBOW on BFS walks"):
        RDF2VecConfig(multi_gpu=True, embedding_model="cbow", backend="pytorch", walk_strategy="bfs")
    RDF2VecConfig(multi_gpu=True, embedding_model="cbow", backend="pytorch")  # random walks
    RDF2VecConfig(multi_gpu=True, embedding_model="cbow", backend="gensim", walk_strategy="bfs")
    RDF2VecConfig(multi_gpu=False, embedding_model="cbow", backend="pytorch", walk_strategy="bfs")


def test_learning_rate_defaults():
    # skip-gram: a step size per pair, gensim's alpha, for both trainers
    assert RDF2VecConfig().learning_rate == 0.025
    assert RDF2VecConfig(backend="pytorch").learning_rate == 0.025
    assert RDF2VecConfig(backend="gensim").learning_rate == 0.025
    assert RDF2VecConfig(backend="gensim", embedding_model="cbow").learning_rate == 0.025
    assert RDF2VecConfig(embedding_model="cbow").learning_rate == 0.025
    # CBOW on BFS walks still uses Adam on a mean loss
    assert RDF2VecConfig(embedding_model="cbow", walk_strategy="bfs").learning_rate == 0.0001


def test_training_defaults_match_gensim():
    config = RDF2VecConfig()
    assert config.min_learning_rate == 0.0001
    assert config.sample == 0.001
    with pytest.raises(ValueError):
        RDF2VecConfig(sample=-0.1)


def test_grad_clip_value_default_and_validation():
    assert RDF2VecConfig().grad_clip_value == 1.0
    assert RDF2VecConfig(grad_clip_value=0).grad_clip_value == 0
    with pytest.raises(ValueError):
        RDF2VecConfig(grad_clip_value=-1)


def test_batch_size_default():
    assert RDF2VecConfig().batch_size == 32_768
    assert RDF2VecConfig(batch_size=4096).batch_size == 4096
    restored = RDF2VecConfig.model_validate_json(RDF2VecConfig().model_dump_json())
    assert restored.batch_size == 32_768


def test_tune_batch_size_is_deprecated_and_ignored():
    assert RDF2VecConfig().tune_batch_size is False
    with pytest.warns(FutureWarning, match="tune_batch_size is deprecated"):
        config = RDF2VecConfig(tune_batch_size=True)
    assert config.batch_size == 32_768
    with pytest.warns(FutureWarning):
        assert RDF2VecConfig(tune_batch_size=True, batch_size=4096).batch_size == 4096


def test_explicit_learning_rate_is_kept():
    assert RDF2VecConfig(backend="gensim", learning_rate=0.01).learning_rate == 0.01
    assert RDF2VecConfig(backend="pytorch", learning_rate=0.01).learning_rate == 0.01


def test_resolved_learning_rate_round_trips():
    """`save()` writes the resolved value; `load()` must read it back unchanged."""
    config = RDF2VecConfig(backend="gensim")
    restored = RDF2VecConfig.model_validate_json(config.model_dump_json())
    assert restored.learning_rate == 0.025


def test_cpu_count_defaults_to_the_available_cores():
    import os

    assert RDF2VecConfig().cpu_count == len(os.sched_getaffinity(0))
    assert RDF2VecConfig(cpu_count=3).cpu_count == 3


def test_order_aware_skipgram_needs_pytorch_and_random_walks():
    config = RDF2VecConfig(embedding_model="order_aware_skipgram")
    assert config.learning_rate == 0.025
    with pytest.raises(ValueError, match="needs backend='pytorch'"):
        RDF2VecConfig(embedding_model="order_aware_skipgram", backend="gensim")
    with pytest.raises(ValueError, match="needs walk_strategy='random'"):
        RDF2VecConfig(embedding_model="order_aware_skipgram", walk_strategy="bfs")
