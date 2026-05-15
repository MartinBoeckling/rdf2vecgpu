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


def test_pytorch_convergence_config_defaults():
    c = RDF2VecConfig()
    # 32768 = walk-coverage rewrite default (spec 2026-05-15); raised from 16384 for
    # Lightning per-step overhead, convergence preconditions unchanged.
    assert c.pair_batch_size == 32768
    assert c.accumulate_grad_batches == 1
    assert c.min_learning_rate == 0.0001
    assert c.lr_decay == "linear"
    # reconciled classic-word2vec defaults
    assert c.learning_rate == 0.025
    assert c.tune_batch_size is False


def test_pytorch_convergence_config_overrides_and_validation():
    c = RDF2VecConfig(pair_batch_size=4096, accumulate_grad_batches=2,
                      min_learning_rate=0.0, lr_decay="none", learning_rate=0.01,
                      tune_batch_size=True)
    assert c.pair_batch_size == 4096 and c.accumulate_grad_batches == 2
    assert c.min_learning_rate == 0.0 and c.lr_decay == "none"
    assert c.learning_rate == 0.01 and c.tune_batch_size is True
    with pytest.raises(ValueError):
        RDF2VecConfig(pair_batch_size=0)
    with pytest.raises(ValueError):
        RDF2VecConfig(accumulate_grad_batches=0)
    with pytest.raises(ValueError):
        RDF2VecConfig(lr_decay="exponential")


def test_pytorch_quality_knob_config_defaults():
    c = RDF2VecConfig()
    assert c.optimizer == "sgd"
    assert c.grad_clip_value == 0.1
    assert c.grad_clip_mode == "element"


def test_pytorch_quality_knob_config_overrides_and_validation():
    c = RDF2VecConfig(optimizer="sparseadam", sample=1e-3, negative_sampling="unigram",
                      grad_clip_value=0.5, grad_clip_mode="row", learning_rate=0.0025)
    assert c.optimizer == "sparseadam" and c.sample == 1e-3
    assert c.negative_sampling == "unigram"
    assert c.grad_clip_value == 0.5 and c.grad_clip_mode == "row"
    with pytest.raises(ValueError):
        RDF2VecConfig(optimizer="adamw")
    with pytest.raises(ValueError):
        RDF2VecConfig(negative_sampling="zipf")
    with pytest.raises(ValueError):
        RDF2VecConfig(grad_clip_mode="global")
    with pytest.raises(ValueError):
        RDF2VecConfig(sample=-1.0)


def test_word2vec_config_default_sample_and_negative_sampling():
    """Spec §7.4 sub_f075 defaults: sample=1e-3 + f^0.75 unigram negatives
    are now the new defaults (overnight-sweep validated as a clean class +
    regression win that never hurts training)."""
    c = RDF2VecConfig()
    assert c.sample == 0.001, f"expected sample=0.001, got {c.sample}"
    assert c.negative_sampling == "unigram", (
        f"expected negative_sampling='unigram', got {c.negative_sampling!r}"
    )
