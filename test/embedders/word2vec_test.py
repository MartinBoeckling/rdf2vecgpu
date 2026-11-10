"""SkipGram (sum loss, SGD with linear decay, gradient clipping, on-GPU pair
expansion of walks, subsampling, unigram negatives) and CBOW embeddings."""
import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.rdf2vecgpu.embedders.word2vec import CBOW, SkipGram


def _skipgram(**kwargs):
    return SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, **kwargs)


def test_skipgram_defaults():
    m = _skipgram()
    assert m.in_embeddings.sparse and m.output_embeddings.sparse
    assert m.learning_rate == 0.025 and m.min_learning_rate == 0.0001
    assert m.sample == 0.0 and m.negative_sampling == "uniform"
    assert hasattr(m, "hparams")


def test_skipgram_rejects_unknown_options():
    with pytest.raises(ValueError):
        _skipgram(lr_decay="cosine")
    with pytest.raises(ValueError):
        _skipgram(negative_sampling="zipf")


def test_skipgram_linear_decay_per_step():
    m = _skipgram(epochs=4, min_learning_rate=0.0001, steps_per_epoch=25)
    out = m.configure_optimizers()
    assert isinstance(out["optimizer"], torch.optim.SGD)
    assert out["optimizer"].param_groups[0]["lr"] == 0.025
    assert out["lr_scheduler"]["interval"] == "step"
    scheduler = out["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)
    assert scheduler.total_iters == 4 * 25 - 1  # the last step runs at min_learning_rate
    assert abs(scheduler.end_factor - 0.0001 / 0.025) < 1e-9


def test_skipgram_decay_fallbacks():
    out = _skipgram(epochs=4, steps_per_epoch=None).configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "epoch"
    assert out["lr_scheduler"]["scheduler"].total_iters == 3
    assert isinstance(_skipgram(epochs=4, lr_decay="none").configure_optimizers(), torch.optim.SGD)
    assert isinstance(_skipgram(epochs=1).configure_optimizers(), torch.optim.SGD)
    # one epoch with a known step count still decays per step
    one = _skipgram(epochs=1, steps_per_epoch=10).configure_optimizers()
    assert one["lr_scheduler"]["scheduler"].total_iters == 9


def test_skipgram_loss_is_summed_over_pairs():
    m = _skipgram()
    center = torch.tensor([1, 2, 3])
    context = torch.tensor([4, 5, 6])
    negative = torch.tensor([[7, 8], [8, 9], [9, 7]])
    total = m(center, context, negative)
    parts = sum(m(center[i : i + 1], context[i : i + 1], negative[i : i + 1]) for i in range(3))
    assert torch.allclose(total, parts)


def test_skipgram_clips_coalesced_sparse_grad():
    m = _skipgram(grad_clip_value=0.1)
    # row 1 appears twice: its coalesced sum 0.6 is clipped; row 3 (0.05) is not
    idx = torch.tensor([[1, 1, 3]])
    val = torch.tensor([[0.3] * 4, [0.3] * 4, [0.05] * 4])
    m.in_embeddings.weight.grad = torch.sparse_coo_tensor(idx, val, size=(10, 4))
    m.output_embeddings.weight.grad = None
    m.on_after_backward()
    dense = m.in_embeddings.weight.grad.coalesce().to_dense()
    assert torch.allclose(dense[1], torch.full((4,), 0.1))
    assert torch.allclose(dense[3], torch.full((4,), 0.05))

    off = _skipgram(grad_clip_value=0.0)
    off.in_embeddings.weight.grad = torch.sparse_coo_tensor(idx, val, size=(10, 4))
    off.on_after_backward()
    assert torch.allclose(off.in_embeddings.weight.grad.coalesce().to_dense()[1], torch.full((4,), 0.6))


def test_skipgram_clips_dense_grad():
    m = _skipgram(grad_clip_value=0.5, use_sparse=False)
    m.in_embeddings.weight.grad = torch.full((10, 4), 2.0)
    m.output_embeddings.weight.grad = torch.full((10, 4), -0.2)
    m.on_after_backward()
    assert torch.all(m.in_embeddings.weight.grad == 0.5)
    assert torch.all(m.output_embeddings.weight.grad == -0.2)


def test_expand_skipgram_pairs_of_full_and_ended_walks():
    m = SkipGram(vocab_size=20, embedding_dim=4, neg_samples=1, window=1)
    # walk 0: 1 -p10-> 2 -p11-> 3; walk 1 ends after one step: 4 -p12-> 5, -1
    vertices = torch.tensor([[1, 2, 3], [4, 5, -1]])
    predicates = torch.tensor([[10, 11], [12, 0]])
    center, context = m._expand_skipgram(vertices, predicates)
    pairs = sorted(zip(center.tolist(), context.tolist()))
    # sequences [1, 10, 2, 11, 3] and [4, 12, 5]; window 1 = neighbours
    expected = sorted(
        [(1, 10), (10, 1), (10, 2), (2, 10), (2, 11), (11, 2), (11, 3), (3, 11)]
        + [(4, 12), (12, 4), (12, 5), (5, 12)]
    )
    assert pairs == expected


def test_expand_skipgram_drops_walks_without_a_step():
    m = SkipGram(vocab_size=20, embedding_dim=4, neg_samples=1, window=2)
    center, context = m._expand_skipgram(torch.tensor([[7, -1, -1]]), torch.tensor([[0, 0]]))
    assert center.numel() == 0 and context.numel() == 0


def test_skipgram_requires_token_counts_for_sample_or_unigram():
    with pytest.raises(ValueError):
        _skipgram(sample=1e-3)
    with pytest.raises(ValueError):
        _skipgram(negative_sampling="unigram")
    with pytest.raises(ValueError):
        _skipgram(sample=1e-3, token_counts=[1, 2, 3])  # wrong length
    SkipGram(vocab_size=3, embedding_dim=4, neg_samples=2, token_counts=[1, 2, 3])


def test_skipgram_unigram_negatives_follow_counts_to_075():
    torch.manual_seed(0)
    counts = [10000, 1, 1, 1, 1]
    m = SkipGram(vocab_size=5, embedding_dim=4, neg_samples=4, negative_sampling="unigram", token_counts=counts)
    negatives = m._sample_negatives(2000, torch.device("cpu"))
    assert negatives.shape == (2000, 4)
    # weights 10000^0.75 = 1000 against 1 for the others
    assert (negatives == 0).float().mean().item() > 0.95
    uniform = SkipGram(vocab_size=5, embedding_dim=4, neg_samples=4)
    assert (uniform._sample_negatives(2000, torch.device("cpu")) == 0).float().mean().item() < 0.5


def test_skipgram_subsampling_thins_pairs_of_frequent_tokens():
    torch.manual_seed(0)
    # token 0 makes up 99.96% of the corpus; tokens 2, 5, 6 are rare
    counts = [999600, 0, 100, 0, 0, 100, 100, 100]
    vertices = torch.tensor([[5, 0, 6]]).repeat(300, 1)
    predicates = torch.tensor([[2, 2]]).repeat(300, 1)
    off = SkipGram(vocab_size=8, embedding_dim=4, neg_samples=2, window=2, token_counts=counts)
    on = SkipGram(vocab_size=8, embedding_dim=4, neg_samples=2, window=2, sample=1e-3, token_counts=counts)
    c_off, x_off = off._expand_skipgram(vertices, predicates)
    c_on, x_on = on._expand_skipgram(vertices, predicates)
    assert ((c_off == 0) | (x_off == 0)).float().mean().item() > 0.4
    assert ((c_on == 0) | (x_on == 0)).float().mean().item() < 0.15
    # pairs of rare tokens only are all kept
    assert int(((c_on != 0) & (x_on != 0)).sum()) == int(((c_off != 0) & (x_off != 0)).sum())


def _clustered_pairs(n=8000, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randint(0, 20, (n,), generator=g)
    context = (centers // 10) * 10 + torch.randint(0, 10, (n,), generator=g)
    return centers.long(), context.long()


def _cluster_gap(embeddings):
    emb = torch.nn.functional.normalize(embeddings, dim=1)
    sim = emb @ emb.t()
    intra = ((sim[:10, :10].sum() + sim[10:, 10:].sum()) - 20.0) / (2 * 90)
    return intra.item() - sim[:10, 10:].mean().item()


def test_skipgram_learns_clustered_cooccurrence_from_pairs():
    L.seed_everything(0, workers=True)
    centers, context = _clustered_pairs()
    loader = DataLoader(TensorDataset(centers, context), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    epochs = 15
    model = SkipGram(vocab_size=20, embedding_dim=8, neg_samples=3, learning_rate=0.1,
                     epochs=epochs, min_learning_rate=1e-3, window=2, steps_per_epoch=len(loader))
    L.Trainer(max_epochs=epochs, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False).fit(model, loader)
    assert torch.isfinite(model.get_embeddings()).all()
    assert _cluster_gap(model.get_embeddings()) > 0.1


def test_skipgram_learns_clustered_cooccurrence_from_walks():
    """Walks inside one of two vertex clusters: the pairs are built from the
    walks by `_expand_skipgram`, as on the GPU path."""
    L.seed_everything(0, workers=True)
    g = torch.Generator().manual_seed(0)
    n = 3000
    cluster = torch.randint(0, 2, (n, 1), generator=g)
    vertices = cluster * 10 + torch.randint(0, 10, (n, 3), generator=g)
    predicates = cluster * 10 + torch.randint(0, 10, (n, 2), generator=g)
    loader = DataLoader(TensorDataset(vertices, predicates), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    counts = np.bincount(torch.cat([vertices.ravel(), predicates.ravel()]).numpy(), minlength=20)
    epochs = 10
    model = SkipGram(vocab_size=20, embedding_dim=8, neg_samples=3, learning_rate=0.05,
                     epochs=epochs, window=2, steps_per_epoch=len(loader), sample=1e-3,
                     negative_sampling="unigram", token_counts=counts)
    L.Trainer(max_epochs=epochs, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False).fit(model, loader)
    assert torch.isfinite(model.get_embeddings()).all()
    assert _cluster_gap(model.get_embeddings()) > 0.1


def test_cbow_learns_clustered_cooccurrence_and_has_embeddings():
    L.seed_everything(0, workers=True)
    g = torch.Generator().manual_seed(0)
    n, n_ctx = 8000, 4
    centers = torch.randint(0, 20, (n,), generator=g)
    context = ((centers // 10).unsqueeze(1) * 10 + torch.randint(0, 10, (n, n_ctx), generator=g)).long()
    loader = DataLoader(TensorDataset(context, centers.long()), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    model = CBOW(vocab_size=20, embedding_dim=8, learning_rate=0.05)
    L.Trainer(max_epochs=20, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False).fit(model, loader)
    assert torch.isfinite(model.get_embeddings()).all()
    assert _cluster_gap(model.get_embeddings()) > 0.1
