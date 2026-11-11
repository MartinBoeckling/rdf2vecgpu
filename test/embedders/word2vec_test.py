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


def test_unigram_negatives_match_the_distribution_and_skip_unseen_tokens():
    torch.manual_seed(0)
    counts = np.array([0, 16, 0, 81, 1, 0], dtype=np.int64)  # count^0.75: 0, 8, 0, 27, 1, 0
    m = SkipGram(vocab_size=6, embedding_dim=4, neg_samples=5, negative_sampling="unigram", token_counts=counts)
    draws = m._sample_negatives(40_000, torch.device("cpu")).ravel()
    share = torch.bincount(draws, minlength=6).double() / draws.numel()
    assert share[[0, 2, 5]].sum().item() == 0.0
    expected = torch.tensor([0, 8, 0, 27, 1, 0], dtype=torch.double) / 36
    assert torch.allclose(share, expected, atol=0.01)


def test_expand_cbow_examples_of_full_and_ended_walks():
    from src.rdf2vecgpu.embedders.word2vec import CBOWNegativeSampling

    m = CBOWNegativeSampling(vocab_size=20, embedding_dim=4, neg_samples=1, window=1)
    # sequences [1, 10, 2, 11, 3] and [4, 12, 5]; walk 3 has no step
    vertices = torch.tensor([[1, 2, 3], [4, 5, -1], [7, -1, -1]])
    predicates = torch.tensor([[10, 11], [12, 0], [0, 0]])
    context, mask, center = m._expand_cbow(vertices, predicates)
    examples = sorted(
        (int(c), tuple(sorted(int(t) for t, k in zip(ctx, msk) if k))) for c, ctx, msk in zip(center, context, mask)
    )
    assert examples == sorted([
        (1, (10,)), (10, (1, 2)), (2, (10, 11)), (11, (2, 3)), (3, (11,)),
        (4, (12,)), (12, (4, 5)), (5, (12,)),
    ])


def test_cbow_negative_sampling_learns_clustered_cooccurrence_from_walks():
    from src.rdf2vecgpu.embedders.word2vec import CBOWNegativeSampling

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
    model = CBOWNegativeSampling(vocab_size=20, embedding_dim=8, neg_samples=3, learning_rate=0.05,
                                 epochs=epochs, window=2, steps_per_epoch=len(loader),
                                 negative_sampling="unigram", token_counts=counts)
    L.Trainer(max_epochs=epochs, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False).fit(model, loader)
    assert torch.isfinite(model.get_embeddings()).all()
    assert _cluster_gap(model.get_embeddings()) > 0.1


def test_expand_skipgram_offsets_are_context_minus_center_position():
    m = SkipGram(vocab_size=20, embedding_dim=4, neg_samples=1, window=2)
    vertices = torch.tensor([[1, 2, 3], [4, 5, -1]])
    predicates = torch.tensor([[10, 11], [12, 0]])
    center, context, offsets = m._expand_skipgram(vertices, predicates, with_offsets=True)
    sequences = [[1, 10, 2, 11, 3], [4, 12, 5]]
    expected = sorted(
        (seq[i], seq[j], j - i)
        for seq in sequences for i in range(len(seq)) for j in range(len(seq)) if 0 < abs(i - j) <= 2
    )
    assert sorted(zip(center.tolist(), context.tolist(), offsets.tolist())) == expected


def test_order_aware_output_rows_per_offset():
    from src.rdf2vecgpu.embedders.word2vec import OrderAwareSkipGram

    m = OrderAwareSkipGram(vocab_size=7, embedding_dim=4, neg_samples=2, window=2)
    assert m.output_embeddings.num_embeddings == 2 * 2 * 7
    offsets = torch.tensor([-2, -1, 1, 2])
    tokens = torch.tensor([3, 3, 3, 3])
    assert m._offset_rows(offsets, tokens).tolist() == [0 * 7 + 3, 1 * 7 + 3, 2 * 7 + 3, 3 * 7 + 3]
    with pytest.raises(ValueError, match="trains on walks"):
        m.training_step((torch.tensor([1]), torch.tensor([2])), 0)


def test_order_aware_skipgram_learns_position_specific_contexts():
    """Token 0 is always followed by 1 and preceded by 2. Plain skip-gram sees
    both as context of 0; order-aware skip-gram must also separate the sides:
    0's score for 1 is high at offset +1 and low at offset -1."""
    from src.rdf2vecgpu.embedders.word2vec import OrderAwareSkipGram

    L.seed_everything(0, workers=True)
    n = 2000
    # sequences 2, 5, 0, 6, 1 (vertices 2, 0, 1; predicates 5, 6)
    vertices = torch.tensor([[2, 0, 1]]).repeat(n, 1)
    predicates = torch.tensor([[5, 6]]).repeat(n, 1)
    loader = DataLoader(TensorDataset(vertices, predicates), batch_size=50, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    epochs = 5
    model = OrderAwareSkipGram(vocab_size=8, embedding_dim=8, neg_samples=3, window=2, learning_rate=0.05,
                               epochs=epochs, steps_per_epoch=len(loader))
    L.Trainer(max_epochs=epochs, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False).fit(model, loader)
    center = model.in_embeddings.weight[0]
    out = model.output_embeddings.weight

    def score(offset, token):
        return float(center @ out[model._offset_rows(torch.tensor(offset), torch.tensor(token))])

    assert score(2, 1) > score(-2, 1) + 1.0  # 1 is two positions after 0
    assert score(-2, 2) > score(2, 2) + 1.0  # 2 is two positions before 0


@pytest.mark.parametrize("model_name", ["skipgram", "cbow", "order_aware"])
@pytest.mark.parametrize("sparse", [True, False])
def test_empty_batch_loss_still_reaches_every_embedding_table(model_name, sparse):
    """Under DDP every rank must produce gradients each step, or the others
    wait forever in the all-reduce: a batch with no pairs left (here: a walk
    without a step) must still give a loss connected to the embeddings."""
    from src.rdf2vecgpu.embedders.word2vec import CBOWNegativeSampling, OrderAwareSkipGram

    cls = {"skipgram": SkipGram, "cbow": CBOWNegativeSampling, "order_aware": OrderAwareSkipGram}[model_name]
    m = cls(vocab_size=10, embedding_dim=4, neg_samples=2, window=2, use_sparse=sparse)
    loss = m.training_step((torch.tensor([[3, -1, -1]]), torch.tensor([[0, 0]])), 0)
    assert loss.item() == 0.0 and loss.requires_grad
    loss.backward()
    assert m.in_embeddings.weight.grad is not None
    assert m.output_embeddings.weight.grad is not None


def test_output_tables_start_at_zero_like_word2vec():
    from src.rdf2vecgpu.embedders.word2vec import CBOWNegativeSampling, OrderAwareSkipGram

    for cls in (SkipGram, CBOWNegativeSampling, OrderAwareSkipGram):
        m = cls(vocab_size=10, embedding_dim=4, neg_samples=2, window=2)
        assert torch.count_nonzero(m.output_embeddings.weight) == 0, cls.__name__
        assert torch.count_nonzero(m.in_embeddings.weight) > 0, cls.__name__


def test_negatives_equal_to_the_positive_token_are_left_out():
    m = _skipgram(use_sparse=False)
    with torch.no_grad():
        m.output_embeddings.weight.normal_()
    center, context = torch.tensor([1]), torch.tensor([4])
    negative = torch.tensor([[4, 7]])  # the first negative is the positive token
    loss_function = torch.nn.BCEWithLogitsLoss(reduction="sum")
    c = m.in_embeddings.weight[1]
    expected = loss_function((c @ m.output_embeddings.weight[4]).view(1), torch.ones(1)) + loss_function(
        (c @ m.output_embeddings.weight[7]).view(1), torch.zeros(1)
    )
    got = m(center, context, negative, negative != context.unsqueeze(1))
    assert torch.allclose(got, expected)
