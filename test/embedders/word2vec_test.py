import torch
from rdf2vecgpu.embedders.word2vec import SkipGram


def test_skipgram_new_kwargs_accepted_and_stored():
    m = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, learning_rate=0.025,
                 epochs=3, min_learning_rate=0.001, lr_decay="linear", window=5,
                 accumulate_grad_batches=2, steps_per_epoch=50)
    assert m.epochs == 3 and m.window == 5
    assert m.accumulate_grad_batches == 2 and m.steps_per_epoch == 50
    assert m.in_embeddings.sparse is True and m.output_embeddings.sparse is True
    assert hasattr(m, "hparams")  # save_hyperparameters() was called


def test_skipgram_configure_optimizers_per_step_lineardecay():
    m = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, learning_rate=0.025,
                 epochs=4, min_learning_rate=0.0001, lr_decay="linear",
                 accumulate_grad_batches=1, steps_per_epoch=25)
    out = m.configure_optimizers()
    assert isinstance(out, dict) and isinstance(out["optimizer"], torch.optim.SGD)
    assert out["optimizer"].param_groups[0]["lr"] == 0.025
    sched_cfg = out["lr_scheduler"]
    assert sched_cfg["interval"] == "step"
    sched = sched_cfg["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.LinearLR)
    assert sched.total_iters == 4 * 25 - 1                       # epochs * opt-steps/epoch - 1
    assert abs(sched.end_factor - (0.0001 / 0.025)) < 1e-9

    # accumulate_grad_batches divides the optimizer-step count per epoch
    m2 = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, learning_rate=0.025,
                  epochs=4, min_learning_rate=0.0001, lr_decay="linear",
                  accumulate_grad_batches=5, steps_per_epoch=23)
    s2 = m2.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert s2.total_iters == 4 * ((23 + 5 - 1) // 5) - 1         # ceil(23/5) = 5 → 4*5-1 = 19


def test_skipgram_configure_optimizers_fallbacks():
    # steps_per_epoch unknown → epoch-interval LinearLR
    m = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, epochs=4,
                 lr_decay="linear", steps_per_epoch=None)
    out = m.configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "epoch"
    assert out["lr_scheduler"]["scheduler"].total_iters == 3
    # lr_decay="none" → plain SGD, no scheduler
    m2 = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, epochs=4, lr_decay="none")
    assert isinstance(m2.configure_optimizers(), torch.optim.SGD)
    # epochs <= 1 → plain SGD, no scheduler
    m3 = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, epochs=1, lr_decay="linear",
                  steps_per_epoch=100)
    assert isinstance(m3.configure_optimizers(), torch.optim.SGD)


def test_skipgram_learns_clustered_cooccurrence():
    import lightning as L
    from torch.utils.data import TensorDataset, DataLoader

    L.seed_everything(0, workers=True)
    g = torch.Generator().manual_seed(0)
    n_pairs = 8000
    centers = torch.randint(0, 20, (n_pairs,), generator=g)
    cluster = centers // 10                                   # 0 for tokens 0..9, 1 for 10..19
    context = (cluster * 10 + torch.randint(0, 10, (n_pairs,), generator=g)).long()
    dl = DataLoader(TensorDataset(centers.long(), context), batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(0))

    epochs = 15
    model = SkipGram(vocab_size=20, embedding_dim=8, neg_samples=3, learning_rate=0.1,
                     epochs=epochs, min_learning_rate=1e-3, lr_decay="linear", window=2,
                     steps_per_epoch=(n_pairs + 63) // 64)
    trainer = L.Trainer(max_epochs=epochs, accelerator="cpu", logger=False,
                        enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model, dl)

    emb = torch.nn.functional.normalize(model.get_embeddings(), dim=1)
    sim = emb @ emb.t()
    intra = ((sim[:10, :10].sum() + sim[10:, 10:].sum()) - 20.0) / (2 * 90)   # exclude diagonal
    inter = sim[:10, 10:].mean()
    assert torch.isfinite(emb).all()
    assert intra.item() > inter.item() + 0.1, (intra.item(), inter.item())


def test_skipgram_on_after_backward_clips_sparse_grad():
    m = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, grad_clip_value=0.1,
                 use_sparse=True)
    # craft a sparse COO grad for in_embeddings: row 1 appears TWICE (so the
    # coalesced sum is 0.6 → must clamp to 0.1); row 3 appears once with 0.05
    # (< 0.1 → must be left alone). output_embeddings.grad stays None (hook skips it).
    idx = torch.tensor([[1, 1, 3]])
    val = torch.tensor([[0.3, 0.3, 0.3, 0.3],
                        [0.3, 0.3, 0.3, 0.3],
                        [0.05, 0.05, 0.05, 0.05]])
    m.in_embeddings.weight.grad = torch.sparse_coo_tensor(idx, val, size=(10, 4))
    m.output_embeddings.weight.grad = None
    m.on_after_backward()
    dense = m.in_embeddings.weight.grad.coalesce().to_dense()
    assert torch.allclose(dense[1], torch.full((4,), 0.1)), dense[1]   # 0.3+0.3=0.6 → clamped
    assert torch.allclose(dense[3], torch.full((4,), 0.05)), dense[3]  # 0.05 → untouched
    # disabled when grad_clip_value == 0
    m2 = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, grad_clip_value=0.0,
                  use_sparse=True)
    m2.in_embeddings.weight.grad = torch.sparse_coo_tensor(idx, val, size=(10, 4))
    m2.output_embeddings.weight.grad = None
    m2.on_after_backward()
    assert torch.allclose(m2.in_embeddings.weight.grad.coalesce().to_dense()[1], torch.full((4,), 0.6))


# ── new convergence/quality knobs: optimizer, grad_clip_mode, sample, negative_sampling ──

def test_skipgram_optimizer_sparseadam():
    m = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, learning_rate=0.0025,
                 use_sparse=True, epochs=4, lr_decay="linear", steps_per_epoch=25,
                 optimizer="sparseadam")
    out = m.configure_optimizers()
    assert isinstance(out, dict) and isinstance(out["optimizer"], torch.optim.SparseAdam)
    assert out["optimizer"].param_groups[0]["lr"] == 0.0025
    assert isinstance(out["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.LinearLR)
    # default is still SGD
    m2 = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, epochs=4, lr_decay="linear",
                  steps_per_epoch=25)
    assert isinstance(m2.configure_optimizers()["optimizer"], torch.optim.SGD)
    # sparseadam + non-sparse embeddings → dense Adam fallback (SparseAdam can't take dense grads)
    m3 = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, epochs=1, use_sparse=False,
                  optimizer="sparseadam", lr_decay="none")
    assert isinstance(m3.configure_optimizers(), torch.optim.Adam)


def test_skipgram_on_after_backward_row_clip():
    m = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, grad_clip_value=0.1,
                 grad_clip_mode="row", use_sparse=True)
    # row 1 twice (coalesced sum 0.6·1⃗ → ‖·‖₂ = 1.2 ≫ 0.1 → rescale to 0.1);
    # row 3 once with ‖·‖₂ = 0.05·2 = 0.1 exactly → boundary, must not blow up;
    # row 7 once with ‖·‖₂ = 0.01·2 = 0.02 < 0.1 → untouched.
    idx = torch.tensor([[1, 1, 3, 7]])
    val = torch.tensor([[0.3, 0.3, 0.3, 0.3],
                        [0.3, 0.3, 0.3, 0.3],
                        [0.05, 0.05, 0.05, 0.05],
                        [0.01, 0.01, 0.01, 0.01]])
    m.in_embeddings.weight.grad = torch.sparse_coo_tensor(idx, val, size=(10, 4))
    m.output_embeddings.weight.grad = None
    m.on_after_backward()
    dense = m.in_embeddings.weight.grad.coalesce().to_dense()
    assert abs(dense[1].norm().item() - 0.1) < 1e-6, dense[1].norm().item()
    assert dense[1].sign().tolist() == [1.0, 1.0, 1.0, 1.0]              # direction preserved
    assert dense[3].norm().item() <= 0.1 + 1e-6
    assert torch.allclose(dense[7], val[3])                              # below threshold → untouched
    # dense path too
    md = SkipGram(vocab_size=4, embedding_dim=3, neg_samples=1, grad_clip_value=0.5,
                  grad_clip_mode="row", use_sparse=False)
    md.in_embeddings.weight.grad = torch.tensor([[3.0, 4.0, 0.0],       # ‖·‖=5 → scale to 0.5
                                                 [0.1, 0.0, 0.0],       # ‖·‖=0.1 → keep
                                                 [0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]])
    md.output_embeddings.weight.grad = torch.zeros(4, 3)
    md.on_after_backward()
    g = md.in_embeddings.weight.grad
    assert abs(g[0].norm().item() - 0.5) < 1e-6
    assert torch.allclose(g[0], torch.tensor([0.3, 0.4, 0.0]))          # 5→0.5: ×0.1
    assert torch.allclose(g[1], torch.tensor([0.1, 0.0, 0.0]))


def test_skipgram_requires_token_counts_for_sample_or_unigram():
    import pytest
    with pytest.raises(ValueError):
        SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, sample=1e-3)
    with pytest.raises(ValueError):
        SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, negative_sampling="unigram")
    with pytest.raises(ValueError):
        SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, sample=1e-3,
                 token_counts=[1, 2, 3])                                # wrong length
    # token_counts without sample/unigram is allowed (just unused)
    SkipGram(vocab_size=3, embedding_dim=4, neg_samples=2, token_counts=[1, 2, 3])


def test_skipgram_unigram_negatives_follow_f075():
    torch.manual_seed(0)
    counts = [10000, 1, 1, 1, 1]                                        # token 0 dominates
    m = SkipGram(vocab_size=5, embedding_dim=4, neg_samples=4, negative_sampling="unigram",
                 token_counts=counts)
    negs = m._sample_negatives(2000, torch.device("cpu"))
    assert negs.shape == (2000, 4)
    # weights ∝ count^0.75 → token 0 has weight 1000 vs 1 for the rest → ~99.6% of draws
    assert (negs == 0).float().mean().item() > 0.95
    assert negs.min().item() >= 0 and negs.max().item() < 5
    # uniform default spreads across the vocab
    mu = SkipGram(vocab_size=5, embedding_dim=4, neg_samples=4)
    negs_u = mu._sample_negatives(2000, torch.device("cpu"))
    assert (negs_u == 0).float().mean().item() < 0.5


def test_skipgram_subsampling_thins_hot_token_pairs():
    torch.manual_seed(0)
    # vocab 8: token 0 is a "hot predicate" (99.96% of the corpus); tokens 5,6 are
    # rare entities; token 2 a rare predicate. Walks: vertices [5,?,6] interleaved
    # with predicate 2 — but we put the hot token 0 as the *middle vertex* so it
    # sits in the window of both 5 and 6: vertices = [5, 0, 6], predicates = [2, 2]
    # → interleaved [5, 2, 0, 2, 6], window 2 → token 0 (position 2) is in 4 pairs.
    counts = [999600, 0, 100, 0, 0, 100, 100, 100]                      # total 1e6
    n = 300
    vertices = torch.tensor([[5, 0, 6]]).repeat(n, 1)
    predicates = torch.tensor([[2, 2]]).repeat(n, 1)

    m_off = SkipGram(vocab_size=8, embedding_dim=4, neg_samples=2, window=2,
                     sample=0.0, token_counts=counts)                   # token_counts unused here
    c_off, x_off = m_off._expand_skipgram(vertices, predicates)
    touches0_off = ((c_off == 0) | (x_off == 0)).float().mean().item()

    m_on = SkipGram(vocab_size=8, embedding_dim=4, neg_samples=2, window=2,
                    sample=1e-3, token_counts=counts)
    c_on, x_on = m_on._expand_skipgram(vertices, predicates)
    touches0_on = ((c_on == 0) | (x_on == 0)).float().mean().item()

    # z(0)=0.9996 → p_keep ≈ (sqrt(0.9996/0.001)+1)·0.001/0.9996 ≈ 0.0326 — pairs
    # touching token 0 should collapse from ~the same fraction to a small one,
    # while the rare tokens (z ≈ 1e-4 → p_keep clamped to 1) are all kept.
    assert touches0_off > 0.4, touches0_off
    assert touches0_on < 0.15, (touches0_off, touches0_on)
    # the non-hot pairs survive: total pairs not touching token 0 are unchanged
    n_keep_pairs_off = int(((c_off != 0) & (x_off != 0)).sum())
    n_keep_pairs_on = int(((c_on != 0) & (x_on != 0)).sum())
    assert n_keep_pairs_on == n_keep_pairs_off, (n_keep_pairs_on, n_keep_pairs_off)


def test_skipgram_default_sample_and_negative_sampling_match_config():
    """SkipGram's __init__ defaults must mirror RDF2VecConfig (spec §7.4)."""
    import numpy as np
    import torch
    from src.rdf2vecgpu.embedders.word2vec import SkipGram

    # Defaults now require token_counts; passing minimal counts here.
    counts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    m = SkipGram(vocab_size=10, embedding_dim=4, neg_samples=2, token_counts=counts)
    assert m.sample == 0.001, f"expected sample=0.001, got {m.sample}"
    assert m.negative_sampling == "unigram", (
        f"expected negative_sampling='unigram', got {m.negative_sampling!r}"
    )
