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
