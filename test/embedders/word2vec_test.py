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
