"""Tests for the torchrec sharded-embedding backend.

Layer 1 tests (this section) run without torchrec/fbgemm installed and
without a GPU — they exercise pre-flight config validation only.
Layers 2+ require either the `gpu` or `multigpu` marker.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from rdf2vecgpu.config import RDF2VecConfig
from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec


# ───────────────────────────── Layer 1 ────────────────────────────────


def test_progress_meter_due_respects_cadence():
    from rdf2vecgpu.embedders.torchrec_word2vec import _ProgressMeter
    m = _ProgressMeter(every_n_steps=200, enabled=True)
    assert not m.due(199)
    assert m.due(200)
    assert m.due(400)
    assert not m.due(401)
    # every_n_steps=0 disables the cadence (and avoids modulo-by-zero)
    assert not _ProgressMeter(every_n_steps=0, enabled=True).due(0)
    # `enabled` toggles the print, not the cadence — `due` is cadence-only
    assert _ProgressMeter(every_n_steps=200, enabled=False).due(200)


def test_progress_meter_format_reports_rate_and_fields():
    from rdf2vecgpu.embedders.torchrec_word2vec import _ProgressMeter
    m = _ProgressMeter(every_n_steps=200, enabled=True)
    first = m.format(epoch=2, step=200, loss=1.25, now=100.0)
    assert "epoch 2" in first and "step 200" in first and "loss 1.2500" in first
    assert "0.0 steps/s" in first                       # no prior sample → rate 0
    second = m.format(epoch=2, step=400, loss=0.5, now=110.0)
    assert "20.0 steps/s" in second                     # 200 steps / 10 s


def test_torchrec_backend_with_cbow_raises_not_implemented():
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="cbow", multi_gpu=True)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(NotImplementedError, match="skipgram"):
        # Paths unused — the pre-flight check fires before any disk read.
        model.fit_from_walks("walks", "vocab.parquet")


def test_torchrec_backend_with_sgd_rejected():
    # The fused optimizer-in-backward applies the update inside the FBGEMM kernel,
    # where gradient clipping is a no-op (the gradient_clipping/max_gradient kwargs
    # are silently dropped). So unclipped SGD diverges on the SUM skip-gram loss for
    # non-trivial vocabularies. Reject sgd at preflight — even with grad_clip_value>0,
    # which used to be (wrongly) accepted — and point to the stable rowwiseadagrad path.
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True,
                        optimizer="sgd", grad_clip_value=0.1)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(ValueError, match="rowwiseadagrad"):
        # Paths unused — the pre-flight check fires before any disk read.
        model.fit_from_walks("walks", "vocab.parquet")


def test_torchrec_backend_without_multi_gpu_raises_value_error():
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=False)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(ValueError, match=r"sharded embedding path is multi-process"):
        model.fit_from_walks("walks", "vocab.parquet")


def test_torchrec_backend_without_distributed_init_raises_runtime_error():
    """No torchrun → torch.distributed.is_initialized() is False."""
    import torch.distributed as dist

    assert not dist.is_initialized(), "this test must run outside a distributed launch"

    # Use a supported optimizer so the distributed-init guard fires — the default
    # "sgd" would be rejected earlier in the preflight (its fused clip is a no-op).
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True,
                        optimizer="rowwiseadagrad")
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(RuntimeError, match="torchrun"):
        model.fit_from_walks("walks", "vocab.parquet")


def test_torchrec_fit_with_edge_df_raises_decoupling_guidance():
    """fit(edge_df) is unsupported for torchrec — data-prep can't run inside
    the torchrun job. The error must point to the prepare_corpus →
    fit_from_walks flow.
    """
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(RuntimeError, match=r"prepare_corpus.*fit_from_walks|does not support fit"):
        model.fit(edge_df=None)


def test_torchrec_transform_without_prior_fit_raises():
    """Calling transform() on backend='torchrec' before fit() (or after
    process group teardown) must raise — the sharded model is rank-local
    and can't be reconstructed.
    """
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True)
    model = GPU_RDF2Vec(config=cfg)
    # No fit() call → model._torchrec_model attribute missing.
    with pytest.raises(RuntimeError, match="rank-local"):
        model.transform()


# ───────────────────────────── Layer 2 ────────────────────────────────


pytest.importorskip("torchrec")
pytest.importorskip("torch")


@pytest.mark.gpu
def test_sharded_skipgram_model_constructs_in_degenerate_world_size_one():
    """world_size=1 → planner degenerates to table-wise on rank 0.

    This test verifies the module wiring without needing a real
    multi-process launch.
    """
    import os
    import torch
    import torch.distributed as dist

    # Single-process distributed init on localhost (gloo backend is fine —
    # we don't need NCCL for module construction).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import ShardedSkipGramModel

        model = ShardedSkipGramModel(vocab_size=1024, embedding_dim=16)
        sd = model.state_dict()
        in_key = f"in_collection.embeddings.{ShardedSkipGramModel.IN_TABLE_NAME}.weight"
        out_key = f"out_collection.embeddings.{ShardedSkipGramModel.OUT_TABLE_NAME}.weight"
        assert in_key in sd, list(sd.keys())
        assert out_key in sd, list(sd.keys())
        assert sd[in_key].shape == (1024, 16)
        assert sd[out_key].shape == (1024, 16)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_sample_negatives_uniform_shape_and_range():
    import torch
    from rdf2vecgpu.embedders.torchrec_word2vec import _sample_negatives

    out = _sample_negatives(
        n_rows=64,
        n_neg=5,
        vocab_size=1024,
        device=torch.device("cpu"),
        strategy="uniform",
        token_counts=None,
    )
    assert out.shape == (64, 5)
    assert out.dtype == torch.long
    assert int(out.min()) >= 0
    assert int(out.max()) < 1024


def test_sample_negatives_unigram_uses_token_counts():
    import torch
    from rdf2vecgpu.embedders.torchrec_word2vec import _sample_negatives

    # All probability mass on a single token (idx=7).
    counts = torch.zeros(1024)
    counts[7] = 1.0
    out = _sample_negatives(
        n_rows=128,
        n_neg=3,
        vocab_size=1024,
        device=torch.device("cpu"),
        strategy="unigram",
        token_counts=counts,
    )
    # With all mass on idx=7, every drawn negative must be 7.
    assert torch.all(out == 7)


def test_unigram_cdf_is_float64_normalised_and_skips_zero_weight():
    import torch
    from rdf2vecgpu.embedders.torchrec_word2vec import _unigram_cdf

    # f^0.75 of [0, 1, 16, 81, 0] = [0, 1, 8, 27, 0] -> p = [0, 1, 8, 27, 0] / 36
    counts = torch.tensor([0, 1, 16, 81, 0])
    cdf = _unigram_cdf(counts, torch.device("cpu"))
    assert cdf.dtype == torch.float64
    assert torch.allclose(cdf, torch.tensor([0, 1, 9, 36, 36], dtype=torch.float64) / 36)
    with pytest.raises(ValueError, match="count > 0"):
        _unigram_cdf(torch.zeros(4), torch.device("cpu"))


def test_sample_negatives_unigram_cdf_path_matches_counts_path_and_distribution():
    import torch
    from rdf2vecgpu.embedders.torchrec_word2vec import _sample_negatives, _unigram_cdf

    counts = torch.tensor([0, 1, 16, 81, 0])
    cdf = _unigram_cdf(counts, torch.device("cpu"))
    kw = dict(n_rows=20_000, n_neg=5, vocab_size=5, device=torch.device("cpu"), strategy="unigram")

    torch.manual_seed(0)
    via_cdf = _sample_negatives(**kw, cdf=cdf)
    torch.manual_seed(0)
    via_counts = _sample_negatives(**kw, token_counts=counts)
    assert via_cdf.shape == (20_000, 5) and via_cdf.dtype == torch.long
    assert torch.equal(via_cdf, via_counts)          # same RNG stream -> identical draws

    freq = torch.bincount(via_cdf.flatten(), minlength=5).double() / via_cdf.numel()
    assert freq[0] == 0 and freq[4] == 0             # zero-count tokens are never negatives
    assert torch.allclose(freq[1:4], torch.tensor([1, 8, 27], dtype=torch.float64) / 36, atol=0.01)


def test_sample_negatives_unigram_without_token_counts_raises():
    import torch
    from rdf2vecgpu.embedders.torchrec_word2vec import _sample_negatives

    with pytest.raises(ValueError, match="token_counts"):
        _sample_negatives(
            n_rows=8, n_neg=2, vocab_size=64,
            device=torch.device("cpu"),
            strategy="unigram", token_counts=None,
        )


def test_sample_negatives_unknown_strategy_raises_with_value():
    import torch
    from rdf2vecgpu.embedders.torchrec_word2vec import _sample_negatives

    with pytest.raises(ValueError, match="bogus"):
        _sample_negatives(
            n_rows=8, n_neg=2, vocab_size=64,
            device=torch.device("cpu"),
            strategy="bogus", token_counts=None,
        )


@pytest.mark.gpu
def test_sharded_skipgram_forward_returns_finite_loss():
    import os
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _sample_negatives,
        )

        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=1024, embedding_dim=16).to(device)

        batch = 32
        center = torch.randint(0, 1024, (batch,), device=device)
        context = torch.randint(0, 1024, (batch,), device=device)
        negative = _sample_negatives(
            n_rows=batch, n_neg=5, vocab_size=1024,
            device=device, strategy="uniform", token_counts=None,
        )

        loss = model(center, context, negative)
        assert loss.dim() == 0
        assert torch.isfinite(loss).item()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_sharded_skipgram_forward_loss_closed_form_with_zero_embeddings():
    """Zero both embedding tables → every score is 0 → loss is the
    closed-form ``batch * (1 + n_neg) * log(2)`` (with ``.sum()`` reduction).

    This pins the reduction to ``torch.sum``: ``.mean()`` would shrink
    the result by the batch size and miss the closed form.
    """
    import math
    import os
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import ShardedSkipGramModel

        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=128, embedding_dim=8).to(device)

        # Zero both tables in-place.
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()

        batch = 16
        n_neg = 4
        center = torch.randint(0, 128, (batch,), device=device)
        context = torch.randint(0, 128, (batch,), device=device)
        negative = torch.randint(0, 128, (batch, n_neg), device=device)

        loss = model(center, context, negative)
        expected = batch * (1 + n_neg) * math.log(2)
        assert torch.isclose(loss, torch.tensor(expected, device=device), atol=1e-4), (
            f"got {loss.item():.6f}, expected {expected:.6f}"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_build_sharding_plan_returns_plan_with_both_tables():
    import os
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _build_sharding_plan,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=1024, embedding_dim=16).to(device)
        plan = _build_sharding_plan(model, world_size=1, device=device)
        # Walk the plan structure: plan.plan is Dict[module_path,
        # Dict[table_name, ParameterSharding]]. We don't assert a
        # specific sharding type (the planner picks based on table
        # size and topology, and that's intentional in v1), but every
        # table must appear with a non-empty sharding_type — that's
        # the contract DistributedModelParallel needs.
        all_table_names: set[str] = set()
        for module_plan in plan.plan.values():
            for table_name, param_sharding in module_plan.items():
                assert param_sharding.sharding_type, (
                    f"empty sharding_type for {table_name!r}"
                )
                all_table_names.add(table_name)
        assert ShardedSkipGramModel.IN_TABLE_NAME in all_table_names, all_table_names
        assert ShardedSkipGramModel.OUT_TABLE_NAME in all_table_names, all_table_names
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_extract_embeddings_to_dask_cudf_schema_and_row_count():
    # NOTE: This exercises the pre-DMP / world_size=1 path of
    # _get_in_table_local_tensor. The DMP-wrapped path (with
    # local_tensor()) is covered by the multi-GPU integration tests
    # in Task 13.
    import os
    import torch
    import torch.distributed as dist
    import cudf

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            extract_embeddings_to_dask_cudf,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=128, embedding_dim=8).to(device)

        # Build a tiny vocab_df mapping token_id -> entity (str).
        vocab_df = cudf.DataFrame({
            "token_id": cudf.Series(range(128), dtype="int64"),
            "entity":   cudf.Series([f"e{i}" for i in range(128)]),
        })

        out = extract_embeddings_to_dask_cudf(model, vocab_df)
        out_pdf = out.compute().to_pandas()

        assert len(out_pdf) == 128
        # Required columns: token_id, entity, and one column per embedding dim
        assert "token_id" in out_pdf.columns
        assert "entity" in out_pdf.columns
        for d in range(8):
            assert str(d) in out_pdf.columns, f"missing dim column {d}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_save_checkpoint_writes_step_dir_metadata_and_latest_symlink(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            model, opt,
            epoch=1, global_step=500,
            checkpoint_dir=str(ckpt_dir),
            keep_last=3,
            vocab_path="/data/vocab.parquet",
            config_snapshot={"backend": "torchrec"},
        )
        # step directory created
        assert (ckpt_dir / "step_00000500").is_dir()
        # metadata.pt at root, with the right keys
        assert (ckpt_dir / "metadata.pt").is_file()
        meta = torch.load(ckpt_dir / "metadata.pt", weights_only=False)
        assert meta["epoch"] == 1
        assert meta["global_step"] == 500
        assert meta["vocab_path"] == "/data/vocab.parquet"
        # latest symlink resolves to the step directory
        latest = ckpt_dir / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == (ckpt_dir / "step_00000500").resolve()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_save_checkpoint_prunes_old_step_dirs(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"

        for step in (100, 200, 300, 400, 500):
            _save_checkpoint(
                model, opt, epoch=0, global_step=step,
                checkpoint_dir=str(ckpt_dir), keep_last=3,
                vocab_path="/v", config_snapshot={},
            )
        kept = sorted(p.name for p in ckpt_dir.iterdir() if p.name.startswith("step_"))
        assert kept == ["step_00000300", "step_00000400", "step_00000500"]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_save_checkpoint_recovers_from_stale_latest_tmp(tmp_path):
    """A prior crash may leave a stale ``latest.tmp`` symlink. The next
    save must clean it up rather than crash. Also verifies that two
    successive saves atomically swap ``latest``.
    """
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        # Simulate stale latest.tmp from a prior crash.
        stale = ckpt_dir / "latest.tmp"
        os.symlink("step_doesnotexist", stale)
        assert stale.is_symlink()

        # First save should clean it up + write a new latest.
        _save_checkpoint(
            model, opt, epoch=0, global_step=100,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/v", config_snapshot={},
        )
        latest_after_1 = ckpt_dir / "latest"
        assert latest_after_1.is_symlink()
        assert latest_after_1.resolve().name == "step_00000100"
        assert not stale.exists()  # latest.tmp consumed

        # Second save should atomically swap the symlink to point at the new step.
        _save_checkpoint(
            model, opt, epoch=0, global_step=200,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/v", config_snapshot={},
        )
        latest_after_2 = ckpt_dir / "latest"
        assert latest_after_2.is_symlink()
        assert latest_after_2.resolve().name == "step_00000200"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_round_trip_restores_weights_and_step(tmp_path):
    import os, copy, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")

        # Save
        m1 = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            m1, opt1, epoch=3, global_step=12345,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/v", config_snapshot={},
        )
        snapshot = copy.deepcopy(m1.state_dict())

        # Resume into a fresh model.
        m2 = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
        start_epoch, start_step = _maybe_resume_checkpoint(
            m2, opt2,
            checkpoint_dir=str(ckpt_dir), overwrite=False, current_vocab_path="/v",
        )
        assert start_epoch == 3
        assert start_step == 12345
        for k, v in snapshot.items():
            torch.testing.assert_close(m2.state_dict()[k], v)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_missing_dir_returns_fresh(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")
        m = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        start_epoch, start_step = _maybe_resume_checkpoint(
            m, opt,
            checkpoint_dir=str(tmp_path / "does_not_exist"),
            overwrite=False, current_vocab_path="/v",
        )
        assert (start_epoch, start_step) == (0, 0)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_with_vocab_mismatch_raises(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")
        m = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            m, opt, epoch=0, global_step=1,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/vocab_A.parquet", config_snapshot={},
        )
        with pytest.raises(RuntimeError, match="vocab"):
            _maybe_resume_checkpoint(
                m, opt,
                checkpoint_dir=str(ckpt_dir),
                overwrite=False,
                current_vocab_path="/vocab_B.parquet",
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_overwrite_wipes_dir_and_returns_fresh(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")
        m = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            m, opt, epoch=2, global_step=999,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/v", config_snapshot={},
        )
        start_epoch, start_step = _maybe_resume_checkpoint(
            m, opt,
            checkpoint_dir=str(ckpt_dir), overwrite=True, current_vocab_path="/v",
        )
        assert (start_epoch, start_step) == (0, 0)
        # No step_* dirs should remain
        assert not any(p.name.startswith("step_") for p in ckpt_dir.iterdir())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_with_corrupted_latest_raises(tmp_path):
    """``latest`` symlink points to a step dir that was deleted out-of-band.

    The resume path must raise ``RuntimeError`` with the documented
    message rather than silently start fresh — silent fallback could
    mask a backup-restore problem.
    """
    import os, shutil, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")
        m = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"

        _save_checkpoint(
            m, opt, epoch=0, global_step=1,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/v", config_snapshot={},
        )
        # Manually delete the step dir, leaving the `latest` symlink dangling.
        step_dir = (ckpt_dir / "latest").resolve()
        shutil.rmtree(step_dir)

        with pytest.raises(RuntimeError, match="missing or corrupt"):
            _maybe_resume_checkpoint(
                m, opt,
                checkpoint_dir=str(ckpt_dir),
                overwrite=False, current_vocab_path="/v",
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ───────────────────────────── Layer 3 ────────────────────────────────


@pytest.fixture(scope="session")
def torchrun_pg():
    """Initialize torch.distributed for the session under torchrun.

    Plain ``pytest`` invocations don't set WORLD_SIZE → skip.
    Under ``torchrun --nproc_per_node=N``, RANK/WORLD_SIZE/MASTER_*
    are set by the launcher; we initialize NCCL once per session here
    and tear down at the end.
    """
    import os
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("requires torchrun --nproc_per_node>=2")

    if not dist.is_initialized():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    yield

    if dist.is_initialized():
        dist.destroy_process_group()


# ── decoupled-flow corpus helpers ─────────────────────────────────────
#
# The torchrec backend trains from a PRE-GENERATED corpus (walks + vocab
# on disk); it does NOT run dask-cudf / cuGraph data-prep inside the
# torchrun job (those paradigms can't share a process — see
# docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md
# §4). These helpers prepare the corpus rank-0-only with single-GPU
# cudf/cuGraph (no Dask cluster needed), then hand every rank a shared
# on-disk path.
#
# Coordination is via the SHARED FILESYSTEM, never via NCCL object
# collectives. ``broadcast_object_list`` / ``all_gather_object`` on the
# NCCL group deadlock when only rank 0 has done prior CUDA work (cuDF /
# RMM prep): the pickled byte-tensor's device placement mismatches across
# ranks, so the collective never pairs. Single-node multi-GPU is assumed
# (all ranks see the same local filesystem).


def _shared_base(tag: str) -> str:
    """A directory whose path is identical on every rank — derived from the
    torchrun run id, with NO collective.

    ``TORCHELASTIC_RUN_ID`` is set identically across ranks within a run
    and uniquely per run, so the path is both shared and collision-free
    across reruns (no stale-marker races).
    """
    import os
    import tempfile

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    base = os.path.join(tempfile.gettempdir(), "rdf2vec_mgpu", run_id, tag)
    os.makedirs(base, exist_ok=True)
    return base


def _connected_string_triples(n, seed, n_entities, n_predicates):
    """A connected RDF graph: subjects and objects are drawn from one
    shared entity pool so random walks can chain past depth 1. Real RDF
    entities are strings, which is also what the vocab tokeniser expects.
    """
    import cudf
    import numpy as np

    rng = np.random.default_rng(seed)
    subj = rng.integers(0, n_entities, size=n)
    obj = rng.integers(0, n_entities, size=n)
    pred = rng.integers(0, n_predicates, size=n)
    return cudf.DataFrame({
        "subject":   [f"e{i}" for i in subj],
        "predicate": [f"p{i}" for i in pred],
        "object":    [f"e{i}" for i in obj],
    })


def _prepare_corpus_fs(base, make_triples, walk_depth, walk_number):
    """rank-0-only single-GPU corpus prep; returns ``(walks_path, vocab_path)``.

    rank 0 runs the single-GPU data path (read → vocab → graph → walks)
    and writes a ``corpus.ready`` marker; peer ranks poll the filesystem
    for that marker — NO NCCL collective (see module note on the object-
    broadcast deadlock). The prep config uses ``backend='pytorch'`` +
    ``multi_gpu=False`` purely to drive the single-GPU path; the resulting
    on-disk corpus is backend-agnostic and the torchrec ranks consume it
    via ``fit_from_walks``.
    """
    import os
    import time
    import torch.distributed as dist

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    corpus_dir = os.path.join(base, "corpus")
    ready = os.path.join(base, "corpus.ready")
    walks_path = os.path.join(corpus_dir, "walks")
    vocab_path = os.path.join(corpus_dir, "vocab.parquet")

    if dist.get_rank() == 0:
        triples = make_triples()
        triples_path = os.path.join(base, "triples.parquet")
        triples.to_parquet(triples_path)

        prep_cfg = RDF2VecConfig(
            backend="pytorch", multi_gpu=False,
            embedding_model="skipgram", walk_strategy="random",
            walk_depth=walk_depth, walk_number=walk_number,
            random_state=0,
        )
        m = GPU_RDF2Vec(config=prep_cfg)
        edges = m.load_data(triples_path)
        m.prepare_corpus(edges, corpus_dir)
        with open(ready, "w") as fh:
            fh.write("ok")
    else:
        deadline = time.monotonic() + 300
        while not os.path.exists(ready):
            if time.monotonic() > deadline:
                raise TimeoutError("rank-0 corpus prep did not complete in time")
            time.sleep(0.5)
    return walks_path, vocab_path


def _gather_full_embeddings_fs(model, base):
    """Gather the per-rank embedding shards into one pandas frame via the
    shared filesystem (each rank writes its shard parquet; rank 0 reads the
    union). Used by tests that need the full table (nearest-neighbour
    comparisons). Production code persists shards to a partitioned parquet
    dataset the same way — no all-gather of the full table into one process.
    """
    import glob
    import os
    import cudf
    import torch.distributed as dist

    emb_dir = os.path.join(base, "emb")
    os.makedirs(emb_dir, exist_ok=True)
    local = model.transform().compute()
    local.to_parquet(os.path.join(emb_dir, f"part_{dist.get_rank()}.parquet"), index=False)
    dist.barrier()
    parts = sorted(glob.glob(os.path.join(emb_dir, "part_*.parquet")))
    return cudf.read_parquet(parts).to_pandas().sort_values("token_id")


@pytest.mark.multigpu
def test_torchrec_end_to_end_smoke_world_size_2(torchrun_pg):
    """Decoupled end-to-end: prepare a corpus (rank-0 single-GPU), then
    train sharded across ranks and verify the shards tile the full vocab.

    Run via: torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu \
        -k test_torchrec_end_to_end_smoke
    """
    import torch.distributed as dist

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec
    from rdf2vecgpu.embedders.torchrec_word2vec import _local_row_range

    base = _shared_base("smoke")
    walks_path, vocab_path = _prepare_corpus_fs(
        base,
        make_triples=lambda: _connected_string_triples(
            10_000, seed=0, n_entities=300, n_predicates=20
        ),
        walk_depth=3, walk_number=5,
    )

    cfg = RDF2VecConfig(
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=5,
        vector_size=8, epochs=1,
        learning_rate=0.025, negative_samples=5, optimizer="sparseadam",
    )
    model = GPU_RDF2Vec(config=cfg)
    model.fit_from_walks(walks_path, vocab_path)

    # transform() returns this rank's shard as a single dask partition
    # (no cross-rank gather — that is the design, to avoid materialising
    # the full table anywhere).
    emb = model.transform()
    assert emb.npartitions == 1
    local = emb.compute().to_pandas()

    # Each rank's shard must be exactly its computed row range. Since
    # _local_row_range tiles [0, vocab_size) across ranks by construction,
    # every rank matching its own range proves the shards tile the vocab
    # exactly once — verified per-rank, no cross-rank collective needed.
    vocab_size = model._torchrec_vocab_size
    start, end = _local_row_range(
        vocab_size, dist.get_world_size(), dist.get_rank()
    )
    local_ids = sorted(int(x) for x in local["token_id"].to_numpy())
    assert local_ids == list(range(start, end)), (
        f"rank {dist.get_rank()} shard {local_ids[:3]}..{local_ids[-3:]} "
        f"≠ expected range [{start}, {end})"
    )


# ───────────────────────────── Layer 4 ────────────────────────────────


@pytest.mark.multigpu
def test_torchrec_recovers_planted_community_structure(torchrun_pg):
    """Quality canary: the sharded torchrec run must produce FINITE,
    non-degenerate embeddings that recover planted community structure.

    This replaces an earlier cross-backend top-K NN overlap check, which was
    ill-posed: the pytorch baseline and torchrec build independent
    vocabularies, and on a structureless random graph neither separates
    entities, so the overlap sat at chance even for sane runs (the pytorch
    reference itself recovered no structure there). Instead we plant three
    communities (dense intra-community edges, sparse inter), so random walks —
    and thus skip-gram co-occurrence — stay mostly within a community, and
    assert torchrec recovers them. This catches the failure modes actually
    observed in practice:

      * divergence to NaN/Inf (unclipped SGD on the SUM skip-gram loss), and
      * embedding collapse / broken sharding (intra ≈ inter cosine).

    It uses optimizer="rowwiseadagrad" (EXACT_ROWWISE_ADAGRAD), torchrec's stable
    SGD-family path: per-row Adagrad is gradient-following (no momentum) with a
    per-row LR decay, so it stays finite AND recovers structure. Plain sgd is
    rejected at preflight (the fused optimizer-in-backward can't clip → unclipped
    sgd diverges); Adam/sparseadam is stable but collapses on this tiny vocab.
    This canary guards exactly that distinction.
    """
    import numpy as np

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    def clustered_triples(n_comm=3, per=200, intra=30_000, inter=600, seed=42):
        import cudf
        rng = np.random.default_rng(seed)

        def name(c, i):
            return f"e{c}_{i}"

        s, o, p = [], [], []
        for _ in range(intra):
            c = int(rng.integers(0, n_comm))
            s.append(name(c, int(rng.integers(0, per))))
            o.append(name(c, int(rng.integers(0, per))))
            p.append("intra")
        for _ in range(inter):
            s.append(name(int(rng.integers(0, n_comm)), int(rng.integers(0, per))))
            o.append(name(int(rng.integers(0, n_comm)), int(rng.integers(0, per))))
            p.append("inter")
        return cudf.DataFrame({"subject": s, "predicate": p, "object": o})

    base = _shared_base("community_big")
    walks_path, vocab_path = _prepare_corpus_fs(
        base, make_triples=clustered_triples, walk_depth=4, walk_number=20,
    )

    cfg = RDF2VecConfig(
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=4, walk_number=20,
        vector_size=32, epochs=40, learning_rate=0.05, negative_samples=5,
        optimizer="rowwiseadagrad",  # per-row Adagrad: gradient-following + per-row
                                     # LR decay → stable AND recovers structure (sgd's
                                     # in-kernel clip is a no-op → diverges; Adam
                                     # collapses on this tiny vocab; LARS unusable in
                                     # FBGEMM 1.4 — no get_optimizer_state)
        random_state=0,
    )
    m = GPU_RDF2Vec(config=cfg)
    m.fit_from_walks(walks_path, vocab_path)
    emb = _gather_full_embeddings_fs(m, base)

    vec_cols = [c for c in emb.columns if c not in ("token_id", "entity")]
    M = emb[vec_cols].to_numpy(dtype="float64")

    # (1) Finiteness — guards against divergence. RowWiseAdagrad's per-row LR
    #     decay keeps it stable on the SUM loss; plain sgd would diverge (its
    #     in-kernel clip is a no-op), which is why sgd is rejected for torchrec.
    assert np.isfinite(M).all(), (
        f"non-finite embeddings: "
        f"{int((~np.isfinite(M)).any(axis=1).sum())}/{len(M)} rows have NaN/Inf"
    )

    # (2) Structure recovery — same-community entities must be markedly closer
    #     (cosine) than different-community ones. Collapse or broken sharding
    #     drives the gap to ~0; a healthy run is ~0.14 on this graph.
    df = emb[emb["entity"].str.startswith("e")].copy()
    comm = df["entity"].str.split("_").str[0].to_numpy()
    V = df[vec_cols].to_numpy(dtype="float64")
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    cos = V @ V.T
    same = comm[:, None] == comm[None, :]
    eye = np.eye(len(comm), dtype=bool)
    intra = cos[same & ~eye].mean()
    inter = cos[~same & ~eye].mean()
    assert intra - inter > 0.05, (
        f"no community structure recovered: intra_cos={intra:.3f} "
        f"inter_cos={inter:.3f} gap={intra - inter:+.3f} (expect > 0.05)"
    )


# ───────────────────────────── Layer 5 ────────────────────────────────


@pytest.mark.multigpu
def test_save_then_reload_preserves_embeddings_bit_equal(torchrun_pg):
    """Round trip: train with checkpointing, then re-launch with the same
    checkpoint_dir + epochs → auto-resume at the final epoch (zero further
    training) and re-extract. Per-rank shards must be bit-equal, proving
    DCP saved and reloaded the sharded weights without drift.
    """
    import os
    import numpy as np

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    base = _shared_base("ckpt_roundtrip")
    walks_path, vocab_path = _prepare_corpus_fs(
        base,
        make_triples=lambda: _connected_string_triples(
            2_000, seed=7, n_entities=80, n_predicates=5
        ),
        walk_depth=3, walk_number=5,
    )

    ckpt = os.path.join(base, "ckpt")
    cfg = RDF2VecConfig(
        # torchrec rejects the default optimizer='sgd' (fused clip is a no-op)
        optimizer="rowwiseadagrad",
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=5,
        vector_size=8, epochs=2,
        learning_rate=0.025, negative_samples=5,
        checkpoint_dir=ckpt, checkpoint_interval_minutes=120,
    )

    m1 = GPU_RDF2Vec(config=cfg)
    m1.fit_from_walks(walks_path, vocab_path)
    emb1 = m1.transform().compute().to_pandas().sort_values("token_id")

    # Restart: same checkpoint_dir + epochs=2 → resume at epoch 2, no
    # further training → identical weights.
    m2 = GPU_RDF2Vec(config=cfg)
    m2.fit_from_walks(walks_path, vocab_path)
    emb2 = m2.transform().compute().to_pandas().sort_values("token_id")

    a1 = emb1.drop(columns=["entity"]).to_numpy()
    a2 = emb2.drop(columns=["entity"]).to_numpy()
    # Guard NaN-blindness: assert_array_equal treats NaN==NaN as equal, so an
    # all-NaN divergence would round-trip as "bit-equal" and pass silently.
    assert np.isfinite(a1).all(), "reloaded embeddings (run 1) contain NaN/Inf"
    np.testing.assert_array_equal(a1, a2)


@pytest.mark.multigpu
def test_overwrite_starts_fresh(torchrun_pg):
    """checkpoint_overwrite=True wipes a populated checkpoint_dir and
    trains from scratch.
    """
    import os

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    base = _shared_base("ckpt_overwrite")
    walks_path, vocab_path = _prepare_corpus_fs(
        base,
        make_triples=lambda: _connected_string_triples(
            1_000, seed=11, n_entities=50, n_predicates=5
        ),
        walk_depth=3, walk_number=5,
    )

    ckpt = os.path.join(base, "ckpt")
    cfg = RDF2VecConfig(
        # torchrec rejects the default optimizer='sgd' (fused clip is a no-op)
        optimizer="rowwiseadagrad",
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=5,
        vector_size=8, epochs=1, checkpoint_dir=ckpt,
    )
    m1 = GPU_RDF2Vec(config=cfg)
    m1.fit_from_walks(walks_path, vocab_path)
    # Shared checkpoint_dir → every rank sees the same files. Assert on all
    # ranks (symmetric) rather than rank-0-only, so a failure fails cleanly
    # everywhere instead of hanging peers.
    assert os.path.exists(os.path.join(ckpt, "latest"))

    cfg_ow = cfg.model_copy(update={"checkpoint_overwrite": True})
    m2 = GPU_RDF2Vec(config=cfg_ow)
    m2.fit_from_walks(walks_path, vocab_path)
    step_dirs = [p for p in os.listdir(ckpt) if p.startswith("step_")]
    assert len(step_dirs) >= 1


@pytest.mark.multigpu
def test_keep_last_prunes_excess_step_dirs(torchrun_pg):
    """Epoch-boundary saves over 5 epochs with keep_last=2 leave exactly
    two step directories.
    """
    import os

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    base = _shared_base("ckpt_keeplast")
    walks_path, vocab_path = _prepare_corpus_fs(
        base,
        make_triples=lambda: _connected_string_triples(
            5_000, seed=13, n_entities=100, n_predicates=5
        ),
        walk_depth=3, walk_number=20,
    )

    ckpt = os.path.join(base, "ckpt")
    cfg = RDF2VecConfig(
        # torchrec rejects the default optimizer='sgd' (fused clip is a no-op)
        optimizer="rowwiseadagrad",
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=20,
        vector_size=8, epochs=5,
        checkpoint_dir=ckpt, checkpoint_interval_minutes=120,
        checkpoint_keep_last=2,
    )
    model = GPU_RDF2Vec(config=cfg)
    model.fit_from_walks(walks_path, vocab_path)
    # Shared checkpoint_dir → assert on all ranks (symmetric).
    step_dirs = sorted(p for p in os.listdir(ckpt) if p.startswith("step_"))
    assert len(step_dirs) == 2, step_dirs


# ───────────── Sharding-plan layout guard (added 2026-09-05) ─────────────
# Left to its cost model, the planner placed dbpedia2016's 6.3M×100 tables
# TABLE-WISE on 2×A100 (in_embeddings on rank 1, out_embeddings on rank 0):
# the DCP checkpoint held one full-table chunk per file, and at export time
# rank 0 found no local shard of the in-table and crashed with
# "'DistributedModelParallel' object has no attribute 'in_collection'".
# Every consumer in the module assumes each rank owns one row block.


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 8])
def test_build_sharding_plan_pins_row_wise_with_ceil_block_ranges(world_size):
    import os
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    from rdf2vecgpu.embedders.torchrec_word2vec import (
        ShardedSkipGramModel,
        _build_sharding_plan,
        _local_row_range,
    )
    device = torch.device("cuda")
    # 100_003 rows: divisible by neither 2 nor 8, so the ceil-block remainder
    # lands on the last rank — the case a naive floor split gets wrong.
    vocab = 100_003
    model = ShardedSkipGramModel(vocab_size=vocab, embedding_dim=16).to(device)
    plan = _build_sharding_plan(model, world_size=world_size, device=device)

    seen: set[str] = set()
    for module_plan in plan.plan.values():
        for table_name, ps in module_plan.items():
            seen.add(table_name)
            assert ps.sharding_type == "row_wise", (table_name, ps.sharding_type)
            shards = ps.sharding_spec.shards
            assert len(shards) == world_size, (table_name, len(shards))
            for rank, shard in enumerate(shards):
                start, end = _local_row_range(vocab, world_size, rank)
                assert shard.shard_offsets[0] == start, (table_name, rank, shard.shard_offsets)
                assert shard.shard_sizes[0] == end - start, (table_name, rank, shard.shard_sizes)
    assert seen == {
        ShardedSkipGramModel.IN_TABLE_NAME, ShardedSkipGramModel.OUT_TABLE_NAME,
    }, seen


@pytest.mark.gpu
def test_get_in_table_local_shard_raises_when_rank_holds_no_shard():
    """A DMP-wrapped model whose in-table has no local shard on this rank
    (table-wise placement elsewhere) must fail loudly — never fall through
    to the unwrapped-module path, which does not exist on the wrapper."""
    from rdf2vecgpu.embedders.torchrec_word2vec import _get_in_table_local_shard

    class _NoLocalShards:               # ShardedTensor API, empty on this rank
        def local_shards(self):
            return []

    class _FakeDMP:
        module = object()               # DistributedModelParallel.module

        def state_dict(self):
            return {"in_collection.embeddings.in_embeddings.weight": _NoLocalShards()}

    with pytest.raises(RuntimeError, match="row-wise"):
        _get_in_table_local_shard(_FakeDMP())


# ───────────── Meta-device construction guard (added 2026-09-06) ─────────────
# The first full-Wikidata run (job 14593196, 8× H200) was OOM-killed at 1.5 TB
# of host RAM before step 0: every rank built the full 2 × 393M × 100 fp32
# tables (314 GB) on the CPU and only then moved them to its GPU. The trainer
# now builds on the meta device and lets DMP allocate just the local shard.


@pytest.mark.gpu
def test_sharded_skipgram_meta_construction_materialises_only_under_dmp():
    import os
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.optim import apply_optimizer_in_backward
    from rdf2vecgpu.embedders.torchrec_word2vec import (
        ShardedSkipGramModel,
        _build_sharding_plan,
        _fused_optimizer_from_config,
        _get_in_table_local_shard,
    )
    device = torch.device("cuda")
    vocab, dim = 4096, 16
    model = ShardedSkipGramModel(vocab_size=vocab, embedding_dim=dim, device=torch.device("meta"))
    params = list(model.parameters())
    assert params and all(p.device.type == "meta" for p in params), [p.device for p in params]

    # Same order as the trainer: fused optimizer on the meta params, then DMP.
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True,
                        optimizer="rowwiseadagrad")
    fused_cls, fused_kwargs = _fused_optimizer_from_config(cfg)
    apply_optimizer_in_backward(fused_cls, model.parameters(), fused_kwargs)
    plan = _build_sharding_plan(model, world_size=1, device=device)
    dmp = DistributedModelParallel(module=model, device=device, plan=plan, init_data_parallel=False)

    shard, start = _get_in_table_local_shard(dmp)
    assert shard.device.type == "cuda" and tuple(shard.shape) == (vocab, dim) and start == 0
    assert torch.isfinite(shard).all() and float(shard.abs().sum()) > 0.0   # initialised on device
    assert all(p.device.type != "meta" for p in dmp.parameters())

    g = torch.Generator(device="cpu").manual_seed(0)
    center = torch.randint(0, vocab, (64,), generator=g)
    context = torch.randint(0, vocab, (64,), generator=g)
    negative = torch.randint(0, vocab, (64, 5), generator=g, device="cpu").to(device)
    loss = dmp(center.to(device), context.to(device), negative)
    assert torch.isfinite(loss), loss


@pytest.mark.gpu
def test_build_walk_pair_loader_delivers_configured_pair_budget(tmp_path):
    """End-to-end loader: parquet walks with cuGraph sentinels → batches whose
    pair count tracks ``pair_batch_size`` (not a row count assuming full walks),
    all pairs delivered exactly once, prefetch thread transparent."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rdf2vecgpu.embedders.word2vec_loader import _walk_pair_counts

    d = tmp_path / "walks"; d.mkdir()
    rng = np.random.default_rng(1)
    n, depth_p1 = 5000, 5
    v = rng.integers(0, 300, size=(n, depth_p1)).astype(np.int64)
    p = rng.integers(300, 320, size=(n, depth_p1 - 1)).astype(np.float64)
    first = rng.integers(1, depth_p1 + 1, size=n)
    for r in range(n):
        v[r, first[r]:] = -1
        p[r, first[r] - 1:] = -1
    for k in range(2):                                    # two files: batches span files
        sl = slice(k * n // 2, (k + 1) * n // 2)
        pq.write_table(pa.table({
            "vertices": pa.array(list(v[sl]), type=pa.list_(pa.int64())),
            "predicates": pa.array(list(p[sl]), type=pa.list_(pa.float64())),
        }), d / f"part.{k}.parquet")

    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True,
                        optimizer="rowwiseadagrad", walk_depth=4, window_size=5,
                        pair_batch_size=4000, random_state=0)
    m = GPU_RDF2Vec(config=cfg)
    batches = list(m._build_walk_pair_loader(str(d), None))
    sizes = [int(c.numel()) for c, _ in batches]
    total = int(_walk_pair_counts(v.astype(np.int32), 5).sum())
    assert sum(sizes) == total and total > 0
    assert all(4000 <= s < 4060 for s in sizes[:-1]) and 0 < sizes[-1] < 4060, sizes[:5]
    # batches overshoot by < one walk (60 pairs), so the count sits between the
    # two bounds rather than at ceil(total / budget)
    assert total // 4060 <= len(sizes) <= total // 4000 + 1, (len(sizes), total)
    import torch
    c0, x0 = batches[0]
    assert c0.dtype == torch.int64 and x0.dtype == torch.int64 and not ((c0 == -1) | (x0 == -1)).any()


# ───────────────────── streaming export (iter_local_embedding_chunks) ─────────────────────


def _write_vocab_parquet(path, entities, *, row_group_size=None, order=None):
    """``vocab.parquet`` in the prepare_corpus layout (token_id, entity); ``order``
    permutes / drops rows to build unsorted or gapped files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = list(range(len(entities)))
    if order is not None:
        ids = [ids[i] for i in order]
        entities = [entities[i] for i in order]
    pq.write_table(
        pa.table({"token_id": pa.array(ids, pa.int64()), "entity": pa.array(entities, pa.string())}),
        path, row_group_size=row_group_size,
    )


def test_read_vocab_slice_returns_contiguous_block_from_row_groups(tmp_path):
    from rdf2vecgpu.embedders.torchrec_word2vec import _read_vocab_slice

    ents = [f"http://x/e{i}" for i in range(40)]
    path = str(tmp_path / "vocab.parquet")
    _write_vocab_parquet(path, ents, row_group_size=7)          # 6 row groups
    assert _read_vocab_slice(path, 10, 25).to_pylist() == ents[10:25]
    assert _read_vocab_slice(path, 0, 40).to_pylist() == ents
    assert _read_vocab_slice(path, 40, 40).to_pylist() == []


def test_read_vocab_slice_sorts_unordered_rows_and_rejects_gaps(tmp_path):
    import numpy as np
    from rdf2vecgpu.embedders.torchrec_word2vec import _read_vocab_slice

    ents = [f"e{i}" for i in range(30)]
    shuffled = str(tmp_path / "shuffled.parquet")
    _write_vocab_parquet(shuffled, ents, row_group_size=8,
                         order=np.random.default_rng(0).permutation(30))
    got = _read_vocab_slice(shuffled, 5, 20)
    assert got.to_pylist() == ents[5:20]
    # int64 offsets, whatever the file holds: the sort materialises the whole
    # block as one array, and a full-Wikidata rank block is > 2 GiB of IRIs.
    import pyarrow as pa
    assert got.type == pa.large_string()
    assert _read_vocab_slice(shuffled, 30, 30).type == pa.large_string()
    gap = str(tmp_path / "gap.parquet")
    _write_vocab_parquet(gap, ents, row_group_size=8, order=[i for i in range(30) if i != 12])
    with pytest.raises(RuntimeError, match="contiguous"):
        _read_vocab_slice(gap, 5, 20)


@pytest.mark.skipif(
    not os.environ.get("RDF2VEC_BIG_TESTS"),
    reason="writes a ~2.3 GB parquet and holds ~5 GB; set RDF2VEC_BIG_TESTS=1 (or =<dir>)",
)
def test_read_vocab_slice_survives_a_block_over_2gib_of_strings(tmp_path):
    """Regression for 2026-09-14 job 14697728: a full-Wikidata rank block is
    49M IRIs ≈ 2.3 GB of entity bytes, over the 2 GiB cap of a ``string``
    array's int32 offsets, and the Wikidata vocab's row groups are not in
    token-id order — so the reader's sort/take materialised the block as one
    ``string`` array and died with ``ArrowInvalid: offset overflow``. Two
    out-of-order row groups of 1.15 GB each reproduce both conditions."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from rdf2vecgpu.embedders.torchrec_word2vec import _read_vocab_slice

    big = os.environ["RDF2VEC_BIG_TESTS"]
    base = tmp_path if big == "1" else Path(big) / f"vocab_slice_big_{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    path = str(base / "vocab.parquet")
    n, width = 1_150_000, 2_000                      # 2.3e9 bytes > 2**31
    half = n // 2
    # second half first, so the reader has to sort — and every value unique,
    # so a wrong ``take`` would be caught by the spot checks below.
    ids = pa.array(list(range(half, n)) + list(range(0, half)), pa.int64())
    ents = pc.utf8_lpad(pc.cast(ids, pa.large_string()), width, "x")
    # store_schema=False: read back as plain ``string`` (what prepare_corpus /
    # make_vocab_parquet.py write), not as the large_string we built it from.
    pq.write_table(pa.table({"token_id": ids, "entity": ents}), path,
                   row_group_size=half, store_schema=False)
    del ids, ents
    pf = pq.ParquetFile(path)
    assert pf.metadata.num_row_groups == 2
    assert pf.schema_arrow.field("entity").type == pa.string()

    # the failure mode the reader must avoid: one int32-offset array of the block
    t = pa.concat_tables([pf.read_row_group(i, columns=["token_id", "entity"]) for i in (0, 1)])
    with pytest.raises(pa.ArrowInvalid, match="offset overflow"):
        t.take(pc.sort_indices(t, sort_keys=[("token_id", "ascending")]))
    del t

    got = _read_vocab_slice(path, 0, n)
    assert got.type == pa.large_string() and len(got) == n
    for i in (0, 1, half - 1, half, n - 1):
        assert got[i].as_py() == str(i).rjust(width, "x")
    del got
    got = _read_vocab_slice(path, half - 3, half + 3)
    assert [s.as_py()[-8:] for s in got] == [str(i).rjust(8, "x") for i in range(half - 3, half + 3)]
    try:
        shutil.rmtree(base)
    except OSError:
        pass


def test_iter_local_embedding_chunks_streams_unsharded_table_in_order(tmp_path):
    import numpy as np
    from rdf2vecgpu.embedders.torchrec_word2vec import (
        ShardedSkipGramModel,
        iter_local_embedding_chunks,
    )

    vocab = 23
    ents = [f"http://x/e{i}" for i in range(vocab)]
    path = str(tmp_path / "vocab.parquet")
    _write_vocab_parquet(path, ents, row_group_size=5)
    model = ShardedSkipGramModel(vocab_size=vocab, embedding_dim=4)
    table = next(iter(model.in_collection.embeddings.values())).weight.data

    chunks = list(iter_local_embedding_chunks(model, path, chunk_rows=10))
    assert [c[2].shape[0] for c in chunks] == [10, 10, 3]
    assert np.concatenate([c[0] for c in chunks]).tolist() == list(range(vocab))
    assert sum((c[1].to_pylist() for c in chunks), []) == ents
    vec = np.concatenate([c[2] for c in chunks])
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, table.detach().cpu().numpy())
    with pytest.raises(ValueError):
        next(iter_local_embedding_chunks(model, path, chunk_rows=0))


@pytest.mark.multigpu
def test_iter_local_embeddings_tiles_vocab_and_matches_transform(torchrun_pg):
    """The streaming export path must yield exactly this rank's row block —
    the same ids and vectors as transform()'s cudf shard — carrying the
    entity strings of vocab.parquet, so per-rank TSV shards concatenate to
    the whole vocabulary exactly once."""
    import numpy as np
    import pyarrow.parquet as pq
    import torch.distributed as dist

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec
    from rdf2vecgpu.embedders.torchrec_word2vec import _local_row_range

    base = _shared_base("stream_export")
    walks_path, vocab_path = _prepare_corpus_fs(
        base,
        make_triples=lambda: _connected_string_triples(
            8_000, seed=3, n_entities=250, n_predicates=10
        ),
        walk_depth=3, walk_number=5,
    )
    cfg = RDF2VecConfig(
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=5,
        vector_size=8, epochs=1,
        learning_rate=0.025, negative_samples=5, optimizer="sparseadam",
    )
    model = GPU_RDF2Vec(config=cfg)
    model.fit_from_walks(walks_path, vocab_path)

    chunks = list(model.iter_local_embeddings(chunk_rows=37))
    ids = np.concatenate([c[0] for c in chunks])
    ents = sum((c[1].to_pylist() for c in chunks), [])
    vec = np.concatenate([c[2] for c in chunks])

    start, end = _local_row_range(
        model._torchrec_vocab_size, dist.get_world_size(), dist.get_rank()
    )
    assert ids.tolist() == list(range(start, end))
    assert len(chunks) == -(-(end - start) // 37) and all(len(c[2]) <= 37 for c in chunks)

    vocab = pq.read_table(vocab_path).to_pandas().set_index("token_id")["entity"]
    assert ents == [vocab.loc[i] for i in range(start, end)]

    ref = model.transform().compute().to_pandas().sort_values("token_id")
    assert ref["token_id"].tolist() == ids.tolist()
    np.testing.assert_array_equal(
        vec, ref[[str(d) for d in range(8)]].to_numpy(dtype=np.float32)
    )


@pytest.mark.gpu
def test_step_profiler_attributes_segments_and_gpu_time_only_when_enabled():
    import torch
    from rdf2vecgpu.embedders.torchrec_word2vec import _StepProfiler

    dev = torch.device("cuda")
    prof = _StepProfiler(True, dev, rank=0, skip=1, profile_window=None)
    x = torch.randn(1024, 1024, device=dev)
    for _ in range(3):
        prof.drain()
        for seg in ("next", "join", "negatives"):
            prof.begin(seg); prof.end(seg)
        prof.gpu_start()
        prof.begin("forward"); y = x @ x; prof.end("forward")
        prof.begin("backward"); y = y @ x; prof.end("backward")
        prof.gpu_end()
        prof.begin("log"); prof.end("log")
        prof.end_step()
    prof.drain()
    s = prof.summary()
    assert s["n"] == 2                                   # step 0 skipped as warm-up
    assert set(s["mean_ms"]) == set(_StepProfiler.SEGMENTS)
    assert s["gpu_ms"] > 0 and s["total_ms"] >= s["mean_ms"]["forward"]
    assert len(prof.gpu_ms) == 2

    off = _StepProfiler(False, dev, rank=0)
    off.drain(); off.begin("next"); off.end("next"); off.gpu_start(); off.gpu_end(); off.end_step()
    assert off.summary() is None and off.step == 0 and off.host["next"] == []


@pytest.mark.gpu
def test_sharded_skipgram_logits_match_batched_gemm_reference():
    """The fused multiply-reduce logits must reproduce the batched-GEMM
    formula they replaced (loss AND gradients), so the kernel swap is
    invisible to training."""
    import os
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import ShardedSkipGramModel

        device = torch.device("cuda")
        torch.manual_seed(0)
        model = ShardedSkipGramModel(vocab_size=96, embedding_dim=12).to(device)
        w_in = next(iter(model.in_collection.embeddings.values())).weight
        w_out = next(iter(model.out_collection.embeddings.values())).weight

        batch, n_neg = 41, 5
        center = torch.randint(0, 96, (batch,), device=device)
        context = torch.randint(0, 96, (batch,), device=device)
        negative = torch.randint(0, 96, (batch, n_neg), device=device)

        loss = model(center, context, negative)
        loss.backward()
        got_in, got_out = w_in.grad.clone(), w_out.grad.clone()

        w_in.grad = None
        w_out.grad = None
        c, p, n = w_in[center], w_out[context], w_out[negative]      # (B,d) (B,d) (B,neg,d)
        pos = (c * p).sum(-1)
        neg = torch.bmm(n, c.unsqueeze(-1)).squeeze(-1)
        ref = (-F.logsigmoid(pos) - F.logsigmoid(-neg).sum(-1)).sum()
        ref.backward()

        torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(got_in, w_in.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(got_out, w_out.grad, rtol=1e-4, atol=1e-5)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
