"""Tests for the torchrec sharded-embedding backend.

Layer 1 tests (this section) run without torchrec/fbgemm installed and
without a GPU — they exercise pre-flight config validation only.
Layers 2+ require either the `gpu` or `multigpu` marker.
"""
from __future__ import annotations

import pytest

from rdf2vecgpu.config import RDF2VecConfig
from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec


# ───────────────────────────── Layer 1 ────────────────────────────────


def test_torchrec_backend_with_cbow_raises_not_implemented():
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="cbow", multi_gpu=True)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(NotImplementedError, match="skipgram"):
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

    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True)
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
    vocab_size = len(model._torchrec_vocab_df)
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
def test_torchrec_vs_pytorch_topk_overlap(torchrun_pg):
    """Quality canary: top-K NN overlap between a single-GPU pytorch run
    and the sharded torchrec run on the same corpus.

    Not bit-exact (PARTIAL_ROWWISE_ADAM ≠ SparseAdam, and the two backends
    expand pairs differently); the threshold is deliberately loose — it
    catches gross breakage (broken sharding → ~random overlap, wrong loss
    sign → anti-correlation), not fine quality differences.
    """
    import os
    import numpy as np
    import pandas as pd

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    base = _shared_base("topk")
    walks_path, vocab_path = _prepare_corpus_fs(
        base,
        make_triples=lambda: _connected_string_triples(
            5_000, seed=42, n_entities=150, n_predicates=10
        ),
        walk_depth=3, walk_number=10,
    )

    common = dict(
        embedding_model="skipgram", walk_strategy="random",
        walk_depth=3, walk_number=10, vector_size=16, epochs=3,
        learning_rate=0.025, negative_samples=5, random_state=0,
    )

    cfg_tr = RDF2VecConfig(backend="torchrec", multi_gpu=True, **common)
    m_tr = GPU_RDF2Vec(config=cfg_tr)
    m_tr.fit_from_walks(walks_path, vocab_path)
    emb_tr = _gather_full_embeddings_fs(m_tr, base)

    def topk_neighbours(df: pd.DataFrame, k: int = 5) -> dict[int, set]:
        df = df.sort_values("token_id")
        m = df.drop(columns=["token_id", "entity"]).to_numpy()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        cos = (m @ m.T) / (norms * norms.T + 1e-12)
        np.fill_diagonal(cos, -np.inf)
        ids = df["token_id"].to_numpy()
        return {
            int(ids[i]): set(int(t) for t in ids[np.argsort(-cos[i])[:k]])
            for i in range(len(ids))
        }

    # The pytorch baseline runs on EVERY rank (single-GPU on each rank's own
    # device — redundant but cheap on this tiny graph) so the assertion is
    # symmetric across ranks: no rank-0-only branch that could desync a
    # subsequent collective. triples.parquet was written by rank 0 before
    # the corpus.ready marker, so every rank can read it here.
    triples_path = os.path.join(base, "triples.parquet")
    cfg_pt = RDF2VecConfig(backend="pytorch", multi_gpu=False, **common)
    m_pt = GPU_RDF2Vec(config=cfg_pt)
    edges_pt = m_pt.load_data(triples_path)
    emb_pt = m_pt.fit_transform(edge_df=edges_pt).to_pandas()

    nn_pt = topk_neighbours(emb_pt)
    nn_tr = topk_neighbours(emb_tr)
    overlap = np.mean([
        len(nn_pt[t] & nn_tr.get(t, set())) / 5 for t in nn_pt
    ])
    assert overlap > 0.5, f"top-5 NN overlap too low: {overlap:.2f}"


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

    np.testing.assert_array_equal(
        emb1.drop(columns=["entity"]).to_numpy(),
        emb2.drop(columns=["entity"]).to_numpy(),
    )


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
