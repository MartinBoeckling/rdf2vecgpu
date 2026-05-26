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
        # No edge_df needed — the pre-flight check fires before walk gen.
        model.fit(edge_df=None)


def test_torchrec_backend_without_multi_gpu_raises_value_error():
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=False)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(ValueError, match=r"sharded embedding path is multi-process"):
        model.fit(edge_df=None)


def test_torchrec_backend_without_distributed_init_raises_runtime_error():
    """No torchrun → torch.distributed.is_initialized() is False."""
    import torch.distributed as dist

    assert not dist.is_initialized(), "this test must run outside a distributed launch"

    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(RuntimeError, match="torchrun"):
        model.fit(edge_df=None)


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
