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


torchrec = pytest.importorskip("torchrec")
torch_lib = pytest.importorskip("torch")


@pytest.mark.gpu
def test_sharded_skipgram_model_constructs_in_degenerate_world_size_one(tmp_path):
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
        # Both in and out tables present.
        assert any("in_embeddings" in k for k in sd), sd.keys()
        assert any("out_embeddings" in k for k in sd), sd.keys()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
