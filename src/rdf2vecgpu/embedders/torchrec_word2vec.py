"""Torchrec-backed sharded Word2Vec training.

See docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md
for the design. v1 covers skip-gram only.

All torchrec / fbgemm imports happen *inside* function/class bodies so
this module can be imported without those extras installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    pass


class ShardedSkipGramModel(nn.Module):
    """Skip-gram with two row-wise sharded embedding tables (in + out).

    Sharding is applied later by wrapping this module in
    ``torchrec.distributed.DistributedModelParallel`` with a plan from
    :func:`_build_sharding_plan`. Prior to wrapping, the table parameters
    live on whichever device the caller placed them.
    """

    IN_TABLE_NAME = "in_embeddings"
    OUT_TABLE_NAME = "out_embeddings"

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        from torchrec import EmbeddingCollection, EmbeddingConfig

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        in_cfg = EmbeddingConfig(
            name=self.IN_TABLE_NAME,
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            feature_names=["center"],
        )
        out_cfg = EmbeddingConfig(
            name=self.OUT_TABLE_NAME,
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            feature_names=["context_or_negative"],
        )
        self.in_collection = EmbeddingCollection(tables=[in_cfg])
        self.out_collection = EmbeddingCollection(tables=[out_cfg])


def train_torchrec_word2vec(model, edge_df, walk_vertices):
    raise NotImplementedError(
        "train_torchrec_word2vec is a stub — implemented in Task 11."
    )
