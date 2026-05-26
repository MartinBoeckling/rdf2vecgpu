"""Torchrec-backed sharded Word2Vec training.

See docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md
for the design. v1 covers skip-gram only.

All torchrec / fbgemm imports happen *inside* function/class bodies so
this module can be imported without those extras installed.
"""
from __future__ import annotations

import torch
from torch import nn


def _sample_negatives(
    n_rows: int,
    n_neg: int,
    vocab_size: int,
    device: torch.device,
    strategy: str = "unigram",
    token_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample ``(n_rows, n_neg)`` negative token ids.

    Mirrors the f^0.75 unigram sampler used by the existing pytorch
    backend (see ``embedders/word2vec.py::_sample_negatives``). Returns
    a flat ``long`` tensor; the caller is responsible for routing
    through the sharded output embedding.
    """
    if strategy == "uniform":
        return torch.randint(
            low=0, high=vocab_size, size=(n_rows, n_neg),
            dtype=torch.long, device=device,
        )

    if strategy == "unigram":
        if token_counts is None:
            raise ValueError("strategy='unigram' requires token_counts")
        weights = (
            token_counts.to(device=device, dtype=torch.float32)
            .clamp(min=0)
            .pow(0.75)
        )
        # torch.multinomial replaces=True is fine for negative sampling
        # (rare duplicates within a row are negligible at typical
        # neg_samples ≤ 25 against vocab_size in the millions).
        return torch.multinomial(
            weights, num_samples=n_rows * n_neg, replacement=True,
        ).view(n_rows, n_neg)

    raise ValueError(f"unknown negative-sampling strategy: {strategy!r}")


class ShardedSkipGramModel(nn.Module):
    """Skip-gram with two row-wise sharded embedding tables (in + out).

    Sharding is applied later by wrapping this module in
    ``torchrec.distributed.DistributedModelParallel`` with a plan from
    `_build_sharding_plan` (Task 7). Prior to wrapping, the table parameters
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

    def forward(
        self,
        center: torch.Tensor,   # (B,) long
        context: torch.Tensor,  # (B,) long
        negative: torch.Tensor, # (B, neg) long
    ) -> torch.Tensor:
        """Skip-gram negative-sampling loss.

        Mirrors :class:`rdf2vecgpu.embedders.word2vec.Word2VecSkipgram.forward`:
        positive logit = <in[center], out[context]>;
        negative logits = <in[center], out[neg]>;
        loss = -log σ(pos) - Σ log σ(-neg_k).
        """
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        batch_size = center.size(0)
        n_neg = negative.size(1)

        # Build KJTs for the two collections. Each row contributes one id
        # for the in-table (center) and (1 + n_neg) ids for the out-table
        # (one positive context + n_neg negatives).
        in_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["center"],
            values=center,
            lengths=torch.ones(batch_size, dtype=torch.long, device=center.device),
        )
        out_values = torch.cat([context.unsqueeze(1), negative], dim=1).reshape(-1)
        out_lengths = torch.full(
            (batch_size,), 1 + n_neg, dtype=torch.long, device=center.device,
        )
        out_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["context_or_negative"],
            values=out_values,
            lengths=out_lengths,
        )

        in_dict = self.in_collection(in_kjt)
        out_dict = self.out_collection(out_kjt)

        center_embed = in_dict["center"].values().view(batch_size, self.embedding_dim)
        out_embed = (
            out_dict["context_or_negative"].values().view(batch_size, 1 + n_neg, self.embedding_dim)
        )

        pos_embed = out_embed[:, 0, :]                # (B, dim)
        neg_embed = out_embed[:, 1:, :]               # (B, neg, dim)

        pos_score = (center_embed * pos_embed).sum(dim=-1)                            # (B,)
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(-1)).squeeze(-1)      # (B, neg)

        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=-1)
        # SUM over the batch of pairs, not mean. See the comment on
        # Word2VecSkipgram.forward in embedders/word2vec.py for the
        # full reasoning: at ~1M pairs/step `.mean()` divides every
        # row's gradient by ~1M and embeddings never move. With `sum`,
        # `lr` is the per-pair learning rate, matching classic word2vec.
        return torch.sum(pos_loss + neg_loss)


def train_torchrec_word2vec(model, edge_df, walk_vertices):
    raise NotImplementedError(
        "train_torchrec_word2vec is a stub — implemented in Task 11."
    )
