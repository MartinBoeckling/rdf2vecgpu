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


def _build_sharding_plan(
    model: nn.Module, world_size: int, device: torch.device,
) -> "ShardingPlan":  # noqa: F821 (lazy import — see body)
    """Defer to torchrec's planner; no constraints in v1.

    The planner will pick row-wise sharding automatically for tables
    that are large enough to benefit from it. Smaller tables may end up
    table-wise on a single rank, which is fine — the user's reason for
    choosing this backend is the *large* tables.
    """
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.types import ModuleSharder
    from torchrec.distributed.embedding import EmbeddingCollectionSharder

    topology = Topology(
        world_size=world_size,
        compute_device=device.type,  # "cuda"
    )
    planner = EmbeddingShardingPlanner(topology=topology)
    sharders: list[ModuleSharder] = [EmbeddingCollectionSharder()]
    return planner.plan(model, sharders=sharders)


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


def extract_embeddings_to_dask_cudf(model, vocab_df):
    """Return embeddings as a ``dask_cudf.DataFrame`` (one partition per rank).

    Reads the *local* shard of the in-embedding table on each rank;
    joins it shard-locally with the vocab slice for that token-id
    range; wraps each rank's cudf.DataFrame as a single dask partition.
    No rank-0 gather.

    The output schema is::

        token_id  int64
        entity    string
        "0"       float32
        "1"       float32
        ...
        str(dim-1) float32
    """
    import cudf
    import cupy
    import dask_cudf
    import torch.distributed as dist

    # Locate the in-embedding table parameter on this rank.
    in_param = _get_in_table_local_tensor(model)         # cuda tensor (local_rows, dim)
    local_rows, dim = in_param.shape

    # Compute the (start, end) row range this rank owns.
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    start, end = _local_row_range(model.vocab_size, world_size, rank)
    assert end - start == local_rows, (
        f"local rows {local_rows} ≠ shard size {end - start} on rank {rank}"
    )

    # Move to a cudf.DataFrame with dim columns labeled "0".."dim-1".
    cupy_arr = cupy.asarray(in_param.detach())
    shard_df = cudf.DataFrame({str(d): cupy_arr[:, d] for d in range(dim)})
    shard_df["token_id"] = cudf.Series(range(start, end), dtype="int64")

    # Shard-local join with the vocab slice for [start, end).
    vocab_slice = vocab_df[(vocab_df["token_id"] >= start) & (vocab_df["token_id"] < end)]
    shard_df = shard_df.merge(vocab_slice, on="token_id", how="left")

    # Wrap each rank's cudf as one dask partition.
    return dask_cudf.from_cudf(shard_df, npartitions=1)


def _local_row_range(vocab_size: int, world_size: int, rank: int) -> tuple[int, int]:
    """Even split with the remainder absorbed by the last rank."""
    base = vocab_size // world_size
    start = rank * base
    end = vocab_size if rank == world_size - 1 else start + base
    return start, end


def _get_in_table_local_tensor(model) -> torch.Tensor:
    """Return the rank-local tensor for the in-embedding table.

    Two paths:
    1. Post-DMP: torchrec wraps the parameter as a ShardedTensor /
       DTensor exposing ``.local_tensor()`` — return that.
    2. Pre-DMP or world_size=1 unwrapped: walk
       ``in_collection.embeddings`` directly and return the
       ``nn.Embedding.weight.data``.

    The two branches are deliberately separate. If torchrec ships a
    state_dict that contains the table key but where the value lacks
    ``local_tensor`` (e.g. a future API change), we fall through to the
    direct access — never return a full unsliced tensor as if it were a
    shard.
    """
    # Path 1: post-DMP sharded value
    try:
        sd = model.state_dict()
    except (AttributeError, RuntimeError):
        sd = {}
    for key, val in sd.items():
        if ShardedSkipGramModel.IN_TABLE_NAME in key and "weight" in key:
            if hasattr(val, "local_tensor"):
                return val.local_tensor()
            # Key matched but no .local_tensor — this is the pre-DMP
            # case where the state_dict value IS the full nn.Embedding
            # weight. Fall through to path 2 rather than returning the
            # full tensor as a "shard".
            break

    # Path 2: pre-DMP direct access to nn.Embedding weights
    for table in model.in_collection.embeddings.values():
        return table.weight.data
    raise RuntimeError("could not locate in-embedding table on this rank")


def _save_checkpoint(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    keep_last: int,
    vocab_path: str,
    config_snapshot: dict,
) -> None:
    """Write a DCP shard set + rank-0 metadata.pt + update ``latest`` symlink.

    Crash safety: shards are written to ``step_<N>.tmp/`` first; only
    after all ranks complete the DCP write does rank 0 rename it to
    ``step_<N>/`` and update the ``latest`` symlink. A crash mid-write
    leaves a ``.tmp`` directory that the next run's pruning ignores.

    Pruning policy: keep the most recent ``keep_last`` ``step_*``
    directories (NOT counting ``*.tmp``); everything else is removed.

    Layout (see spec §4.4):
        {checkpoint_dir}/
          step_<global_step:08d>/    # DCP shards, written by all ranks
          metadata.pt                # rank-0 only
          latest -> step_*
    """
    import os
    import shutil
    from pathlib import Path

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    assert keep_last >= 1, f"keep_last must be >= 1, got {keep_last!r}"

    ckpt_root = Path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    final_name = f"step_{global_step:08d}"
    tmp_dir = ckpt_root / f"{final_name}.tmp"
    final_dir = ckpt_root / final_name

    # If a stale tmp from a prior crash exists, rank 0 wipes it.
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0 and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if dist.is_initialized():
        dist.barrier()

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # DCP save into the .tmp directory: all ranks participate.
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dcp.save(
        state_dict=state_dict,
        storage_writer=dcp.FileSystemWriter(str(tmp_dir)),
    )

    # Barrier so every rank has finished writing into .tmp before
    # rank 0 promotes it to the final name.
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        # Promote .tmp -> final via rename. If final_dir already exists
        # (rerunning the same global_step), remove it first.
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)

        # metadata.pt
        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "vocab_path": vocab_path,
            "config_snapshot": config_snapshot,
        }
        torch.save(meta, ckpt_root / "metadata.pt")

        # Atomic symlink update via tmp + os.replace.
        tmp_link = ckpt_root / "latest.tmp"
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        os.symlink(final_dir.name, tmp_link)        # relative target
        os.replace(tmp_link, ckpt_root / "latest")

    # Barrier so all ranks see the promoted final_dir before pruning.
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        # Prune: keep the most recent `keep_last` *finalised* step dirs.
        # We deliberately exclude ``*.tmp`` so a concurrent save in
        # flight is never pruned.
        step_dirs = sorted(
            [
                p for p in ckpt_root.iterdir()
                if p.is_dir() and p.name.startswith("step_") and not p.name.endswith(".tmp")
            ],
            key=lambda p: p.name,
        )
        for stale in step_dirs[:-keep_last]:
            shutil.rmtree(stale)

    if dist.is_initialized():
        dist.barrier()


def _maybe_resume_checkpoint(
    model,
    optimizer,
    checkpoint_dir: str | None,
    overwrite: bool,
    current_vocab_path: str,
) -> tuple[int, int]:
    """Resume from ``{checkpoint_dir}/latest`` if it exists, else fresh.

    Returns ``(start_epoch, start_step)``. See spec §4.4 for the full
    decision table.

    RNG caveat: ``cuda_rng_state_all`` covers all *local* CUDA devices on
    the saving node only, not cross-node state. Resuming on a different
    ``world_size`` (or a different host) therefore carries small RNG
    reproducibility caveats — the model/optimizer weights are restored
    exactly, but the negative-sampling stream may diverge from a no-crash
    baseline. Accepted for v1.
    """
    import shutil
    from pathlib import Path

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    if checkpoint_dir is None:
        return 0, 0

    ckpt_root = Path(checkpoint_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0

    if overwrite and ckpt_root.exists():
        if rank == 0:
            shutil.rmtree(ckpt_root)
            ckpt_root.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return 0, 0

    if not ckpt_root.exists():
        if rank == 0:
            ckpt_root.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return 0, 0

    latest = ckpt_root / "latest"
    if not latest.exists():
        # Empty dir, or a never-completed save — start fresh in place.
        return 0, 0

    step_dir = latest.resolve()
    if not step_dir.is_dir():
        raise RuntimeError(
            f"Checkpoint at {step_dir} is missing or corrupt. Pass "
            "checkpoint_overwrite=True to start fresh, or restore the "
            "checkpoint contents."
        )

    # Vocab-mismatch guard.
    meta = torch.load(ckpt_root / "metadata.pt", weights_only=False)
    saved_vocab_path = meta["vocab_path"]
    if saved_vocab_path != current_vocab_path:
        raise RuntimeError(
            f"Cannot resume: saved checkpoint was built against vocab at "
            f"{saved_vocab_path!r}, but current run resolved vocab to "
            f"{current_vocab_path!r}. Vocab IDs would no longer match. Pass "
            "checkpoint_overwrite=True to start fresh."
        )

    # DCP load into the already-instantiated model/optimizer.
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dcp.load(
        state_dict=state_dict,
        storage_reader=dcp.FileSystemReader(str(step_dir)),
    )
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])

    # Restore RNG state on the loading rank only — DCP doesn't reshard
    # RNG and we accept this small reproducibility caveat (documented).
    if rank == 0:
        torch.set_rng_state(meta["rng_state"])
        torch.cuda.set_rng_state_all(meta["cuda_rng_state_all"])

    if dist.is_initialized():
        dist.barrier()

    return int(meta["epoch"]), int(meta["global_step"])


def train_torchrec_word2vec(model, edge_df, walk_vertices):
    raise NotImplementedError(
        "train_torchrec_word2vec is a stub — implemented in Task 11."
    )
