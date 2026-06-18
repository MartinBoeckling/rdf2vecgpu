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


def _env_progress_enabled() -> bool:
    """Convenience override so the SLURM harness can flip the progress print
    without touching config plumbing: RDF2VEC_TORCHREC_PROGRESS=1."""
    import os
    return os.environ.get("RDF2VEC_TORCHREC_PROGRESS", "").strip().lower() not in (
        "", "0", "false", "no",
    )


class _ProgressMeter:
    """Decides when to emit a torchrec training-progress point and formats it.

    Host-only (no GPU / process group) so it is unit-testable. ``due(step)`` is
    cadence-only and gates BOTH tracker logging and the optional stdout print;
    ``enabled`` toggles only the flushed print. ``format`` tracks the previous
    emission to report steps/sec.
    """

    def __init__(self, *, every_n_steps: int, enabled: bool):
        self.every = int(every_n_steps)
        self.enabled = bool(enabled)
        self._last_step = 0
        self._last_time = None

    def due(self, step: int) -> bool:
        return self.every > 0 and step % self.every == 0

    def format(self, *, epoch: int, step: int, loss: float, now: float) -> str:
        if self._last_time is None:
            sps = 0.0
        else:
            dt = now - self._last_time
            sps = (step - self._last_step) / dt if dt > 0 else 0.0
        self._last_step, self._last_time = step, now
        return (
            f"[torchrec] epoch {epoch} | step {step:,} | "
            f"loss {loss:.4f} | {sps:.1f} steps/s"
        )


def _build_sharding_plan(
    model: nn.Module, world_size: int, device: torch.device, pg=None,
) -> "ShardingPlan":  # noqa: F821 (lazy import — see body)
    """Defer to torchrec's planner; no constraints in v1.

    The planner will pick row-wise sharding automatically for tables
    that are large enough to benefit from it. Smaller tables may end up
    table-wise on a single rank, which is fine — the user's reason for
    choosing this backend is the *large* tables.

    Uses ``collective_plan`` (not the rank-local ``plan``) whenever a
    process group is available: the planner's proposal search is NOT
    guaranteed to be bit-identical across ranks, and divergent plans give
    each rank a different shard layout. ``DistributedModelParallel`` then
    desyncs on the per-shard ``all_gather_object`` in
    ``_init_from_local_shards`` and hangs forever. ``collective_plan``
    computes the plan on rank 0 and broadcasts it, so every rank shards
    identically. With no process group (single-process unit tests) we
    fall back to the rank-local ``plan``.
    """
    import torch.distributed as dist
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.types import ModuleSharder
    from torchrec.distributed.embedding import EmbeddingCollectionSharder

    topology = Topology(
        world_size=world_size,
        compute_device=device.type,  # "cuda"
    )
    planner = EmbeddingShardingPlanner(topology=topology)
    sharders: list[ModuleSharder] = [EmbeddingCollectionSharder()]

    if pg is None and dist.is_available() and dist.is_initialized():
        pg = dist.group.WORLD
    if pg is not None:
        return planner.collective_plan(model, sharders, pg)
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

    # Locate this rank's shard of the in-embedding table AND its authoritative
    # global row offset, read straight from torchrec's shard metadata. Do NOT
    # recompute the offset from vocab_size/world_size: torchrec's row-wise
    # block sizing (ceil block, remainder on the last rank) disagrees with a
    # naive split for any vocab_size not divisible by world_size, which would
    # mislabel every token id on at least one rank. ``model`` here is also the
    # DMP wrapper, which does not expose ``vocab_size`` — another reason to
    # trust the shard metadata over the module.
    in_param, start = _get_in_table_local_shard(model)   # cuda tensor (local_rows, dim)
    local_rows, dim = in_param.shape
    end = start + local_rows

    # Move to a cudf.DataFrame with dim columns labeled "0".."dim-1". Cast to
    # float32 (the documented output schema) and force contiguity: under
    # precision="medium" the shard may be bf16 — which cupy has no dtype for —
    # and a sharded local tensor is not guaranteed C-contiguous.
    cupy_arr = cupy.asarray(in_param.detach().to(torch.float32).contiguous())
    shard_df = cudf.DataFrame({str(d): cupy_arr[:, d] for d in range(dim)})
    shard_df["token_id"] = cudf.Series(range(start, end), dtype="int64")

    # Shard-local join with the vocab slice for [start, end).
    vocab_slice = vocab_df[(vocab_df["token_id"] >= start) & (vocab_df["token_id"] < end)]
    shard_df = shard_df.merge(vocab_slice, on="token_id", how="left")

    # Wrap each rank's cudf as one dask partition.
    return dask_cudf.from_cudf(shard_df, npartitions=1)


def _local_row_range(vocab_size: int, world_size: int, rank: int) -> tuple[int, int]:
    """Row range ``[start, end)`` that ``rank`` owns under torchrec row-wise
    sharding.

    MUST mirror torchrec's block sizing: ``block = ceil(vocab_size /
    world_size)``; earlier ranks each own a full ``block`` and the remainder
    lands on the last populated rank, which owns *fewer*. This is the OPPOSITE
    of a naive floor split (remainder on the last rank); the two disagree
    whenever ``vocab_size % world_size != 0`` — verified against
    ``ShardedTensor.local_shards()[0].metadata.shard_offsets`` (V=321, W=2 →
    rank0 owns [0,161), rank1 owns [161,321)).

    Production extraction reads the offset straight from the shard metadata
    (:func:`_get_in_table_local_shard`); this helper exists for callers (the
    tests) that need to predict the tiling without a live sharded model.
    """
    import math

    block = math.ceil(vocab_size / world_size)
    start = min(rank * block, vocab_size)
    end = min(start + block, vocab_size)
    return start, end


def _get_in_table_local_shard(model) -> "tuple[torch.Tensor, int]":
    """Return ``(local_tensor, start_row)`` for the in-embedding table on
    this rank.

    ``start_row`` is the AUTHORITATIVE global row offset of this rank's
    shard, read from torchrec's ``ShardedTensor`` metadata — never
    recomputed (see :func:`_local_row_range` for why a recomputed offset
    drifts from torchrec's actual ceil-block split).

    Two paths:
    1. Post-DMP: the state_dict value is a ``ShardedTensor`` whose single
       local shard carries both the tensor and its ``shard_offsets`` —
       return ``(shard.tensor, shard_offsets[0])``.
    2. Pre-DMP or world_size=1 unwrapped: walk ``in_collection.embeddings``
       directly and return ``(weight.data, 0)`` — the whole table lives on
       this rank starting at row 0.

    The two branches are deliberately separate: if torchrec ever ships a
    state_dict whose value lacks the sharded API, we fall through to direct
    access rather than mislabel a full tensor as a shard at offset 0.
    """
    # Path 1: post-DMP sharded value
    try:
        sd = model.state_dict()
    except (AttributeError, RuntimeError):
        sd = {}
    for key, val in sd.items():
        if ShardedSkipGramModel.IN_TABLE_NAME in key and "weight" in key:
            if hasattr(val, "local_shards"):
                shards = val.local_shards()
                if shards:
                    shard = shards[0]
                    return shard.tensor, int(shard.metadata.shard_offsets[0])
            # Key matched but no ShardedTensor shards — this is the pre-DMP
            # case where the state_dict value IS the full nn.Embedding
            # weight. Fall through to path 2 rather than returning the full
            # tensor as a "shard".
            break

    # Path 2: pre-DMP direct access to nn.Embedding weights (start row 0)
    for table in model.in_collection.embeddings.values():
        return table.weight.data, 0
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
    # ``exists()`` follows symlinks, so a dangling ``latest`` symlink would
    # silently fall through to "start fresh". Check ``is_symlink()`` first
    # to surface the corrupt-checkpoint case documented in spec §4.4.
    if not latest.is_symlink() and not latest.exists():
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


def train_torchrec_word2vec(rdf2vec, walks_path, vocab_path) -> None:
    """Top-level training entry point for the torchrec backend (decoupled).

    Called from :meth:`GPU_RDF2Vec.fit_from_walks` once pre-flight checks
    have passed. ``rdf2vec`` is the :class:`GPU_RDF2Vec` instance, which
    owns the config and the tracker.

    Consumes a PRE-GENERATED walk corpus + vocab from disk — it does NOT
    generate walks or build vocab. Those steps require a Dask-CUDA cluster
    / cuGraph dask comms, which cannot coexist with the torchrun/NCCL
    process group this function runs under (each multi-GPU paradigm wants
    to own all GPUs; you cannot run a ``LocalCUDACluster`` inside every
    torchrun rank). Prepare the corpus separately via
    :meth:`GPU_RDF2Vec.prepare_corpus` (outside torchrun). See
    ``docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md``
    §4.

    Parameters
    ----------
    walks_path : str
        Parquet directory of walks-as-sequences (``vertices`` /
        ``predicates`` list columns), as written by ``prepare_corpus``.
        Lives on a filesystem visible at the same path to every rank.
    vocab_path : str
        Parquet file mapping ``token_id`` (int64) ↔ ``entity`` (string),
        as written by ``prepare_corpus``. Also used as the DCP resume
        identity key.

    Implementation notes
    --------------------
    * Pair-loader construction lives on the rdf2vec instance
      (:meth:`_build_walk_pair_loader`) — see ``gpu_rdf2vec.py``.
    * All embedding parameters are fused via
      ``apply_optimizer_in_backward``; the standalone optimizer is a
      ``_NullOptimizer`` whose ``state_dict()`` is empty (DCP still
      requires *some* optimizer object to round-trip).
    """
    import time

    import torch
    from lightning.fabric import Fabric
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.optim import apply_optimizer_in_backward

    cfg = rdf2vec.config

    # 0. Seed for reproducible weight init + negative sampling. cfg.random_state
    #    was previously ignored on this path (the pytorch backend seeds via
    #    Lightning). Seed identically across ranks so the pre-shard table init
    #    is consistent; torchrec then shards it deterministically.
    if cfg.random_state is not None:
        torch.manual_seed(cfg.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_state)

    # 1. Load vocab + token counts from the prepared corpus. No walk
    #    generation, no dask, no cuGraph here — see the function docstring
    #    for why those cannot run under torchrun. ``vocab_path`` doubles as
    #    the DCP resume identity key.
    vocab_df, token_counts, vocab_path = rdf2vec._load_corpus_vocab(
        walks_path, vocab_path
    )
    vocab_size = len(vocab_df)

    # 2. Fabric.
    fabric = Fabric(
        accelerator="gpu",
        devices="auto",
        precision=_fabric_precision_from_config(cfg),
    )
    fabric.launch()

    # 3. Build the sharded model.
    model = ShardedSkipGramModel(
        vocab_size=vocab_size, embedding_dim=cfg.vector_size,
    ).to(fabric.device)

    # 4. Attach fused optimizer via apply_optimizer_in_backward BEFORE
    #    wrapping in DistributedModelParallel. This is the torchrec
    #    idiom: DMP detects the optimizer-in-backward marker on the raw
    #    embedding parameters and produces fused embedding+optimizer
    #    FBGEMM kernels. Calling apply_optimizer_in_backward on
    #    DMP-proxy parameters attaches hooks to the wrappers, NOT the
    #    underlying sharded tables, so the fused state is silently
    #    dropped.
    fused_cls, fused_kwargs = _fused_optimizer_from_config(cfg)
    apply_optimizer_in_backward(
        fused_cls, model.parameters(), fused_kwargs,
    )
    # All skip-gram params live in sharded embedding tables, so the
    # entire update fuses into backward. The standalone optimizer is a
    # no-op stand-in whose ``state_dict()`` is empty (DCP still requires
    # *some* optimizer object to round-trip).
    optimizer = _NullOptimizer()

    # 5. Build plan + wrap in DMP. The optimizer-in-backward hooks
    #    attached above survive the DMP wrapping.
    plan = _build_sharding_plan(model, world_size=fabric.world_size, device=fabric.device)
    model = DistributedModelParallel(
        module=model, device=fabric.device, plan=plan, init_data_parallel=False,
    )

    # 6. Maybe resume.
    start_epoch, start_step = _maybe_resume_checkpoint(
        model, optimizer,
        checkpoint_dir=cfg.checkpoint_dir,
        overwrite=cfg.checkpoint_overwrite,
        current_vocab_path=vocab_path,
    )

    # 7. DataLoader — yields (center, context) long-tensor batches.
    loader = rdf2vec._build_walk_pair_loader(walks_path, vocab_df)
    loader = fabric.setup_dataloaders(loader, use_distributed_sampler=False)

    # 8. Training loop.
    #    torchrec's sharded forward issues an all_to_all on EVERY step, so
    #    every rank MUST execute the same number of forward()s per epoch or
    #    the collective desyncs and the job deadlocks. The row-striped pair
    #    loader (see _build_walk_pair_loader) keeps the batch COUNT equal in
    #    the common case, but multi-file / multi-worker loaders can still
    #    drift by a batch, so we additionally gate each step on a min-reduce
    #    "join": the moment ANY rank runs out of batches, EVERY rank stops
    #    this epoch together (dropping at most the uneven tail). Do NOT skip
    #    empty batches mid-stream — a size-0 row stride still carries a
    #    tensor and must join the collective (its SUM loss is 0); only true
    #    exhaustion (next() -> None) lowers the continue flag.
    import torch.distributed as dist

    global_step = start_step
    last_ckpt_time = time.monotonic()
    progress = _ProgressMeter(
        every_n_steps=cfg.progress_log_every_n_steps,
        enabled=cfg.progress_print or _env_progress_enabled(),
    )
    token_counts_dev = (
        token_counts.to(fabric.device) if hasattr(token_counts, "to") else None
    )
    for epoch in range(start_epoch, cfg.epochs):
        data_iter = iter(loader)
        while True:
            batch = next(data_iter, None)
            have = torch.ones(1, device=fabric.device, dtype=torch.int32)
            if batch is None:
                have.zero_()
            dist.all_reduce(have, op=dist.ReduceOp.MIN)
            if int(have.item()) == 0:
                break
            center, context = batch
            negative = _sample_negatives(
                n_rows=center.size(0), n_neg=cfg.negative_samples,
                vocab_size=vocab_size, device=fabric.device,
                strategy=cfg.negative_sampling, token_counts=token_counts_dev,
            )
            loss = model(center, context, negative)
            fabric.backward(loss)
            # optimizer step is fused into backward via apply_optimizer_in_backward.
            # Progress emission (tracker + optional flushed stdout print) is
            # cadenced and rank-0-only. loss.item() forces a device sync, so we
            # only pay it on a logging step and only when something consumes it.
            # The block contains no collectives, so gating it to rank 0 is safe.
            want_log = (cfg.tracker != "none") or progress.enabled
            if fabric.is_global_zero and want_log and progress.due(global_step):
                loss_val = loss.item()
                if cfg.tracker != "none":
                    rdf2vec.tracker.log_metrics(
                        {"train/loss": loss_val, "epoch": float(epoch)},
                        step=global_step,
                    )
                if progress.enabled:
                    print(
                        progress.format(epoch=epoch, step=global_step,
                                        loss=loss_val, now=time.monotonic()),
                        flush=True,
                    )
            global_step += 1

            # Time-based checkpoint. The decision MUST be agreed across ranks
            # BEFORE any rank enters _save_checkpoint, which issues collectives
            # (dist.barrier / dcp.save). A per-rank wall-clock check would let
            # one rank checkpoint while a peer runs the next forward's
            # all-to-all — a collective mismatch that deadlocks. all_reduce(MAX)
            # makes every rank checkpoint on the same step as soon as ANY rank's
            # timer is due (clocks differ by ms, so they agree within a step),
            # and last_ckpt_time resets on all ranks together.
            if cfg.checkpoint_dir:
                due = torch.zeros(1, device=fabric.device, dtype=torch.int32)
                if (
                    time.monotonic() - last_ckpt_time
                    >= cfg.checkpoint_interval_minutes * 60
                ):
                    due.fill_(1)
                dist.all_reduce(due, op=dist.ReduceOp.MAX)
                if int(due.item()):
                    _save_checkpoint(
                        model, optimizer,
                        epoch=epoch, global_step=global_step,
                        checkpoint_dir=cfg.checkpoint_dir,
                        keep_last=cfg.checkpoint_keep_last,
                        vocab_path=vocab_path,
                        config_snapshot=cfg.model_dump(),
                    )
                    last_ckpt_time = time.monotonic()

        # Epoch-boundary checkpoint.
        if cfg.checkpoint_dir:
            _save_checkpoint(
                model, optimizer,
                epoch=epoch + 1, global_step=global_step,
                checkpoint_dir=cfg.checkpoint_dir,
                keep_last=cfg.checkpoint_keep_last,
                vocab_path=vocab_path,
                config_snapshot=cfg.model_dump(),
            )

    # 9. Stash for transform() — consumed by GPU_RDF2Vec.transform() in Task 12.
    rdf2vec._torchrec_model = model
    rdf2vec._torchrec_vocab_df = vocab_df


def _fabric_precision_from_config(cfg) -> str:
    """Map ``cfg.matmul_precision`` → Lightning Fabric ``precision=`` arg."""
    return {
        "highest": "32-true",
        "high":    "32-true",
        "medium":  "bf16-mixed",
    }.get(getattr(cfg, "matmul_precision", "high"), "32-true")


def _fused_optimizer_from_config(cfg) -> tuple[type, dict]:
    """Map cfg.optimizer → (torchrec fused optimizer class, kwargs).

    Notes on the torchrec 1.2.0 import paths
    ----------------------------------------
    The torchrec 1.2.0 install in this environment re-exports the fused
    optimizers from ``torchrec.optim.optimizers``; the older import
    location ``torchrec.optim.fused`` only contains the *base classes*
    (``FusedOptimizer``, ``EmptyFusedOptimizer``) and does NOT contain
    ``FusedSGD`` / ``PartialRowWiseAdam``.

    The plan's pseudocode used ``from torchrec.optim.fused import
    FusedSGD, PartialRowWiseAdam`` — this was speculative; the
    contract was that we land on classes compatible with
    ``apply_optimizer_in_backward``. The names below match what is
    actually exported and have the expected
    ``__init__(params, lr=...)`` signature.
    """
    # Plain SGD is rejected (also gated at _torchrec_preflight): the fused
    # optimizer-in-backward applies the update inside the FBGEMM kernel and CANNOT
    # clip — `gradient_clipping`/`max_gradient` are silently dropped (they belong to
    # a separate GradientClippingOptimizer wrapper for a standalone `.step()`), so
    # unclipped SGD diverges to garbage on large vocab (wikidata5m: loss 33k → 1.79M;
    # a 100× tighter clip changed nothing). Two stable fused optimizers are offered:
    #  - rowwiseadagrad (EXACT_ROWWISE_ADAGRAD): per-row Adagrad. Gradient-following
    #    (SGD-family, no momentum) with a per-row gradient-accumulation scaling that
    #    decays the effective per-row LR → inherently stable, no clip needed. The
    #    canonical torchrec/DLRM embedding optimizer.
    #  - sparseadam (PartialRowWiseAdam): per-row adaptive; stable but Adam-family
    #    (momentum can collapse embeddings to a common direction on tiny vocabs).
    # NB: LARS-SGD (OptimType.LARS_SGD) would be the ideal SGD-family fit, but
    # FBGEMM 1.4 has no get_optimizer_state for it, so DistributedModelParallel
    # cannot even construct a LARS-fused model (NotImplementedError at DMP init).
    # Defense-in-depth — never return a plain SGD here.
    if cfg.optimizer == "rowwiseadagrad":
        from torchrec.optim import RowWiseAdagrad
        return RowWiseAdagrad, {"lr": cfg.learning_rate}
    if cfg.optimizer == "sparseadam":
        from torchrec.optim import PartialRowWiseAdam
        return PartialRowWiseAdam, {"lr": cfg.learning_rate}
    raise ValueError(
        f"backend='torchrec' supports optimizer in {{'rowwiseadagrad', 'sparseadam'}} "
        f"(got {cfg.optimizer!r}); plain SGD's fused path cannot clip gradients → divergence."
    )


class _NullOptimizer:
    """Stand-in optimizer when every parameter is fused via
    :func:`apply_optimizer_in_backward`.

    Required because :func:`_save_checkpoint` /
    :func:`_maybe_resume_checkpoint` call ``optimizer.state_dict()`` /
    ``optimizer.load_state_dict()`` unconditionally and DCP needs some
    object to round-trip — an empty dict is fine, the fused-in-backward
    state lives inside the model's sharded buffers.
    """

    def step(self) -> None:  # noqa: D401 — protocol method
        pass

    def zero_grad(self) -> None:  # noqa: D401 — protocol method
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict) -> None:
        pass


def _materialize_walks_sequences_to_parquet(walks_seq, walks_parquet=None) -> str:
    """Persist the (dask_)cudf walks-sequences frame to a parquet directory
    and return its path.

    Mirrors what :meth:`GPU_RDF2Vec._fit_via_gensim` and
    :meth:`GPU_RDF2Vec._fit_via_pytorch_sequences` do — re-persist the
    dask frame defensively (so a future refactor that moves the persist
    out of the corpus class doesn't silently re-roll the stochastic walk
    generator on ``to_parquet``) and write each partition as a parquet
    file. For an in-memory cudf frame, write a single ``part.0.parquet``.

    Parameters
    ----------
    walks_seq
        The (dask_)cudf walks-as-sequences frame.
    walks_parquet : str, optional
        Target parquet directory. When ``None`` (the default), a fresh
        ``tempfile.mkdtemp`` directory is used — handy for throwaway
        corpora. ``prepare_corpus`` passes an explicit, stable path so
        the corpus is addressable across the prep→train process boundary
        and across DCP resume runs.

    Notes
    -----
    The directory is NOT cleaned up by this function — callers either
    rely on OS tempfile cleanup at shutdown, or (for ``prepare_corpus``)
    own the lifecycle of the corpus directory explicitly.
    """
    import os
    import tempfile

    if walks_parquet is None:
        walks_dir = tempfile.mkdtemp(prefix="rdf2vec_torchrec_walks_")
        walks_parquet = f"{walks_dir}/walks"
    if hasattr(walks_seq, "persist"):
        from dask.distributed import wait as dask_wait

        walks_seq = walks_seq.persist()
        dask_wait(walks_seq)
        walks_seq.to_parquet(walks_parquet, write_index=False)
    else:
        os.makedirs(walks_parquet, exist_ok=True)
        walks_seq.to_parquet(f"{walks_parquet}/part.0.parquet", index=False)
    return walks_parquet
