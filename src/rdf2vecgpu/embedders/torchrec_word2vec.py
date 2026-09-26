"""Torchrec-backed sharded Word2Vec training.

See docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md
for the design. v1 covers skip-gram only.

All torchrec / fbgemm imports happen *inside* function/class bodies so
this module can be imported without those extras installed.
"""
from __future__ import annotations

import os

import torch
from torch import nn


def _unigram_cdf(token_counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalised float64 CDF of the f^0.75 unigram distribution, built ONCE.

    float64 is not optional: at a 393M-token vocab the total weight is ~1e9-1e10,
    and a float32 running sum cannot represent increments below total × 6e-8 —
    every rare token (weight ~1) would collapse to zero sampling probability.
    Costs vocab_size × 8 B on the device (3.1 GB at 393M) — cheap next to the
    embedding shards it sits beside.
    """
    import math
    w = token_counts.to(device=device, dtype=torch.float64).clamp(min=0).pow(0.75)
    cdf = w.cumsum(0)
    total = cdf[-1].item()      # one host sync per run; a view would alias the in-place divide below
    if not math.isfinite(total) or total <= 0:
        raise ValueError("unigram negative sampling needs at least one token with count > 0")
    cdf /= total
    return cdf


def _sample_negatives(
    n_rows: int,
    n_neg: int,
    vocab_size: int,
    device: torch.device,
    strategy: str = "unigram",
    token_counts: torch.Tensor | None = None,
    cdf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample ``(n_rows, n_neg)`` negative token ids.

    Mirrors the f^0.75 unigram sampler used by the existing pytorch
    backend (see ``embedders/word2vec.py::_sample_negatives``). Returns
    a flat ``long`` tensor; the caller is responsible for routing
    through the sharded output embedding.

    ``unigram`` draws by inverse-CDF (``searchsorted`` on a cached CDF from
    :func:`_unigram_cdf`). Pass ``cdf`` from the training loop so the
    vocab-sized work happens once per run; when only ``token_counts`` is
    given the CDF is built on the fly (tests / one-off callers). The
    previous ``torch.multinomial`` on freshly-computed weights re-read and
    cumsum'd the whole vocab every step — ~6-9 GB of HBM traffic per step
    at 393M tokens, against ~0.1 GB of actual embedding work.
    """
    if strategy == "uniform":
        return torch.randint(
            low=0, high=vocab_size, size=(n_rows, n_neg),
            dtype=torch.long, device=device,
        )

    if strategy == "unigram":
        if cdf is None:
            if token_counts is None:
                raise ValueError("strategy='unigram' requires token_counts (or a prebuilt cdf)")
            cdf = _unigram_cdf(token_counts, device)
        u = torch.rand(n_rows * n_neg, device=device, dtype=torch.float64)
        # right=True -> first index whose cdf exceeds u: zero-weight tokens (flat
        # cdf steps) are never selected; clamp guards the u == 1.0 edge.
        idx = torch.searchsorted(cdf, u, right=True).clamp_(max=vocab_size - 1)
        return idx.view(n_rows, n_neg)

    raise ValueError(f"unknown negative-sampling strategy: {strategy!r}")


def _env_profile_enabled() -> bool:
    """RDF2VEC_TORCHREC_PROFILE=1 turns on :class:`_StepProfiler` (diagnostic
    runs only — it synchronises the device every step)."""
    import os
    return os.environ.get("RDF2VEC_TORCHREC_PROFILE", "").strip().lower() not in (
        "", "0", "false", "no",
    )


class _StepProfiler:
    """Per-step time attribution for diagnostic runs (RDF2VEC_TORCHREC_PROFILE=1).

    Host wall-clock per segment of the training step — ``drain`` (waiting at
    the top of the step for the previous step's GPU work: large means
    GPU-bound, ~0 means the GPU idles on the host), ``next`` (loader wait +
    H2D copy), ``join`` (the per-step all_reduce + ``.item()``),
    ``negatives``, ``forward`` (torchrec input all-to-all + lookup),
    ``backward`` (output all-to-all + fused optimizer) and ``log`` — plus the
    GPU time of forward+backward from CUDA events. Every rank measures; the
    report gathers the means so a slow rank (the one everybody's join waits
    for) shows up. Rank 0 additionally records a ``torch.profiler`` window and
    prints the kernel / collective table. Disabled (the default) every method
    returns immediately; enabled, ``drain()`` synchronises the device once per
    step, so the numbers attribute time but are not a production rate.
    """

    SEGMENTS = ("drain", "next", "join", "negatives", "forward", "backward", "log")

    def __init__(self, enabled: bool, device, rank: int, *, skip: int = 50,
                 profile_window: "tuple[int, int] | None" = (150, 20)):
        self.enabled = bool(enabled)
        self.device = device
        self.rank = int(rank)
        self.skip = int(skip)
        self.profile_window = profile_window
        self.step = 0
        self.host = {s: [] for s in self.SEGMENTS}
        self.gpu_ms: list = []
        self._t0 = {}
        self._events = None
        self._tp = None
        if self.enabled and self.rank == 0:
            print("[torchrec][profile] enabled: device synchronised every step — "
                  "the steps/s below are NOT a production rate", flush=True)

    def begin(self, seg: str) -> None:
        if self.enabled:
            import time
            self._t0[seg] = time.perf_counter()

    def end(self, seg: str) -> None:
        if self.enabled and self.step >= self.skip:
            import time
            self.host[seg].append(time.perf_counter() - self._t0[seg])

    def drain(self) -> None:
        """Top of the step: wait for outstanding GPU work (timed as ``drain``)
        and close the previous step's CUDA events."""
        if not self.enabled:
            return
        self.begin("drain")
        torch.cuda.synchronize(self.device)
        self.end("drain")
        if self._events is not None:
            a, b = self._events
            if b is not None and self.step - 1 >= self.skip:
                self.gpu_ms.append(a.elapsed_time(b))
            self._events = None

    def gpu_start(self) -> None:
        if self.enabled:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._events = (ev, None)

    def gpu_end(self) -> None:
        if self.enabled and self._events is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._events = (self._events[0], ev)

    def end_step(self) -> None:
        if not self.enabled:
            return
        self.step += 1
        if self.profile_window is None or self.rank != 0:
            return
        start, length = self.profile_window
        start += self.skip
        if self.step == start:
            from torch.profiler import ProfilerActivity, profile
            self._tp = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self._tp.start()
        elif self.step == start + length and self._tp is not None:
            self._tp.stop()
            print(f"[torchrec][profile] torch.profiler over steps {start}-{start + length} "
                  "(rank 0), by CUDA time:", flush=True)
            print(self._tp.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
            print("[torchrec][profile] same window by CPU time:", flush=True)
            print(self._tp.key_averages().table(sort_by="cpu_time_total", row_limit=15), flush=True)
            self._tp = None

    def summary(self) -> "dict | None":
        if not self.enabled:
            return None
        import numpy as np
        mean = {s: (1e3 * float(np.mean(v)) if v else 0.0) for s, v in self.host.items()}
        p95 = {s: (1e3 * float(np.percentile(v, 95)) if v else 0.0) for s, v in self.host.items()}
        return {
            "n": len(self.host["next"]), "mean_ms": mean, "p95_ms": p95,
            "total_ms": sum(mean.values()),
            "gpu_ms": float(np.mean(self.gpu_ms)) if self.gpu_ms else 0.0,
            "gpu_p95_ms": float(np.percentile(self.gpu_ms, 95)) if self.gpu_ms else 0.0,
        }

    def report(self, epoch: int) -> None:
        """Gather every rank's summary (collective) and print it on rank 0."""
        if not self.enabled:
            return
        import torch.distributed as dist
        mine = self.summary()
        if dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, mine)
        else:
            gathered = [mine]
        if self.rank != 0:
            return
        segs = " | ".join(f"{s} {mine['mean_ms'][s]:.2f}" for s in self.SEGMENTS)
        print(f"[torchrec][profile] epoch {epoch}: {mine['n']} measured steps after {self.skip} "
              f"warm-up; rank 0 host ms/step: {segs} | total {mine['total_ms']:.2f} ; "
              f"gpu fwd+bwd {mine['gpu_ms']:.2f} ms (p95 {mine['gpu_p95_ms']:.2f})", flush=True)
        p95 = " | ".join(f"{s} {mine['p95_ms'][s]:.2f}" for s in self.SEGMENTS)
        print(f"[torchrec][profile] rank 0 p95 ms: {p95}", flush=True)
        for r, g in enumerate(gathered):
            if g is None:
                continue
            m = g["mean_ms"]
            print(f"[torchrec][profile] rank {r}: next {m['next']:.2f} | drain {m['drain']:.2f} | "
                  f"join {m['join']:.2f} | forward {m['forward']:.2f} | backward {m['backward']:.2f} | "
                  f"total {g['total_ms']:.2f} | gpu {g['gpu_ms']:.2f} ms", flush=True)


def _env_step_sync_enabled() -> bool:
    """RDF2VEC_TORCHREC_STEP_SYNC=1: a device-wide ``torch.cuda.synchronize()``
    at the top of every training step (default off). Kept as an experiment
    switch: an A/B on dbpedia (2 GPUs, same node, 2026-09-08, jobs 14698160 /
    14698161) measured 22.1 vs 21.8 steps/s — no effect. The apparent 1.5×
    that motivated it was a node difference (o05c01 vs the gpu4 nodes), not
    the sync."""
    import os
    return os.environ.get("RDF2VEC_TORCHREC_STEP_SYNC", "0").strip().lower() not in (
        "", "0", "false", "no",
    )


def _host_mem_gib() -> tuple[float, float]:
    """This process's current and peak resident host memory in GiB
    (``VmRSS`` / ``VmHWM`` from ``/proc/self/status``; ``(nan, nan)`` where
    procfs is absent). Logged per rank at checkpoints and first batches:
    2026-09-21 job 14697729 (8x A100, 1500G) was OOM-killed writing its
    epoch-end checkpoint at 1.57 TB total, 450 GB above what the same epoch
    peaked at on the H200 node two weeks earlier, with nothing in the log to
    say whether that was growth over the epoch or a spike in the save."""
    cur = peak = float("nan")
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 2**20
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) / 2**20
    except OSError:
        pass
    return cur, peak


def _env_progress_enabled() -> bool:
    """Convenience override so the SLURM harness can flip the progress print
    without touching config plumbing: RDF2VEC_TORCHREC_PROGRESS=1."""
    import os
    return os.environ.get("RDF2VEC_TORCHREC_PROGRESS", "").strip().lower() not in (
        "", "0", "false", "no",
    )


def _start_stall_reporter(label: str, *, after_s: float = 300.0, every_s: float = 300.0):
    """Daemon thread that prints every Python thread's stack of this process,
    tagged ``label``, if the returned ``threading.Event`` is still clear after
    ``after_s`` seconds — and again every ``every_s`` until it is set. Setting
    the event ends the thread; a set event before the deadline costs nothing.

    Wrapped around the first ``next()`` of an epoch: 2026-09-14 job 14697730
    lost ranks 1, 2 and 5 there for over ten minutes — the other five ranks
    reached the first-batch all_reduce, the NCCL watchdog fired, and the log
    held nothing at all from the three stalled ranks. A rank that stalls
    there now says where it sits (loader thread included) before the
    process group gives up on it.
    """
    import sys
    import threading
    import time
    import traceback

    stop = threading.Event()
    t0 = time.monotonic()

    def run():
        wait = after_s
        while not stop.wait(wait):
            me = threading.get_ident()
            names = {t.ident: t.name for t in threading.enumerate()}
            out = [f"[torchrec] STALL {label}: {time.monotonic() - t0:.0f} s and counting; thread stacks:"]
            for tid, frame in sys._current_frames().items():
                if tid == me:
                    continue
                out.append(f"  --- thread {names.get(tid, '?')} ({tid})")
                out.extend("  " + line.rstrip("\n") for line in traceback.format_stack(frame))
            print("\n".join(out), flush=True)
            wait = every_s

    threading.Thread(target=run, name="torchrec-stall-reporter", daemon=True).start()
    return stop


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
        # rank 0's own resident host memory: a slow climb over an epoch is
        # visible here long before a checkpoint's spike lands on top of it.
        rss, _ = _host_mem_gib()
        return (
            f"[torchrec] epoch {epoch} | step {step:,} | "
            f"loss {loss:.4f} | {sps:.1f} steps/s | rss {rss:.1f} GiB"
        )


def _build_sharding_plan(
    model: nn.Module, world_size: int, device: torch.device, pg=None,
) -> "ShardingPlan":  # noqa: F821 (lazy import — see body)
    """Plan with torchrec's planner, with every table pinned to ROW-WISE.

    Left to its cost model the planner places any table that fits one GPU
    *table-wise* — measured 2026-09-05 on dbpedia2016 d100 (6.3M rows,
    2×A100): in_embeddings landed on rank 1 and out_embeddings on rank 0.
    That silently breaks the one-row-block-per-rank layout everything
    downstream assumes (:func:`_get_in_table_local_shard`,
    :func:`_local_row_range`, the DCP shard files) and leaves one GPU's HBM
    idle per table. The constraint fixes the sharding type only; the
    planner still picks the compute kernel. ``world_size == 1`` has nothing
    to split and stays unconstrained.

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

    # local_world_size = GPUs per node. Without it Topology assumes all ranks
    # live on one host (local_world_size==world_size) and places global rank r's
    # shard on cuda:r — which doesn't exist on a multi-node run (node 1's rank 4
    # is physically cuda:0). Derive it from the elastic env; falls back to
    # world_size for a single-node job (unchanged behaviour).
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or world_size
    # hbm_cap: the planner's default is a 32 GB GPU (torchrec HBM_CAP). A
    # full-Wikidata d100 rank block is 2 x 18.3 GiB of tables, so under that
    # default the planner could place only ONE table device-resident and put
    # the other on a UVM-caching kernel: 18.3 GiB per rank of pinned HOST
    # memory behind a 4 GiB GPU cache (probe 2026-09-25, job 14908855). That
    # is where the training's ~140 GB host RSS per rank came from, and every
    # cache miss on the in-table crossed PCIe — 48 steps/s on A100 vs 80 on
    # H200. Telling the planner the real per-GPU memory puts both tables on
    # the device (36.6 GiB per rank; fits A100-80GB and H200-141GB).
    topology = Topology(
        world_size=world_size,
        local_world_size=local_world_size,
        compute_device=device.type,  # "cuda"
        hbm_cap=_planner_hbm_cap(device),
    )
    constraints = None
    if world_size > 1:
        from torchrec.distributed.planner.types import ParameterConstraints
        from torchrec.distributed.types import ShardingType

        constraints = {
            cfg.name: ParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
            )
            for m in model.modules()
            if hasattr(m, "embedding_configs")
            for cfg in m.embedding_configs()
        }
    planner = EmbeddingShardingPlanner(topology=topology, constraints=constraints)
    sharders: list[ModuleSharder] = [EmbeddingCollectionSharder()]

    if pg is None and dist.is_available() and dist.is_initialized():
        pg = dist.group.WORLD
    if pg is not None:
        plan = planner.collective_plan(model, sharders, pg)
    else:
        plan = planner.plan(model, sharders=sharders)
    _print_plan_kernels(plan, pg)
    return plan


def _planner_hbm_cap(device: torch.device) -> int | None:
    """Per-GPU memory in bytes to hand the torchrec planner as ``hbm_cap``.

    ``RDF2VEC_TORCHREC_HBM_CAP_GB`` overrides (a small value forces the UVM
    kernels on purpose — the 2026-09-25 kernel-swap test used 2 GB on
    dbpedia); otherwise the real capacity of the compute device; ``None``
    (the planner's own 32 GB default) off-CUDA, where no table is placed on
    a device anyway.
    """
    env = os.environ.get("RDF2VEC_TORCHREC_HBM_CAP_GB", "").strip()
    if env:
        return int(float(env) * 1024**3)
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return int(torch.cuda.get_device_properties(idx).total_memory)
    return None


def _print_plan_kernels(plan, pg) -> None:
    """One line per table on rank 0: sharding type, compute kernel, ranks.
    The kernel decides where the table lives (``fused`` = device,
    ``fused_uvm*`` = host memory) — the log never said, and it took a
    separate probe to learn the in-table had been host-resident all along."""
    import torch.distributed as dist

    if pg is not None and dist.is_initialized() and dist.get_rank(pg) != 0:
        return
    try:
        for module_path, tables in plan.plan.items():
            for name, ps in tables.items():
                print(
                    f"[torchrec] plan: {module_path}/{name}: sharding={ps.sharding_type} "
                    f"kernel={ps.compute_kernel} ranks={ps.ranks}",
                    flush=True,
                )
    except AttributeError as exc:  # a plan type without .plan (tests may stub it)
        print(f"[torchrec] plan: (not printable: {exc})", flush=True)


class ShardedSkipGramModel(nn.Module):
    """Skip-gram with two row-wise sharded embedding tables (in + out).

    Sharding is applied later by wrapping this module in
    ``torchrec.distributed.DistributedModelParallel`` with a plan from
    `_build_sharding_plan` (Task 7). Prior to wrapping, the table parameters
    live on whichever device the caller placed them.
    """

    IN_TABLE_NAME = "in_embeddings"
    OUT_TABLE_NAME = "out_embeddings"

    def __init__(self, vocab_size: int, embedding_dim: int, device=None):
        """``device=torch.device("meta")`` builds the tables WITHOUT storage;
        ``DistributedModelParallel`` then materialises only this rank's shard
        on its GPU. The trainer always does that: the two tables are
        ``2 × vocab × dim`` fp32 (314 GB for Wikidata d100), so building
        them on the default device — as before 2026-09-06 — allocated the
        full tables in host RAM on EVERY rank (8 × 314 GB) before ``.to(gpu)``
        and was OOM-killed at 1.5 TB. ``device=None`` (tests, world_size=1
        experiments) keeps the plain eager allocation."""
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
        self.in_collection = EmbeddingCollection(tables=[in_cfg], device=device)
        self.out_collection = EmbeddingCollection(tables=[out_cfg], device=device)

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

        # One fused multiply + reduce over (B, 1+neg, dim) yields every logit.
        # This used to be torch.bmm(neg_embed, center.unsqueeze(-1)): a batch
        # of B tiny (neg×dim)·(dim×1) GEMMs that cuBLAS runs as ~B independent
        # micro-kernels — 33% of the step's GPU time in the 2026-09-08 dbpedia
        # profile (22 of 68 ms at 525k pairs/rank), more than the sharded
        # lookups, the all-to-all and the optimizer together. Same math, same
        # gradients; only the kernel changed.
        scores = (out_embed * center_embed.unsqueeze(1)).sum(dim=-1)   # (B, 1+neg)
        pos_score = scores[:, 0]                                        # (B,)
        neg_score = scores[:, 1:]                                       # (B, neg)

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

    Paths:
    1. Post-DMP, ``ShardedTensor`` state_dict value (the default
       ``ShardingEnv``): ``(shard.tensor, shard_offsets[0])``.
    2. Post-DMP, ``DTensor`` value (``ShardingEnv.output_dtensor=True``):
       torchrec wraps this rank's shards in a ``LocalShardsWrapper`` that
       carries the same offsets.
    3. Pre-DMP / unwrapped ``EmbeddingCollection``: the whole table lives on
       this rank starting at row 0.

    A DMP-wrapped model whose in-table has NO local shard on this rank (the
    planner placed it table-wise elsewhere) raises instead of falling
    through to path 3 — on the wrapper that path would either fail
    (``DistributedModelParallel`` has no ``in_collection``) or mislabel a
    whole table as a shard at offset 0. :func:`_build_sharding_plan` pins
    the plan to row-wise precisely so this cannot happen.
    """
    try:
        sd = model.state_dict()
    except (AttributeError, RuntimeError) as exc:
        print(f"[torchrec] state_dict() unavailable for extraction "
              f"({type(exc).__name__}: {exc}); trying direct table access")
        sd = {}
    matched = "no state_dict key matched"
    for key, val in sd.items():
        if ShardedSkipGramModel.IN_TABLE_NAME in key and "weight" in key:
            matched = f"{key} -> {type(val).__name__}"
            if hasattr(val, "local_shards"):                    # ShardedTensor
                shards = val.local_shards()
                if shards:
                    shard = shards[0]
                    return shard.tensor, int(shard.metadata.shard_offsets[0])
                matched += " with no local shard on this rank"
            elif hasattr(val, "to_local"):                      # DTensor
                local = val.to_local()
                if hasattr(local, "local_shards"):              # LocalShardsWrapper
                    shards, offsets = local.local_shards(), local.local_offsets()
                    if shards:
                        return shards[0], int(offsets[0][0])
                    matched += " with no local shard on this rank"
                else:
                    from torch.distributed.tensor._utils import (
                        compute_local_shape_and_global_offset,
                    )
                    _, offset = compute_local_shape_and_global_offset(
                        val.shape, val.device_mesh, val.placements,
                    )
                    return local, int(offset[0])
            # Plain tensor (unwrapped nn.Embedding weight): path 3 below.
            break

    # Path 3: unwrapped EmbeddingCollection — the whole table, offset 0.
    inner = _unwrap_dmp(model)
    collection = getattr(inner, "in_collection", None)
    tables = getattr(collection, "embeddings", None)
    if tables is not None and not _is_sharded_module(collection):
        for table in tables.values():
            return table.weight.data, 0
    raise RuntimeError(
        "could not locate a local shard of the in-embedding table on this "
        f"rank ({matched}). Extraction needs the plan to be row-wise so every "
        "rank owns one contiguous row block — see _build_sharding_plan."
    )


def _unwrap_dmp(model):
    """``DistributedModelParallel`` keeps the (sharded) user module behind
    ``.module``; a bare :class:`ShardedSkipGramModel` is returned as is."""
    return getattr(model, "module", model)


def _is_sharded_module(module) -> bool:
    try:
        from torchrec.distributed.types import ShardedModule
    except ImportError:  # pragma: no cover - torchrec absent
        return False
    return isinstance(module, ShardedModule)


def _read_vocab_slice(vocab_path: str, start: int, end: int):
    """Host-side ``entity`` strings for token ids ``[start, end)`` of the
    corpus ``vocab.parquet``, in token-id order, as a ``pyarrow`` ChunkedArray
    of ``large_string``.

    Reads only the row groups whose ``token_id`` statistics overlap the range
    (``prepare_corpus`` writes the vocab in token-id order, so that is one
    contiguous run of row groups) and scans any row group that carries no
    statistics. The result is verified to be exactly ``range(start, end)``:
    a gap or duplicate would silently mislabel every row after it.

    The entity column is promoted to ``large_string`` (int64 offsets) as soon
    as it is read: a ``string`` array caps at 2 GiB of bytes, and a
    full-Wikidata rank block (49M IRIs ≈ 2.3 GB) crosses that the moment the
    sort below materialises the block as one array — 2026-09-14 job 14697728
    died there with ``ArrowInvalid: offset overflow while concatenating
    arrays``. The Wikidata vocab's row groups are not in token-id order, so
    the sort path is the normal path at that scale, not a fallback.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(vocab_path)
    tid_idx = pf.schema_arrow.get_field_index("token_id")
    parts = []
    for rg in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(rg).column(tid_idx).statistics
        if st is not None and st.has_min_max and (st.max < start or st.min >= end):
            continue
        t = pf.read_row_group(rg, columns=["token_id", "entity"])
        tid = t.column("token_id")
        t = t.filter(pc.and_(pc.greater_equal(tid, start), pc.less(tid, end)))
        if t.num_rows:
            ent_idx = t.schema.get_field_index("entity")
            t = t.set_column(ent_idx, "entity", t.column("entity").cast(pa.large_string()))
            parts.append(t)
    table = pa.concat_tables(parts) if parts else pa.table(
        {"token_id": pa.array([], pa.int64()), "entity": pa.array([], pa.large_string())}
    )
    ids = table.column("token_id").to_numpy()
    if ids.shape[0] and not np.all(np.diff(ids) == 1):
        table = table.take(pc.sort_indices(table, sort_keys=[("token_id", "ascending")]))
        ids = table.column("token_id").to_numpy()
    expected = end - start
    contiguous = ids.shape[0] == expected and (
        expected == 0
        or (int(ids[0]) == start and int(ids[-1]) == end - 1 and bool(np.all(np.diff(ids) == 1)))
    )
    if not contiguous:
        raise RuntimeError(
            f"vocab slice [{start}, {end}) of {vocab_path} is not the contiguous token-id "
            f"range expected for this shard (got {ids.shape[0]} rows, expected {expected}); "
            "the checkpoint and vocab.parquet do not belong together"
        )
    return table.column("entity")


def iter_local_embedding_chunks(model, vocab_path: str, *, chunk_rows: int = 1_000_000):
    """Stream this rank's row block of the in-embedding table to the host.

    Yields ``(token_ids, entities, vectors)`` in token-id order, at most
    ``chunk_rows`` rows per item: an int64 numpy array, a ``pyarrow``
    ChunkedArray of entity strings and a float32 ``(rows, dim)`` numpy array.

    This is the export path for vocabularies where
    :func:`extract_embeddings_to_dask_cudf` — a cudf copy of the shard joined
    with the full vocab frame on the device — does not fit next to the sharded
    tables (at 393M Wikidata tokens that join needs ~110 GB per rank, more
    than an A100-80GB holds). Here the device holds nothing beyond the model;
    host memory is the block's entity strings (~2.5 GB for 49M IRIs) plus one
    chunk. The row offset comes from the shard metadata
    (:func:`_get_in_table_local_shard`), so the ids are exact under torchrec's
    ceil-block row-wise split.
    """
    import numpy as np

    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    in_param, start = _get_in_table_local_shard(model)
    local_rows = int(in_param.shape[0])
    entities = _read_vocab_slice(vocab_path, start, start + local_rows)
    for i in range(0, local_rows, chunk_rows):
        n = min(chunk_rows, local_rows - i)
        vectors = in_param[i:i + n].detach().to(torch.float32).cpu().numpy()
        yield (
            np.arange(start + i, start + i + n, dtype=np.int64),
            entities.slice(i, n),
            vectors,
        )


def _checkpoint_writer(tmp_dir):
    """``FileSystemWriter`` for :func:`_save_checkpoint` with the GPU->host
    copy-ahead turned OFF (``per_thread_copy_ahead=0``; env
    ``RDF2VEC_DCP_COPY_AHEAD_BYTES`` restores torch's 10 MB default).

    With copy-ahead on (torch's default for a single write thread) the
    writer stages every device tensor with ``to("cpu", non_blocking=True)``,
    i.e. through the caching *pinned* host allocator, which rounds each
    block up to a power of two and keeps it. Measured 2026-09-26 (jobs
    14912432 / 14919655, dbpedia on one A100, tables 2 x 2.35 GiB): host RSS
    rose by exactly 8 GiB = 2 x 4 GiB at EVERY checkpoint and never came
    back, under both the UVM and the fused kernel, until the 64G job was
    OOM-killed. At Wikidata scale that is 2 x 32 GiB per rank per save —
    what killed the 8x A100 run (14697729) at its third checkpoint. With
    copy-ahead off the writer's serial loader copies with a blocking
    ``.cpu()`` into pageable memory that is freed after the write.
    """
    import torch.distributed.checkpoint as dcp

    ahead = int(os.environ.get("RDF2VEC_DCP_COPY_AHEAD_BYTES", "0") or 0)
    return dcp.FileSystemWriter(str(tmp_dir), per_thread_copy_ahead=ahead)


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
    rss0, hwm0 = _host_mem_gib()
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dcp.save(
        state_dict=state_dict,
        storage_writer=_checkpoint_writer(tmp_dir),
    )
    rss1, hwm1 = _host_mem_gib()
    print(
        f"[torchrec] rank {rank}: checkpoint {final_name} written; host RSS "
        f"{rss0:.1f} -> {rss1:.1f} GiB, peak {hwm0:.1f} -> {hwm1:.1f} GiB",
        flush=True,
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

    if rank == 0:
        # Also the marker an export-only resubmit's race claim keys on
        # (performance/helix/train_torchrec_gpu8.sbatch): the checkpoint is
        # loaded, so the memory-critical start-up is behind us.
        print(
            f"[torchrec] resumed from {step_dir.name}: epoch={int(meta['epoch'])} "
            f"global_step={int(meta['global_step'])}",
            flush=True,
        )
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
    #    load_frame=False: training never touches the entity strings, and at
    #    Wikidata scale the frame is 393M IRIs (~20 GB of GPU memory and a
    #    7.4 GB parquet read PER RANK). transform() reads it lazily instead.
    vocab_df, token_counts, vocab_path = rdf2vec._load_corpus_vocab(
        walks_path, vocab_path, load_frame=False,
    )
    vocab_size = int(token_counts.numel())

    # 2. Fabric. Under torchrun, WORLD_SIZE spans all nodes while devices="auto"
    #    resolves to the per-node GPU count (LOCAL_WORLD_SIZE); Fabric defaults
    #    num_nodes=1, so a multi-node launch fails its devices*num_nodes==world_size
    #    check. Derive num_nodes from the elastic env (=1 for single-node).
    _ws = int(os.environ.get("WORLD_SIZE", "0"))
    _lws = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    _num_nodes = max(1, _ws // _lws) if (_ws and _lws) else 1
    fabric = Fabric(
        accelerator="gpu",
        devices="auto",
        num_nodes=_num_nodes,
        precision=_fabric_precision_from_config(cfg),
    )
    fabric.launch()

    # 3. Build the sharded model ON THE META DEVICE (no storage). DMP below
    #    materialises only this rank's row block directly on the GPU
    #    (init_parameters=True, its default). See ShardedSkipGramModel.__init__
    #    for why eager construction cannot work at Wikidata scale.
    model = ShardedSkipGramModel(
        vocab_size=vocab_size, embedding_dim=cfg.vector_size,
        device=torch.device("meta"),
    )

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
    # Unigram negatives: build the f^0.75 CDF ONCE on the device; every step then
    # only pays a searchsorted over its own samples instead of a vocab-sized pass.
    neg_cdf = (
        _unigram_cdf(token_counts, fabric.device)
        if cfg.negative_sampling == "unigram" and hasattr(token_counts, "to")
        else None
    )
    prof = _StepProfiler(_env_profile_enabled(), fabric.device, fabric.global_rank)
    step_sync = _env_step_sync_enabled() and not prof.enabled   # the profiler drains itself
    if fabric.is_global_zero:
        print(f"[torchrec] per-step device sync: {'on' if step_sync or prof.enabled else 'off'}", flush=True)
    who = f"rank {fabric.global_rank}/{fabric.world_size}"
    for epoch in range(start_epoch, cfg.epochs):
        data_iter = iter(loader)
        first_pending = True
        while True:
            if step_sync:
                torch.cuda.synchronize(fabric.device)
            prof.drain()
            prof.begin("next")
            if first_pending:
                # Every rank reports its first batch (not just rank 0): the
                # first ``next()`` reads the first walk file and starts the
                # prefetch thread, and a rank that never gets past it is
                # otherwise invisible until the NCCL watchdog kills the job.
                # NB: the gpu8 race claim greps "[torchrec] epoch|resumed" —
                # this line must not start with either word.
                t_first = time.monotonic()
                stall = _start_stall_reporter(f"{who} waiting for the first batch of epoch {epoch}")
            batch = next(data_iter, None)
            if first_pending:
                stall.set()
                first_pending = False
                rss, hwm = _host_mem_gib()
                print(
                    f"[torchrec] {who}: first batch of epoch {epoch} "
                    f"{'ready' if batch is not None else 'ABSENT (loader yielded nothing)'} "
                    f"after {time.monotonic() - t_first:.1f} s; host RSS {rss:.1f} GiB, peak {hwm:.1f} GiB",
                    flush=True,
                )
            prof.end("next")
            prof.begin("join")
            have = torch.ones(1, device=fabric.device, dtype=torch.int32)
            if batch is None:
                have.zero_()
            dist.all_reduce(have, op=dist.ReduceOp.MIN)
            ok = int(have.item())
            prof.end("join")
            if ok == 0:
                break
            center, context = batch
            prof.begin("negatives")
            negative = _sample_negatives(
                n_rows=center.size(0), n_neg=cfg.negative_samples,
                vocab_size=vocab_size, device=fabric.device,
                strategy=cfg.negative_sampling, cdf=neg_cdf,
            )
            prof.end("negatives")
            prof.gpu_start()
            prof.begin("forward")
            loss = model(center, context, negative)
            prof.end("forward")
            prof.begin("backward")
            fabric.backward(loss)
            prof.end("backward")
            prof.gpu_end()
            # optimizer step is fused into backward via apply_optimizer_in_backward.
            # Progress emission (tracker + optional flushed stdout print) is
            # cadenced and rank-0-only. loss.item() forces a device sync, so we
            # only pay it on a logging step and only when something consumes it.
            # The block contains no collectives, so gating it to rank 0 is safe.
            prof.begin("log")
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
            prof.end("log")
            global_step += 1
            prof.end_step()

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

        prof.report(epoch)

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
    rdf2vec._torchrec_vocab_df = vocab_df      # None → transform() reads the parquet lazily
    rdf2vec._torchrec_vocab_path = vocab_path
    rdf2vec._torchrec_vocab_size = vocab_size


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
