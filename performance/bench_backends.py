"""Cross-backend RDF2Vec/Word2Vec performance harness (current `RDF2VecConfig` API).

Measures, on the SAME graph + hyperparameters, for each training backend:
  - wall-clock: data load, corpus/walk prep, training, total
  - throughput: walks/sec (walk_number x start-vertices)
  - peak GPU memory: torch.cuda.max_memory_allocated() per process

and appends one JSON row per run to ``--out`` (default
``performance/results/bench_results.jsonl``).

Workflow — prepare-once, then train-only (removes the walk-generation confound):

    # 1. Prepare corpus once from your graph
    python performance/bench_backends.py --prepare --data G.parquet --corpus corpus/

    # 2. Train each backend from the shared corpus (training-only timing)
    python performance/bench_backends.py --backend gensim     --corpus corpus/
    python performance/bench_backends.py --backend pytorch    --corpus corpus/
    python performance/bench_backends.py --backend pytorch-mg --corpus corpus/
    NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=N \\
      performance/bench_backends.py --backend torchrec --corpus corpus/

Full-pipeline fallback (prepare + train in one shot, original behaviour):

    backend     training        launch
    --------    -----------     ------------------------------------------------
    gensim      CPU             python performance/bench_backends.py --backend gensim --data G.parquet
    pytorch     single-GPU      python performance/bench_backends.py --backend pytorch --data G.parquet
    pytorch-mg  multi-GPU DDP   python performance/bench_backends.py --backend pytorch-mg --data G.parquet
    torchrec    sharded         NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=N \\
                                  performance/bench_backends.py --backend torchrec --data G.parquet

IMPORTANT — read before interpreting results:

torchrec's value is sharding embedding tables that EXCEED one GPU's VRAM. On a
graph whose table fits on a single GPU (wikidata-5m ≈ 4.8 M tokens × 100 dim ×
4 B × 2 tables ≈ 3.8 GB — fits an A6000), torchrec's row-wise sharding +
per-step all-to-all is pure overhead and will NOT beat single-GPU or DDP. Report
peak VRAM next to time so the trade-off is explicit, and ALSO run a "doesn't-fit"
scenario (a very large vocab) where the dense backends OOM and only torchrec
completes — that is torchrec's actual selling point. See performance/README.md.

This script is the SCAFFOLD: wire up your dataset paths and run the commands
above. It does not run anything on import.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager


# ── hyperparameters (override on the CLI; defaults match performance/evaluation_parameters.py) ──
DEFAULTS = dict(
    walk_strategy="random",
    walk_depth=4,
    walk_number=100,
    embedding_model="skipgram",
    epochs=5,
    vector_size=100,
    window_size=5,
    min_count=1,
    learning_rate=0.025,
    negative_samples=5,
    random_state=42,
    cpu_count=20,
    grad_clip_value=0.1,
    optimizer="sgd",
)


@contextmanager
def _timed(store: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round(time.perf_counter() - t0, 3)


def _peak_vram_gib() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    except Exception:
        return None


def _reset_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _is_leader() -> bool:
    """Only the leader process writes the result row.

    - single-process backends (gensim / pytorch single-GPU): no RANK/LOCAL_RANK → leader.
    - torchrec (torchrun): non-rank-0 already returns None, but guard anyway.
    - pytorch-mg: Lightning's DDP-subprocess launcher re-runs this whole script in
      each worker (LOCAL_RANK set), so every rank would otherwise append a row.
    """
    return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"


def _config(backend: str, multi_gpu: bool, args) -> "RDF2VecConfig":  # noqa: F821
    from rdf2vecgpu import RDF2VecConfig
    return RDF2VecConfig(
        backend=backend,
        walk_strategy=args.walk_strategy,
        walk_depth=args.walk_depth,
        walk_number=args.walk_number,
        embedding_model=args.embedding_model,
        epochs=args.epochs,
        vector_size=args.vector_size,
        window_size=args.window_size,
        min_count=args.min_count,
        learning_rate=args.learning_rate,
        negative_samples=args.negative_samples,
        random_state=args.random_state,
        multi_gpu=multi_gpu,
        cpu_count=args.cpu_count,
        grad_clip_value=args.grad_clip_value,
        optimizer=args.optimizer,
    )


def _common_metrics(args, backend: str) -> dict:
    return {
        "backend": backend,
        "dataset": args.data,
        "walk_depth": args.walk_depth,
        "walk_number": args.walk_number,
        "vector_size": args.vector_size,
        "epochs": args.epochs,
        "tag": args.tag,
    }


# ─────────────────────────── prepare-only mode ───────────────────────────
def run_prepare(args) -> dict:
    """Prepare (walk + vocab) corpus from --data, write to --corpus. Times the whole step."""
    from rdf2vecgpu import GPU_RDF2Vec
    cfg = _config("pytorch", False, args)
    m = GPU_RDF2Vec(config=cfg)
    metrics = _common_metrics(args, "prepare")
    with _timed(metrics, "t_prepare_corpus_s"):
        edges = m.load_data(args.data)
        m.prepare_corpus(edges, args.corpus)
    return metrics


# ─────────────────────────── train-from-corpus (gensim/pytorch/pytorch-mg) ───────────────────────────
def run_from_corpus(args) -> dict:
    """Train one backend from a shared prepared corpus (fit_from_walks only)."""
    from rdf2vecgpu import GPU_RDF2Vec
    backend = "pytorch" if args.backend in ("pytorch", "pytorch-mg") else args.backend
    multi_gpu = args.backend == "pytorch-mg"
    cfg = _config(backend, multi_gpu, args)
    m = GPU_RDF2Vec(config=cfg)                    # pytorch-mg needs NO Dask client now
    walks = os.path.join(args.corpus, "walks")
    vocab = os.path.join(args.corpus, "vocab.parquet")
    metrics = _common_metrics(args, args.backend)
    _reset_vram()
    with _timed(metrics, "t_fit_from_walks_s"):
        m.fit_from_walks(walks, vocab)
    with _timed(metrics, "t_transform_s"):
        emb = m.transform()
        # torchrec returns a dask_cudf shard per rank; materialise to force the
        # extraction work so the timing is comparable across backends.
        if hasattr(emb, "compute"):
            emb = emb.compute()
    metrics["peak_vram_gib"] = _peak_vram_gib()
    return metrics


# ─────────────────────────── single-process backends (full pipeline) ───────────────────────────
def run_single_process(args) -> dict:
    """gensim / pytorch (single-GPU) / pytorch-mg (DDP via a Dask client) — full pipeline fallback."""
    from rdf2vecgpu import GPU_RDF2Vec

    backend = "pytorch" if args.backend in ("pytorch", "pytorch-mg") else args.backend
    multi_gpu = args.backend == "pytorch-mg"
    cfg = _config(backend, multi_gpu, args)

    client = None
    comms = None
    if multi_gpu:
        # pytorch-mg trains via Lightning DDP and prepares data (walks + vocab)
        # on a Dask-CUDA cluster — both live in this single process. cuGraph's
        # multi-GPU walk generation needs its dask comms initialized explicitly
        # (mirrors test/e2e_multi_gpu_test.py).
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        import cugraph.dask.comms.comms as Comms
        cluster = LocalCUDACluster()
        client = Client(cluster)
        Comms.initialize(p2p=True)
        comms = Comms

    m = GPU_RDF2Vec(config=cfg, client=client) if client else GPU_RDF2Vec(config=cfg)

    metrics = _common_metrics(args, args.backend)
    _reset_vram()
    try:
        with _timed(metrics, "t_load_s"):
            edges = m.load_data(args.data)
        with _timed(metrics, "t_fit_transform_s"):
            m.fit_transform(edge_df=edges, walk_vertices=None)
        metrics["peak_vram_gib"] = _peak_vram_gib()
    finally:
        if comms is not None:
            comms.destroy()
        if client is not None:
            client.close()
            client.cluster.close()
    return metrics


# ─────────────────────────── torchrec (under torchrun) ───────────────────────────
def run_torchrec(args) -> dict | None:
    """Train with torchrec under torchrun.

    If ``--corpus`` is given: every rank trains directly from the pre-prepared
    corpus (training-only timing — no prep step).  Rank 0 does NOT re-prepare.

    If ``--corpus`` is NOT given (legacy / full-pipeline): rank 0 prepares the
    corpus, then every rank trains from it.

    Returns metrics on rank 0, None on other ranks (so only rank 0 writes the
    result row).
    """
    import torch
    import torch.distributed as dist
    from rdf2vecgpu import GPU_RDF2Vec

    if "RANK" not in os.environ:
        raise SystemExit(
            "--backend torchrec must be launched under torchrun, e.g.\n"
            "  NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 "
            "performance/bench_backends.py --backend torchrec --corpus corpus/"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    metrics = _common_metrics(args, "torchrec")
    metrics["world_size"] = world_size

    if args.corpus:
        # ── train-from-corpus mode: skip prep, time fit_from_walks only ──
        walks_path = os.path.join(args.corpus, "walks")
        vocab_path = os.path.join(args.corpus, "vocab.parquet")
    else:
        # ── legacy full-pipeline mode: rank 0 prepares, peers wait ──
        corpus_dir = args.corpus_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "bench_corpus"
        )
        walks_path = os.path.join(corpus_dir, "walks")
        vocab_path = os.path.join(corpus_dir, "vocab.parquet")
        ready = os.path.join(corpus_dir, "READY")

        if rank == 0:
            prep_cfg = _config("pytorch", False, args)  # single-GPU data path
            prep = GPU_RDF2Vec(config=prep_cfg)
            with _timed(metrics, "t_prepare_corpus_s"):
                edges = prep.load_data(args.data)
                prep.prepare_corpus(edges, corpus_dir)
            open(ready, "w").write("ok")
        else:
            deadline = time.monotonic() + args.prep_timeout_s
            while not os.path.exists(ready):
                if time.monotonic() > deadline:
                    raise TimeoutError("rank-0 corpus prep timed out")
                time.sleep(1.0)

    # Sharded training on every rank.
    m = GPU_RDF2Vec(config=_config("torchrec", True, args))
    _reset_vram()
    dist.barrier()
    with _timed(metrics, "t_fit_from_walks_s"):
        m.fit_from_walks(walks_path, vocab_path)
    metrics["peak_vram_gib"] = _peak_vram_gib()
    dist.barrier()
    dist.destroy_process_group()

    return metrics if rank == 0 else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── mode flags ──
    p.add_argument("--prepare", action="store_true",
                   help="prepare walk corpus from --data into --corpus, write a 'prepare' metrics row, then exit")
    p.add_argument("--backend", choices=["gensim", "pytorch", "pytorch-mg", "torchrec"],
                   help="training backend (required unless --prepare is set)")
    # ── data / corpus ──
    p.add_argument("--data", default="",
                   help="path to the triple parquet (required for --prepare or full-pipeline mode)")
    p.add_argument("--corpus", default="",
                   help="directory holding a prepared corpus (walks/ + vocab.parquet). "
                        "When set (and not --prepare), backends train from this corpus via "
                        "fit_from_walks only (training-only timing).")
    # ── output ──
    p.add_argument("--out", default="performance/results/bench_results.jsonl")
    p.add_argument("--tag", default="", help="free-form label for this run (e.g. 'wikidata5m')")
    # ── legacy / torchrec options ──
    p.add_argument("--nproc", type=int, default=0, help="pytorch-mg: Dask-CUDA workers (0=auto). torchrec uses torchrun --nproc_per_node.")
    p.add_argument("--corpus-dir", default="", help="torchrec full-pipeline fallback: where to write the prepared corpus (default: alongside --out)")
    p.add_argument("--prep-timeout-s", type=int, default=3600)
    for k, v in DEFAULTS.items():
        argtype = type(v)
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=argtype, default=v)
    args = p.parse_args()

    # ── validate arg combinations ──
    if args.prepare:
        if not args.data:
            p.error("--prepare requires --data")
        if not args.corpus:
            p.error("--prepare requires --corpus")
        metrics = run_prepare(args)
    elif args.backend is None:
        p.error("--backend is required unless --prepare is set")
    elif args.backend == "torchrec":
        metrics = run_torchrec(args)
    elif args.corpus:
        metrics = run_from_corpus(args)
    else:
        # Full-pipeline fallback: --data must be provided
        if not args.data:
            p.error("--data is required when --corpus is not set")
        metrics = run_single_process(args)

    if metrics is not None and _is_leader():  # rank 0 (or single-process)
        metrics["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(metrics) + "\n")
        print("[bench]", json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
