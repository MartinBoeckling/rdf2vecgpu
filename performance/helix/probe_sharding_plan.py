"""What does torchrec's planner do with the Wikidata tables? — one GPU, no training.

Builds the skip-gram model on the meta device at the real vocab size and asks
``EmbeddingShardingPlanner`` for a plan at ``--world-size`` ranks, for several
``hbm_cap`` values: the planner's default (``None`` → ``HBM_CAP`` = 32 GB, what
the trainer has always used) and the real per-GPU memory of the gpu8 nodes
(A100-80GB, H200-141GB). Prints, per table, the compute kernel and the
planner's per-rank HBM / DDR storage estimate.

Why (2026-09-25): a full-Wikidata d100 rank block is 2 × 19.6 GB = 39 GB of
tables. If the planner believes a GPU holds 32 GB it cannot place that as a
FUSED (device-resident) kernel and falls back to a UVM kernel — the tables
then live in pinned HOST memory with a small GPU cache. That would explain
three things at once: the 28.5 GiB GPU peak (below the table size), the
~140 GB host RSS per rank, and A100 steps running at 48/s vs 80/s on H200
(cache misses cross PCIe). Needs a GPU only because importing torchrec pulls
in fbgemm, which needs libcuda.

  python performance/helix/probe_sharding_plan.py --vocab-size 392910288 --dim 100 --world-size 8
"""
from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-size", type=int, default=392_910_288)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--hbm-gb", type=float, nargs="*", default=[0, 40, 80, 141],
                    help="hbm_cap values to try in GB; 0 = planner default (None)")
    ap.add_argument("--ddr-gb", type=float, default=0, help="ddr_cap in GB; 0 = planner default")
    ap.add_argument("--verbose-stats", action="store_true", help="let the planner log its full stats table")
    args = ap.parse_args()

    import torch
    from torchrec.distributed.embedding import EmbeddingCollectionSharder
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.types import ShardingType

    from rdf2vecgpu.embedders.torchrec_word2vec import ShardedSkipGramModel

    if args.verbose_stats:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger("torchrec.distributed.planner.stats").setLevel(logging.INFO)

    dev = torch.cuda.get_device_properties(0)
    print(f"[plan] probe GPU: {dev.name}, {dev.total_memory / 2**30:.1f} GiB; torchrec planner default HBM_CAP=32 GB", flush=True)
    model = ShardedSkipGramModel(vocab_size=args.vocab_size, embedding_dim=args.dim, device=torch.device("meta"))
    table_gb = args.vocab_size * args.dim * 4 / 1e9
    print(f"[plan] vocab={args.vocab_size:,} dim={args.dim}: {table_gb:.1f} GB per table, "
          f"{2 * table_gb / args.world_size:.1f} GB per rank for both tables at world_size={args.world_size}", flush=True)

    constraints = {
        cfg.name: ParameterConstraints(sharding_types=[ShardingType.ROW_WISE.value])
        for m in model.modules() if hasattr(m, "embedding_configs")
        for cfg in m.embedding_configs()
    }
    for hbm in args.hbm_gb:
        hbm_cap = int(hbm * 1024**3) if hbm else None
        ddr_cap = int(args.ddr_gb * 1024**3) if args.ddr_gb else None
        topo = Topology(world_size=args.world_size, local_world_size=args.world_size,
                        compute_device="cuda", hbm_cap=hbm_cap, ddr_cap=ddr_cap)
        planner = EmbeddingShardingPlanner(topology=topo, constraints=constraints)
        label = f"hbm_cap={'default(32 GB)' if hbm_cap is None else f'{hbm:g} GB'}"
        try:
            plan = planner.plan(model, sharders=[EmbeddingCollectionSharder()])
        except Exception as exc:  # noqa: BLE001 — the planner raises PlannerError when nothing fits
            print(f"[plan] {label}: PLANNER FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}", flush=True)
            continue
        for module_path, tables in plan.plan.items():
            for name, ps in tables.items():
                print(f"[plan] {label}: {module_path}/{name}: sharding={ps.sharding_type} "
                      f"kernel={ps.compute_kernel} ranks={ps.ranks}", flush=True)
        # per-rank storage the planner budgeted for the chosen kernels
        try:
            best = planner._best_plan  # list[ShardingOption]
            for so in best:
                hbm_r = [s.storage.hbm / 2**30 for s in so.shards]
                ddr_r = [s.storage.ddr / 2**30 for s in so.shards]
                print(f"[plan] {label}: {so.name}: kernel={so.compute_kernel} "
                      f"hbm/rank={min(hbm_r):.1f}..{max(hbm_r):.1f} GiB ddr/rank={min(ddr_r):.1f}..{max(ddr_r):.1f} GiB "
                      f"cache_load_factor={so.cache_load_factor}", flush=True)
        except Exception as exc:  # noqa: BLE001 — private attribute; informational only
            print(f"[plan] {label}: (no storage breakdown: {type(exc).__name__}: {exc})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
