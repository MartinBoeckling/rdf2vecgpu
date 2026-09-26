"""Which checkpoint-writer setting keeps host memory flat across saves? — 1 GPU.

Saves the same device tensors ``--saves`` times per variant and prints this
process's RSS after each save. Each variant runs in a fresh subprocess so
allocator caches do not carry over. Written 2026-09-26: with torch's default
writer every ``dcp.save`` grew host RSS by 2 x (table rounded to a power of
two) and with ``per_thread_copy_ahead=0`` still by exactly the table bytes
(jobs 14919861/2) — the staged host copy is retained either way.

Variants (``--variants`` to pick):
  default        FileSystemWriter()                      (overlapping loader, pinned)
  ahead0         per_thread_copy_ahead=0                 (serial loader, pageable)
  ahead0_gc      ahead0 + gc.collect() + malloc_trim(0) after each save
  threads2       thread_count=2                          (serial loader in threads)
  cpu_stage      manual .cpu() copy, dcp.save of the CPU dict, del, gc, malloc_trim
  torch_save     torch.save(tensor.cpu()) to a file, del, gc, malloc_trim  (no DCP)

Plain tensors, ``no_dist=True`` — no process group, no torchrec:
  python performance/helix/probe_dcp_memory.py --gib 2.35
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time

VARIANTS = ["default", "ahead0", "ahead0_gc", "threads2", "cpu_stage", "torch_save"]


def _rss_gib() -> float:
    return _mem()["VmRSS"]


def _mem() -> dict:
    """VmRSS and its anonymous / file-backed parts, GiB (from /proc/self/status)."""
    out = {"VmRSS": float("nan"), "RssAnon": float("nan"), "RssFile": float("nan")}
    with open("/proc/self/status") as fh:
        for line in fh:
            key = line.split(":")[0]
            if key in out:
                out[key] = int(line.split()[1]) / 2**20
    return out


def _mem_str() -> str:
    m = _mem()
    return f"rss {m['VmRSS']:.2f} GiB (anon {m['RssAnon']:.2f}, file {m['RssFile']:.2f})"


def _trim() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def run_variant(variant: str, gib: float, tables: int, saves: int, root: str) -> None:
    import torch
    import torch.distributed.checkpoint as dcp

    torch.cuda.set_device(0)
    n = int(gib * 2**30 / 4)
    state = {f"t{i}": torch.rand(n, device="cuda") for i in range(tables)}
    torch.cuda.synchronize()
    print(f"[dcp] {variant}: {tables} x {gib} GiB device tensors; before saves {_mem_str()}", flush=True)
    for k in range(saves):
        d = os.path.join(root, f"{variant}_{k}")
        t0 = time.monotonic()
        if variant == "default":
            dcp.save(state, storage_writer=dcp.FileSystemWriter(d), no_dist=True)
        elif variant == "ahead0":
            dcp.save(state, storage_writer=dcp.FileSystemWriter(d, per_thread_copy_ahead=0), no_dist=True)
        elif variant == "ahead0_gc":
            dcp.save(state, storage_writer=dcp.FileSystemWriter(d, per_thread_copy_ahead=0), no_dist=True)
            _trim()
        elif variant == "threads2":
            dcp.save(state, storage_writer=dcp.FileSystemWriter(d, thread_count=2), no_dist=True)
        elif variant == "cpu_stage":
            cpu = {key: v.cpu() for key, v in state.items()}
            dcp.save(cpu, storage_writer=dcp.FileSystemWriter(d, per_thread_copy_ahead=0), no_dist=True)
            del cpu
            _trim()
        elif variant == "torch_save":
            os.makedirs(d, exist_ok=True)
            for key, v in state.items():
                c = v.cpu()
                torch.save(c, os.path.join(d, key + ".pt"))
                del c
            _trim()
        else:
            raise SystemExit(f"unknown variant {variant}")
        dt = time.monotonic() - t0
        print(f"[dcp] {variant}: save {k + 1}/{saves} {dt:.1f} s -> {_mem_str()}", flush=True)
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=2.35, help="size of each device tensor in GiB")
    ap.add_argument("--tables", type=int, default=2)
    ap.add_argument("--saves", type=int, default=4)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--root", default=None, help="scratch dir for the checkpoint files")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    root = args.root or tempfile.mkdtemp(prefix="dcp_probe_")
    if args._child:
        run_variant(args._child, args.gib, args.tables, args.saves, root)
        return 0
    for v in args.variants:
        cmd = [sys.executable, __file__, "--gib", str(args.gib), "--tables", str(args.tables),
               "--saves", str(args.saves), "--root", root, "--_child", v]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[dcp] {v}: child exited {rc}", flush=True)
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
