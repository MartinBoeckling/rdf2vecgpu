"""Collect the metrics of torchrec runs on Helix into a small bundle of CSV files.

No experiment tracker runs with these jobs; the metrics live in three places —
the job logs (progress lines every 200 steps, per-rank checkpoint lines), the
benchmark's ``run.jsonl`` (when the job reached it) and Slurm accounting. This
script extracts them into ``--out``:

  runs_summary.csv/.json      one row per training job
  loss_curve_<run>.csv        epoch, global step, loss, steps/s, rank-0 host RSS
  checkpoints_<run>.csv       per rank and checkpoint: RSS before / after save / after release
  run_records.csv             the benchmark's own records (``--run-jsonl``)
  cpu_export.csv              per-rank rows and seconds of a CPU export (``--cpu-export-logs``)
  dimension_probe.csv         steps/s and per-phase ms of profiled probe runs (``--probe-logs``)
  README.md                   what each file holds (``--readme-notes`` appends run-specific notes)

A run is given as ``--run NAME:JOBID:LOG[:NOTE]`` (repeatable). ``sacct`` must
be reachable (it is on compute nodes); ``git`` is not, so pass the commit in
``BUNDLE_COMMIT``.

  BUNDLE_COMMIT=$(git rev-parse --short HEAD) srun -p devel -n1 -c2 --mem=4G -t 5 --export=ALL \\
    python3 performance/helix/build_metrics_bundle.py --out $WS/results/metrics_wikidata_d200 \\
      --run "d200_e5_h200:14984051:wd_d200_e5_h200_14984051.log:5 epochs, 8x H200" ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

PROG = re.compile(r"\[torchrec\] epoch (\d+) \| step ([\d,]+) \| loss ([\d.]+) \| ([\d.]+) steps/s(?: \| rss ([\d.]+) GiB)?")
CKPT = re.compile(r"\[torchrec\] rank (\d+): checkpoint (step_\d+) written; host RSS ([\d.]+) -> ([\d.]+) -> ([\d.]+) GiB "
                  r"after release, peak ([\d.]+) -> ([\d.]+) GiB")
PLAN = re.compile(r"\[torchrec\] plan: \S+/(\w+): sharding=(\w+) kernel=(\w+)")
PHASES = re.compile(r"\[torchrec\]\[profile\] epoch (\d+): (\d+) measured steps.*?rank 0 host ms/step: (.*?) ; gpu fwd\+bwd ([\d.]+) ms")
CPUX = re.compile(r"\[ckpt-export\] rank (\d+): ([\d,]+) of ([\d,]+) block rows -> \S+ in (\d+) s")
HEADER = re.compile(r"\[gpu8\] host=(\S+) gpus=(\d+) corpus=(\S+) epochs=(\d+) batch=(\d+) lr=(\S+)")


def sacct(job: str) -> dict:
    f = "JobID,JobName,State,Start,End,Elapsed,Timelimit,NodeList,ReqMem"
    top = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "-o", f], capture_output=True, text=True).stdout.strip().split("\n")[0].split("|")
    d = dict(zip(f.split(","), top))
    tres = subprocess.run(["sacct", "-j", job + ".batch", "-n", "-P", "-o", "TRESUsageInTot"], capture_output=True, text=True).stdout.strip()
    for kv in tres.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d["usage_" + k] = v
    return d


def parse_log(path: str):
    rows, ck, plan, phases, header = [], [], {}, [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = PROG.search(line)
            if m:
                rows.append([int(m[1]), int(m[2].replace(",", "")), float(m[3]), float(m[4]), m[5] or ""])
                continue
            for m in CKPT.finditer(line):
                ck.append([int(m[1]), m[2], float(m[3]), float(m[4]), float(m[5]), float(m[6]), float(m[7])])
            m = PLAN.search(line)
            if m:
                plan[m[1]] = f"{m[2]}/{m[3]}"
            m = PHASES.search(line)
            if m:
                d = {"epoch": int(m[1]), "measured_steps": int(m[2]), "gpu_fwd_bwd_ms": float(m[4])}
                for part in m[3].split("|"):
                    k, v = part.split()
                    d[f"host_{k}_ms"] = float(v)
                phases.append(d)
            if header is None:
                m = HEADER.search(line)
                if m:
                    header = dict(host=m[1], gpus=int(m[2]), corpus=os.path.basename(m[3]), epochs=int(m[4]),
                                  pair_batch_size=int(m[5]), lr=m[6])
    return rows, ck, plan, phases, header


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[], help="NAME:JOBID:LOG[:NOTE] (repeatable)")
    ap.add_argument("--run-jsonl", action="append", default=[], help="run.jsonl files to include")
    ap.add_argument("--cpu-export-logs", default="", help="glob of export_from_checkpoint.py rank logs")
    ap.add_argument("--probe-logs", default="", help="glob of profiled/unprofiled probe logs")
    ap.add_argument("--title", default="torchrec runs on Helix — metrics bundle")
    ap.add_argument("--readme-notes", default="", help="file whose text is appended to README.md")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = {}

    summary = []
    for spec in args.run:
        name, job, log, *note = spec.split(":", 3)
        rows, ck, plan, _, header = parse_log(log)
        with open(f"{args.out}/loss_curve_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch", "global_step", "loss_sum_per_batch", "steps_per_s", "rank0_host_rss_gib"])
            w.writerows(rows)
        files[f"loss_curve_{name}.csv"] = "every 200th step: epoch, global step, loss, steps/s, rank-0 host RSS"
        if ck:
            with open(f"{args.out}/checkpoints_{name}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["rank", "checkpoint", "rss_before_gib", "rss_after_save_gib", "rss_after_release_gib", "peak_before_gib", "peak_after_gib"])
                w.writerows(sorted(ck, key=lambda x: (x[1], x[0])))
            files[f"checkpoints_{name}.csv"] = "per rank and checkpoint: host RSS before the save, after it, after release"
        rates = sorted(x[3] for x in rows if x[3] > 0)
        acc = sacct(job)
        mem_k = acc.get("usage_mem", "").rstrip("K")
        gpumem_m = acc.get("usage_gres/gpumem", "").rstrip("M")
        row = dict(
            run=name, slurm_job=job, node=acc.get("NodeList"), state=acc.get("State"), start=acc.get("Start"),
            end=acc.get("End"), wall=acc.get("Elapsed"), time_limit=acc.get("Timelimit"),
            progress_points=len(rows), first_step=rows[0][1] if rows else None, last_step=rows[-1][1] if rows else None,
            last_epoch=rows[-1][0] if rows else None,
            first_loss=rows[1][2] if len(rows) > 1 else None, last_loss=rows[-1][2] if rows else None,
            steps_per_s_mean=round(sum(rates) / len(rates), 1) if rates else None,
            steps_per_s_median=rates[len(rates) // 2] if rates else None,
            host_mem_peak_gb=round(int(mem_k) * 1024 / 1e9, 1) if mem_k.isdigit() else None,
            gpu_mem_total_gb=round(int(gpumem_m) * 1024**2 / 1e9, 1) if gpumem_m.isdigit() else None,
            gpu_util_sum_pct=acc.get("usage_gres/gpuutil"), cpu_time=acc.get("usage_cpu"),
            checkpoints_logged=len({c[1] for c in ck}),
            rank_rss_at_rest_gib=min((c[4] for c in ck), default=None),
            rank_rss_during_save_gib=max((c[3] for c in ck), default=None),
            table_plan="; ".join(f"{k}={v}" for k, v in sorted(plan.items())),
            note=note[0] if note else "",
        )
        if header:
            row.update(gpus=header["gpus"], corpus=header["corpus"], epochs_requested=header["epochs"],
                       pair_batch_size=header["pair_batch_size"], lr=header["lr"])
        summary.append(row)
    if summary:
        keys = list(dict.fromkeys(k for r in summary for k in r))
        with open(f"{args.out}/runs_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)
        json.dump(summary, open(f"{args.out}/runs_summary.json", "w"), indent=2)
        files["runs_summary.csv / .json"] = "one row per training job: wall time, step rates, host and GPU memory, loss at start and end, table placement"

    recs = []
    for p in args.run_jsonl:
        d = os.path.basename(os.path.dirname(os.path.abspath(p)))
        shutil.copy(p, f"{args.out}/run_{d}.jsonl")
        for line in open(p):
            if line.strip():
                rec = json.loads(line)
                rec["results_dir"] = d
                recs.append(rec)
    if recs:
        keys = ["results_dir"] + sorted({k for r in recs for k in r} - {"results_dir"})
        with open(f"{args.out}/run_records.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(recs)
        files["run_records.csv, run_*.jsonl"] = "the benchmark's own records: fit seconds, peak VRAM per GPU, export seconds and rows"

    if args.cpu_export_logs:
        rows = []
        for p in sorted(glob.glob(args.cpu_export_logs)):
            for line in open(p, errors="replace"):
                m = CPUX.search(line)
                if m:
                    job = re.search(r"_(\d+)\.log$", p)
                    rows.append([int(m[1]), int(m[2].replace(",", "")), int(m[3].replace(",", "")), int(m[4]), job[1] if job else ""])
        if rows:
            rows.sort()
            with open(f"{args.out}/cpu_export.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["rank", "rows_written", "block_rows", "seconds", "slurm_job"])
                w.writerows(rows)
                w.writerow(["total", sum(r[1] for r in rows), sum(r[2] for r in rows), max(r[3] for r in rows), ""])
            files["cpu_export.csv"] = "CPU export from the checkpoint: rows written and seconds per rank (total row: sum of rows, slowest rank)"

    if args.probe_logs:
        rows = []
        for p in sorted(glob.glob(args.probe_logs)):
            prog, _, plan, phases, header = parse_log(p)
            rates = sorted(x[3] for x in prog[3:] if x[3] > 0)
            vec = re.search(r'"vector_size": (\d+)', open(p, errors="replace").read())
            d = dict(log=os.path.basename(p), gpus=header["gpus"] if header else "", corpus=header["corpus"] if header else "",
                     pair_batch_size=header["pair_batch_size"] if header else "", vector_size=int(vec[1]) if vec else "",
                     profiled=bool(phases), steps_per_s_median=rates[len(rates) // 2] if rates else "")
            if phases:
                d.update({k: v for k, v in phases[-1].items() if k != "epoch"})
            rows.append(d)
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(f"{args.out}/dimension_probe.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            files["dimension_probe.csv"] = ("small probe runs at the same pairs per rank: median steps/s; for profiled runs the last "
                                            "epoch's host ms per step by phase and the GPU forward+backward ms (profiling "
                                            "synchronises the device each step, so use unprofiled rows for rates)")

    table = "\n".join(f"| `{k}` | {v} |" for k, v in files.items())
    notes = open(args.readme_notes).read() if args.readme_notes else ""
    open(f"{args.out}/README.md", "w").write(
        f"# {args.title}\n\nBuilt by `performance/helix/build_metrics_bundle.py` (repo at "
        f"{os.environ.get('BUNDLE_COMMIT', 'unknown')}) from the job logs, the benchmark's run records and Slurm "
        f"accounting. No experiment tracker was active during these runs.\n\n| file | content |\n|---|---|\n{table}\n\n"
        "Column notes\n"
        "* `loss_sum_per_batch` is the SUM over the batch on rank 0, not a mean; divide by the pairs per rank "
        "(pair batch size / GPUs) for a per-pair value.\n"
        "* `host_mem_peak_gb` is Slurm's cgroup total for the job (all ranks); `gpu_mem_total_gb` is the sum over "
        "the GPUs; `gpu_util_sum_pct` is the sum of the utilisations (800 = eight GPUs busy). Energy is not recorded.\n"
        "* `table_plan` is each table's sharding type / compute kernel (`fused` = on the GPU).\n\n" + notes)
    print("wrote", args.out)
    for f in sorted(os.listdir(args.out)):
        print(f"  {os.path.getsize(os.path.join(args.out, f)):>10,d}  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
