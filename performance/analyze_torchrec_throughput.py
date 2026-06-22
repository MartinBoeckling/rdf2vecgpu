#!/usr/bin/env python
"""Summarize torchrec training throughput across a pair_batch_size sweep.

Parses the rank-0 progress lines
    [torchrec] epoch E | step S | loss L | R steps/s
from each job log, plus sacct Elapsed, and derives steps/epoch and a speedup
factor vs the first (baseline) variant.

steps/epoch = ceil(total_walks / (pair_batch_size // pairs_per_walk)); each step
is one all-to-all collective, so wall-clock/epoch ~= steps_per_epoch / steps_per_s.
Throughput (pairs/s) should be ~flat if overhead-bound — larger batch then wins
purely by doing fewer collective steps.

Usage:
  python analyze_torchrec_throughput.py <total_walks> <pairs_per_walk> \
      "label::log::jobid::pair_batch_size" [...]
"""
import sys
import re
import math
import statistics
import subprocess

LINE = re.compile(r"epoch (\d+) \| step ([\d,]+) \| loss [\d.eE+-]+ \| ([\d.]+) steps/s")


def parse(log):
    sps, maxstep, lastep = [], 0, 0
    try:
        with open(log) as f:
            for line in f:
                m = LINE.search(line)
                if m:
                    lastep = int(m.group(1))
                    maxstep = max(maxstep, int(m.group(2).replace(",", "")))
                    sps.append(float(m.group(3)))
    except FileNotFoundError:
        pass
    steady = sps[5:] if len(sps) > 12 else sps      # drop warmup points
    med = round(statistics.median(steady), 1) if steady else 0.0
    return med, maxstep, lastep


def elapsed(jobid):
    try:
        out = subprocess.run(["sacct", "-j", jobid, "--format=Elapsed", "-n"],
                             capture_output=True, text=True, timeout=20).stdout
        return out.strip().split("\n")[0].strip()
    except Exception:
        return "?"


def main():
    total_walks = int(sys.argv[1])
    ppw = int(sys.argv[2])
    rows = []
    for a in sys.argv[3:]:
        label, log, jobid, pbs = a.split("::")
        pbs = int(pbs)
        med, maxstep, lastep = parse(log)
        wpb = max(1, pbs // ppw)
        spe = math.ceil(total_walks / wpb)
        sec_epoch = spe / med if med else float("nan")
        pairs_s = med * pbs                       # global pairs/sec
        rows.append(dict(label=label, pbs=pbs, med=med, spe=spe, maxstep=maxstep,
                         lastep=lastep, el=elapsed(jobid),
                         sec_epoch=sec_epoch, pairs_s=pairs_s))

    base = rows[0]["sec_epoch"] if rows else float("nan")
    hdr = (f"{'variant':<20}{'pair_batch':>11}{'steps/epoch':>12}"
           f"{'med st/s':>10}{'pairs/s':>12}{'s/epoch':>10}{'speedup':>9}{'elapsed':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sp = base / r["sec_epoch"] if r["sec_epoch"] and r["sec_epoch"] == r["sec_epoch"] else float("nan")
        print(f"{r['label']:<20}{r['pbs']:>11,}{r['spe']:>12,}{r['med']:>10}"
              f"{r['pairs_s']:>12,.0f}{r['sec_epoch']:>10,.0f}{sp:>8.2f}x{r['el']:>11}")
    print("-" * len(hdr))
    print("speedup = baseline_s/epoch / variant_s/epoch  (>1 = faster). "
          "pairs/s ~flat => overhead-bound => batch scaling pays off.")


if __name__ == "__main__":
    main()
