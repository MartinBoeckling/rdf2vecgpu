"""End-to-end runtime benchmark of RDF2Vec frameworks on one machine.

Runs every (framework, dataset, walk configuration, repeat) one at a time, each
in a fresh process in the framework's own environment, and appends one JSON line
per run to ``runs.jsonl``. Re-running with the same output directory resumes.

On Slurm, ``--write-plan`` lists the runs per framework and ``--only ID`` runs a
single one, so every run can be its own job. ``--precheck``
decides with the system Python, before anything is copied, whether that run is
already recorded or blocked by an earlier timeout.

A framework that times out or runs out of memory is not started again on any run
that is at least as large in every dimension (same graph family, no fewer
triples, walks and depth). Such runs are recorded as ``skipped``.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

RUNNER = Path(__file__).resolve().with_name("benchmark_runner.py")

# Fewest walks and depth of the ISWC 2025 grid: gives the CPU baselines the best
# chance of finishing on every graph.
REFERENCE_WALKS, REFERENCE_DEPTH, EPOCHS = 100, 4, 5
SWEEP_WALKS = (100, 500, 1000)
SWEEP_DEPTHS = (4, 8, 12)
SWEEP_DATASET = "fb15k-237"

FRAMEWORKS = {
    "gpu-random-pytorch": "wheel",
    "gpu-random-gensim": "wheel",
    "gpu-bfs-pytorch": "wheel",
    "jrdf2vec": "jrdf2vec",
    "pyrdf2vec": "pyrdf2vec",
    "spark-bfs": "spark",
}
# BFS expands every path, so these two have no walks-per-vertex setting.
IGNORES_WALK_NUMBER = {"gpu-bfs-pytorch", "spark-bfs"}
USES_GPU = {"gpu-random-pytorch", "gpu-random-gensim", "gpu-bfs-pytorch"}

SYNTHETIC = [f"{kind}_graph_{size}" for kind in ("barabasi", "erdos_renyi", "random") for size in (100, 1000, 10000)]
OOM_PATTERN = re.compile(
    r"MemoryError|out of memory|OutOfMemoryError|bad_alloc|Cannot allocate memory", re.IGNORECASE
)


@dataclass(frozen=True)
class Dataset:
    name: str
    family: str
    parquet: Path
    nt: Path


def datasets(root):
    found = [
        Dataset("fb15k-237", "real", root / "fb15k_kg.parquet", root / "fb15k_kg.nt"),
        Dataset("wikidata5m", "real", root / "wikidata5m_kg.parquet", root / "wikidata5m_kg.nt"),
    ]
    for name in SYNTHETIC:
        found.append(Dataset(name, name.split("_graph")[0], root / f"{name}.parquet", root / f"{name}.nt"))
    return found


def describe(dataset):
    import pyarrow
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(dataset.parquet, columns=["subject", "predicate", "object"])
    vertices = pc.unique(
        pc.cast(
            pyarrow.concat_arrays(
                [c.combine_chunks().cast("string") for c in (table["subject"], table["object"])]
            ),
            "string",
        )
    )
    predicates = pc.unique(table["predicate"].cast("string"))
    return {
        "triples": table.num_rows,
        "vertices": len(vertices),
        "predicates": len(predicates),
        "subjects": len(pc.unique(table["subject"].cast("string"))),
    }


def plan(names, stats, frameworks, repeats, sweep):
    """List runs: every repeat of the smallest graphs first, cheap to expensive."""
    runs = []
    ordered = sorted(names, key=lambda d: stats[d.name]["triples"])
    for repeat in range(repeats):
        for dataset in ordered:
            configs = [(REFERENCE_WALKS, REFERENCE_DEPTH)]
            if sweep and dataset.name == SWEEP_DATASET:
                configs += [
                    (w, d) for d in SWEEP_DEPTHS for w in SWEEP_WALKS if (w, d) != configs[0]
                ]
            for walks, depth in configs:
                for framework in frameworks:
                    run_walks = None if framework in IGNORES_WALK_NUMBER else walks
                    if run_walks is None and walks != REFERENCE_WALKS:
                        continue
                    runs.append((repeat, dataset, framework, run_walks, depth))
    return runs


def run_id(repeat, dataset, framework, walks, depth):
    return f"{framework}__{dataset.name}__w{walks}__d{depth}__r{repeat}"


def blocked_by(record, dataset, framework, walks, depth, stats):
    return (
        record["status"] in ("timeout", "oom")
        and record["framework"] == framework
        and record["family"] == dataset.family
        and stats[dataset.name]["triples"] >= record["triples"]
        and (walks or 0) >= (record["walks"] or 0)
        and depth >= record["depth"]
    )


class Monitor(threading.Thread):
    """Samples peak memory of a process session, in RAM and on the GPU.

    RAM is the summed proportional set size (PSS): forked workers share pages,
    and summing their RSS counted those pages once per worker (pyRDF2Vec
    reported 750 GB on a 380 GB node). GPU memory counts only this session's
    processes, so a second job on the node's other GPU doesn't show up.
    """

    def __init__(self, session, gpu):
        super().__init__(daemon=True)
        self.session, self.gpu = session, gpu
        self.stop_event = threading.Event()
        self.peak_rss_gb = self.peak_gpu_gb = 0.0

    def _session_pids(self):
        pids = set()
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/stat") as handle:
                    fields = handle.read().rsplit(")", 1)[1].split()
                if int(fields[3]) == self.session:
                    pids.add(int(entry.name))
            except (OSError, IndexError, ValueError):
                continue
        return pids

    @staticmethod
    def _pss_gb(pids):
        total_kb = 0
        for pid in pids:
            try:
                with open(f"/proc/{pid}/smaps_rollup") as handle:
                    for line in handle:
                        if line.startswith("Pss:"):
                            total_kb += int(line.split()[1])
                            break
            except (OSError, IndexError, ValueError):
                continue
        return total_kb * 1024 / 1e9

    @staticmethod
    def _gpu_gb(pids):
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        total_mib = 0.0
        for line in output.splitlines():
            try:
                pid, used = (field.strip() for field in line.split(","))
                if int(pid) in pids:
                    total_mib += float(used)
            except ValueError:
                continue
        return total_mib / 1024

    def _sample_gpu(self):
        # Own thread on a fixed 1 s schedule: nvidia-smi can take most of a
        # second on an 8-GPU node, and sharing the RAM loop stretched the gap
        # until short GPU phases (gensim on 100-vertex graphs) read 0.
        start = time.monotonic()
        ticks = 1
        while not self.stop_event.wait(max(0.0, start + ticks - time.monotonic())):
            self.peak_gpu_gb = max(self.peak_gpu_gb, self._gpu_gb(self._session_pids()))
            ticks = max(ticks + 1, int(time.monotonic() - start) + 1)

    def run(self):
        gpu_thread = threading.Thread(target=self._sample_gpu, daemon=True) if self.gpu else None
        if gpu_thread:
            gpu_thread.start()
        while not self.stop_event.wait(1):
            self.peak_rss_gb = max(self.peak_rss_gb, self._pss_gb(self._session_pids()))
        if gpu_thread:
            gpu_thread.join()


def environment(framework, args):
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
        # Slurm sets the job's own GPU; outside Slurm, use the first one.
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0") if framework in USES_GPU else "",
    }
    if framework == "spark-bfs":
        prefix = Path(args.spark_python).parent.parent
        env.update(
            JAVA_HOME=str(prefix),
            PYSPARK_PYTHON=args.spark_python,
            PYSPARK_DRIVER_PYTHON=args.spark_python,
            PATH=f"{prefix}/bin:/usr/bin:/bin",
        )
    return env


def execute(repeat, dataset, framework, walks, depth, stats, args, log_name=None):
    python = {
        "wheel": args.wheel_python,
        "jrdf2vec": args.jrdf2vec_python,
        "pyrdf2vec": args.pyrdf2vec_python,
        "spark": args.spark_python,
    }[FRAMEWORKS[framework]]
    identifier = run_id(repeat, dataset, framework, walks, depth)
    workdir = args.scratch / identifier
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    env = {**environment(framework, args), "TMPDIR": str(workdir)}
    result_path = workdir / "result.json"
    command = [
        python, "-I", str(RUNNER),
        "--framework", framework,
        "--parquet", str(dataset.parquet),
        "--nt", str(dataset.nt),
        "--depth", str(depth),
        "--epochs", str(EPOCHS),
        "--seed", str(42 + repeat),
        "--result", str(result_path),
        "--jar", str(args.jar),
        "--java", args.java,
        "--java-heap", args.java_heap,
    ]
    if walks is not None:
        command += ["--walks", str(walks)]
    log_path = workdir / "log.txt"

    def be_first_oom_victim():
        with open("/proc/self/oom_score_adj", "w") as handle:
            handle.write("1000")

    start = time.perf_counter()
    with open(log_path, "wb") as log:
        process = subprocess.Popen(
            command, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=be_first_oom_victim,
        )
        monitor = Monitor(process.pid, framework in USES_GPU)
        monitor.start()
        try:
            code = process.wait(timeout=args.timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    wall = time.perf_counter() - start
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    monitor.stop_event.set()
    monitor.join()

    tail = log_path.read_bytes()[-args.log_tail_bytes:]
    text = tail.decode("utf-8", "replace")
    outcome = json.loads(result_path.read_text()) if result_path.exists() else {}
    if timed_out:
        status = "timeout"
    elif code == 0 and outcome:
        status = "success" if outcome.get("usable") and outcome.get("dimension") == 100 else "invalid"
    elif code in (-9, 137) or OOM_PATTERN.search(text):
        status = "oom"
    else:
        status = "failed"

    saved_log = args.output / "logs" / (log_name or f"{identifier}.log")
    saved_log.parent.mkdir(parents=True, exist_ok=True)
    saved_log.write_bytes(tail)
    shutil.rmtree(workdir, ignore_errors=True)
    # let the GPU driver and the page cache settle before the next run
    time.sleep(args.settle_seconds)
    return {
        "id": identifier,
        "framework": framework,
        "dataset": dataset.name,
        "family": dataset.family,
        "triples": stats[dataset.name]["triples"],
        "tokens": stats[dataset.name]["vertices"] + stats[dataset.name]["predicates"],
        "walks": walks,
        "depth": depth,
        "epochs": EPOCHS,
        "repeat": repeat,
        "status": status,
        "exit_code": code,
        "wall_seconds": round(wall, 3),
        "peak_rss_gb": round(monitor.peak_rss_gb, 3),
        "peak_gpu_gb": round(monitor.peak_gpu_gb, 3),
        "node": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        **outcome,
    }


def load_records(path):
    """Read runs.jsonl; a line another job is still writing is ignored."""
    if not path.exists():
        return []
    records = []
    with open(path) as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        lines = handle.read().splitlines()
        fcntl.flock(handle, fcntl.LOCK_UN)
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_record(path, record):
    """Append one line under an exclusive lock: several jobs share the file."""
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def skipped_record(identifier, repeat, dataset, framework, walks, depth, stats):
    return {
        "id": identifier, "framework": framework, "dataset": dataset.name,
        "family": dataset.family, "triples": stats[dataset.name]["triples"],
        "walks": walks, "depth": depth, "epochs": EPOCHS, "repeat": repeat,
        "status": "skipped",
    }


def warm_up(framework, smallest, stats, args, log_name):
    # The first start of each environment reads it from a cold disk; pay for
    # that in a discarded run so it doesn't land on a measured one.
    walks = None if framework in IGNORES_WALK_NUMBER else REFERENCE_WALKS
    warmup = execute(0, smallest, framework, walks, REFERENCE_DEPTH, stats, args, log_name=log_name)
    print(f"warm-up {framework}: {warmup['status']} {warmup['wall_seconds']} s", flush=True)


NEEDS_RUN = 10  # --precheck exit code: the run has to be executed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="directory holding all parquet and nt files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, help="fast local directory for run output")
    parser.add_argument("--wheel-python")
    parser.add_argument("--jrdf2vec-python")
    parser.add_argument("--pyrdf2vec-python")
    parser.add_argument("--spark-python")
    parser.add_argument("--jar", type=Path)
    parser.add_argument("--java", help="Java used for jRDF2Vec")
    parser.add_argument("--java-heap", default="250g")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--frameworks", nargs="+", default=list(FRAMEWORKS), choices=list(FRAMEWORKS))
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--log-tail-bytes", type=int, default=65536)
    parser.add_argument("--write-plan", action="store_true",
                        help="write plan/<framework>.txt (run ids in order) and plan/warmup_dataset.txt, then exit")
    parser.add_argument("--only", metavar="ID", help="run just this planned run")
    parser.add_argument("--precheck", action="store_true",
                        help=f"with --only: record a skip if blocked; exit 0 if nothing to run, {NEEDS_RUN} if it must run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    every = datasets(args.data)
    stats_path = args.output / "datasets.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    missing = [d for d in every if d.name not in stats]
    if missing:
        if args.precheck:
            parser.error("--precheck needs datasets.json; run --write-plan first")
        for dataset in missing:
            stats[dataset.name] = describe(dataset)
        stats_path.write_text(json.dumps(stats, indent=2))

    chosen = [d for d in every if args.max_triples is None or stats[d.name]["triples"] <= args.max_triples]
    smallest = min(chosen, key=lambda d: stats[d.name]["triples"])
    results_path = args.output / "runs.jsonl"
    todo = plan(chosen, stats, args.frameworks, args.repeats, not args.no_sweep)

    if args.write_plan:
        plan_dir = args.output / "plan"
        plan_dir.mkdir(exist_ok=True)
        (plan_dir / "warmup_dataset.txt").write_text(smallest.name + "\n")
        for framework in args.frameworks:
            ids = [run_id(*entry[:2], framework, *entry[3:]) for entry in todo if entry[2] == framework]
            (plan_dir / f"{framework}.txt").write_text("".join(i + "\n" for i in ids))
            print(f"{framework}: {len(ids)} runs", flush=True)
        return

    if args.precheck and not args.only:
        parser.error("--precheck needs --only")
    if not args.precheck:
        absent = [name for name in ("scratch", "wheel_python", "jrdf2vec_python", "pyrdf2vec_python",
                                    "spark_python", "jar", "java") if getattr(args, name) is None]
        if absent:
            parser.error("running needs --" + ", --".join(a.replace("_", "-") for a in absent))

    if args.only:
        entries = [e for e in todo if run_id(*e) == args.only]
        if not entries:
            parser.error(f"{args.only} is not in the plan")
        repeat, dataset, framework, walks, depth = entries[0]
        records = load_records(results_path)
        if any(r["id"] == args.only for r in records):
            print(f"{args.only}: already recorded", flush=True)
            sys.exit(0)
        if any(blocked_by(r, dataset, framework, walks, depth, stats) for r in records):
            append_record(results_path, skipped_record(args.only, repeat, dataset, framework, walks, depth, stats))
            print(f"{args.only}: skipped", flush=True)
            sys.exit(0)
        if args.precheck:
            sys.exit(NEEDS_RUN)
        warm_up(framework, smallest, stats, args, log_name=f"warmup/{args.only}.log")
        record = execute(repeat, dataset, framework, walks, depth, stats, args)
        append_record(results_path, record)
        print(f"{args.only}: {record['status']} {record['wall_seconds']} s "
              f"vectors={record.get('n_vectors', '-')}", flush=True)
        return

    records = load_records(results_path)
    finished = {record["id"] for record in records}
    print(f"{len(todo)} runs planned, {len(finished)} already recorded", flush=True)
    for framework in args.frameworks:
        warm_up(framework, smallest, stats, args, log_name=f"warmup/{framework}.log")
    for position, (repeat, dataset, framework, walks, depth) in enumerate(todo, 1):
        identifier = run_id(repeat, dataset, framework, walks, depth)
        if identifier in finished:
            continue
        if any(blocked_by(r, dataset, framework, walks, depth, stats) for r in records):
            record = skipped_record(identifier, repeat, dataset, framework, walks, depth, stats)
        else:
            record = execute(repeat, dataset, framework, walks, depth, stats, args)
        records.append(record)
        append_record(results_path, record)
        print(
            f"[{position}/{len(todo)}] {identifier}: {record['status']} "
            f"{record.get('wall_seconds', '-')} s vectors={record.get('n_vectors', '-')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
