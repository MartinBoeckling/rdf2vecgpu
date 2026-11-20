"""Run one RDF2Vec framework once and write its timings to a JSON file.

Started by ``benchmark.py`` in a fresh process, inside the framework's own
environment, so the file only imports a framework inside its ``run_*`` function.
Every run covers the same work: load the graph, generate walks, train Word2Vec,
and write the embeddings to disk. The checks after the write are not timed.
"""
import argparse
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_COUNT = 1
NEGATIVE_SAMPLES = 5
# The cores this process may use (e.g. what Slurm gave the job), not the whole node.
CPU_COUNT = len(os.sched_getaffinity(0))
SAMPLE_SIZE = 10_000


class Stages:
    """Collects the wall-clock seconds of each named stage."""

    def __init__(self):
        self.seconds = {}

    def measure(self, name):
        return _Timer(self, name)

    def add(self, name, seconds):
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    def wrap(self, name, function):
        def timed(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)

        return timed


class _Timer:
    def __init__(self, stages, name):
        self.stages = stages
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *_):
        self.stages.add(self.name, time.perf_counter() - self.start)


def check_vectors(rows):
    """Return (dimension, usable) for a sample of embedding rows."""
    if not rows:
        return 0, False
    dimension = len(rows[0])
    values = [value for row in rows for value in row]
    usable = (
        all(len(row) == dimension for row in rows)
        and all(math.isfinite(value) for value in values)
        and any(value != 0 for value in values)
    )
    return dimension, usable


def run_gpu(args, stages, workdir):
    with stages.measure("import"):
        import rdf2vecgpu
        from lightning.pytorch.tuner import Tuner
        from rdf2vecgpu import GPU_RDF2Vec

    Tuner.scale_batch_size = stages.wrap("batch_size_tuner", Tuner.scale_batch_size)
    backend = args.framework.split("-")[-1]
    model = GPU_RDF2Vec(
        walk_strategy="bfs" if "bfs" in args.framework else "random",
        walk_depth=args.depth,
        walk_number=args.walks or 100,
        embedding_model="skipgram",
        epochs=args.epochs,
        batch_size=None,
        vector_size=VECTOR_SIZE,
        window_size=WINDOW_SIZE,
        min_count=MIN_COUNT,
        negative_samples=NEGATIVE_SAMPLES,
        learning_rate=0.025,   # word2vec's and jRDF2Vec's alpha; the default since 0.6.0
        backend=backend,
        random_state=args.seed,
        multi_gpu=False,
        generate_artifact=False,
        cpu_count=CPU_COUNT,
    )
    for method, stage in (
        ("walk_generation", "walks"),
        ("walk_generation_sequences", "walks"),
        ("_fit_from_tensors", "train"),
        ("_fit_via_pytorch_sequences", "train"),   # 0.6.0: PyTorch on walk sequences
        ("_fit_via_gensim", "train"),
    ):
        if hasattr(model, method):   # each version has only some of these
            setattr(model, method, stages.wrap(stage, getattr(model, method)))

    with stages.measure("load"):
        edges = model.load_data(args.parquet)
    model.fit(edges)
    with stages.measure("transform"):
        embeddings = model.transform()
    with stages.measure("write"):
        embeddings.to_parquet(workdir / "embeddings.parquet")

    floats = embeddings.select_dtypes("floating").head(SAMPLE_SIZE).to_pandas()
    dimension, usable = check_vectors(floats.values.tolist())
    return {
        "n_vectors": int(len(embeddings)),
        "dimension": dimension,
        "usable": usable,
        "version": rdf2vecgpu.__version__,
        "module_file": rdf2vecgpu.__file__,
    }


def run_jrdf2vec(args, stages, workdir):
    # jRDF2Vec starts its gensim sidecar from the python found on PATH, which
    # is this environment's python.
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    walk_dir = workdir / "walks"
    command = [
        args.java, f"-Xmx{args.java_heap}", "-jar", args.jar,
        "-graph", args.nt,
        "-walkDirectory", str(walk_dir),
        "-numberOfWalks", str(args.walks),
        "-depth", str(args.depth),
        "-walkGenerationMode", "RANDOM_WALKS",
        "-threads", str(CPU_COUNT),
        "-trainingMode", "sg",
        "-window", str(WINDOW_SIZE),
        "-minCount", str(MIN_COUNT),
        "-dimension", str(VECTOR_SIZE),
        "-epochs", str(args.epochs),
        "-noVectorTextFileGeneration",
    ]
    start = time.perf_counter()
    training_started = None
    process = subprocess.Popen(
        command, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(f"[{time.perf_counter() - start:10.2f}] {line}", end="", flush=True)
        if training_started is None and re.search(args.jrdf2vec_train_marker, line):
            training_started = time.perf_counter() - start
    process.wait()
    total = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(f"jRDF2Vec exited with code {process.returncode}")
    if training_started is None:
        stages.add("walks_and_train", total)
    else:
        stages.add("load_and_walks", training_started)
        stages.add("train_and_write", total - training_started)

    from gensim.models import KeyedVectors

    model_files = sorted(walk_dir.rglob("model.kv"))
    if not model_files:
        raise RuntimeError("jRDF2Vec exited 0 but wrote no model.kv")
    vectors = KeyedVectors.load(str(model_files[0]), mmap="r")
    dimension, usable = check_vectors(vectors.vectors[:SAMPLE_SIZE].tolist())
    return {"n_vectors": len(vectors), "dimension": dimension, "usable": usable}


def run_pyrdf2vec(args, stages, workdir):
    with stages.measure("import"):
        import pandas as pd
        from pyrdf2vec import RDF2VecTransformer
        from pyrdf2vec.embedders import Word2Vec
        from pyrdf2vec.graphs import KG, Vertex
        from pyrdf2vec.walkers import RandomWalker

    with stages.measure("load"):
        frame = pd.read_parquet(args.parquet, columns=["subject", "predicate", "object"])
        graph = KG()
        entities = set()
        for subject, predicate, object_ in frame.astype(str).itertuples(index=False):
            source = Vertex(subject)
            target = Vertex(object_)
            entities.update((subject, object_))
            graph.add_walk(
                source, Vertex(predicate, predicate=True, vprev=source, vnext=target), target
            )
        entities = list(entities)

    transformer = RDF2VecTransformer(
        Word2Vec(
            epochs=args.epochs,
            workers=CPU_COUNT,
            window=WINDOW_SIZE,
            min_count=MIN_COUNT,
            vector_size=VECTOR_SIZE,
            negative=NEGATIVE_SAMPLES,
        ),
        walkers=[
            RandomWalker(
                max_walks=args.walks,
                max_depth=args.depth,
                random_state=args.seed,
                n_jobs=CPU_COUNT,
            )
        ],
    )
    with stages.measure("walks"):
        walks = transformer.get_walks(graph, entities)
    with stages.measure("train"):
        transformer.fit(walks)
    with stages.measure("transform"):
        embeddings, _ = transformer.transform(graph, entities)
    with stages.measure("write"):
        with open(workdir / "embeddings.pkl", "wb") as handle:
            pickle.dump(embeddings, handle)

    dimension, usable = check_vectors([list(map(float, row)) for row in embeddings[:SAMPLE_SIZE]])
    # pyRDF2Vec removes duplicate walks per entity, so its corpus can be far
    # smaller than walks * entities; recorded to make that visible.
    return {
        "n_vectors": len(embeddings),
        "dimension": dimension,
        "usable": usable,
        "n_walks": sum(len(entity_walks) for entity_walks in walks),
    }


def run_spark(args, stages, workdir):
    with stages.measure("import"):
        from pyspark.sql import SparkSession
        from sparkkgml.kg import KG
        from sparkkgml.motif_walks import MotifWalks

    with stages.measure("spark_startup"):
        spark = (
            SparkSession.builder.master(f"local[{CPU_COUNT}]")
            .appName("rdf2vec-benchmark")
            .config("spark.driver.memory", args.java_heap)
            .config("spark.local.dir", str(workdir / "spark_local"))
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12")
            .getOrCreate()
        )
    with stages.measure("load"):
        kg = KG(location=args.nt, fmt="nt", sparkSession=spark)
        graph_frame = kg.createKG()
        entities = list(kg.vertex_to_key_hashMap.keys())
        walker = MotifWalks(kg_instance=kg, entities=entities, sparkSession=spark)
    output = workdir / "embeddings"
    # Spark evaluates lazily: the walks run inside the fit and the write.
    with stages.measure("walks_and_train"):
        paths = walker.motif_walk(graph_frame, depth=args.depth, walktype="BFS")
        vectors = walker.word2Vec_embeddings(
            df=paths,
            vector_size=VECTOR_SIZE,
            min_count=MIN_COUNT,
            num_partitions=CPU_COUNT * 4,
            step_size=0.025,
            max_iter=args.epochs,
            seed=args.seed,
            input_col="paths",
            output_col="vectors",
            window_size=WINDOW_SIZE,
            max_sentence_length=1_000_000,
        )
    with stages.measure("write"):
        vectors.write.parquet(str(output), mode="overwrite")

    written = spark.read.parquet(str(output))
    sample = [row["vector"].toArray().tolist() for row in written.select("vector").limit(1000).collect()]
    dimension, usable = check_vectors(sample)
    result = {"n_vectors": written.count(), "dimension": dimension, "usable": usable}
    spark.stop()
    return result


RUNNERS = {
    "gpu-random-pytorch": run_gpu,
    "gpu-random-gensim": run_gpu,
    "gpu-bfs-pytorch": run_gpu,
    "jrdf2vec": run_jrdf2vec,
    "pyrdf2vec": run_pyrdf2vec,
    "spark-bfs": run_spark,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--nt", required=True)
    parser.add_argument("--walks", type=int, default=None)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--jar")
    parser.add_argument("--java", default="java")
    parser.add_argument("--java-heap", default="250g")
    parser.add_argument("--jrdf2vec-train-marker", default=r"Walks successfully generated")
    args = parser.parse_args()

    stages = Stages()
    workdir = Path.cwd()
    outcome = RUNNERS[args.framework](args, stages, workdir)
    Path(args.result).write_text(json.dumps({"stages": stages.seconds, "threads": CPU_COUNT, **outcome}))
    # Skip interpreter teardown: GPU and JVM handles can hang on exit, and the
    # result is already on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
