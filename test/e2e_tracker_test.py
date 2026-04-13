"""E2E test: tracker integration (mlflow + wandb) on a small graph.

Run with:  python test/e2e_tracker_test.py [mlflow|wandb]
"""
import sys
import tempfile
import os

from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

DATA = "data/generated_graphs/barabasi_graph_100.parquet"
tracker_name = sys.argv[1] if len(sys.argv) > 1 else "mlflow"

print(f"=== Testing tracker: {tracker_name} ===")

if tracker_name == "mlflow":
    tmpdir = tempfile.mkdtemp(prefix="mlflow_test_")
    tracker_kwargs = {
        "mlflow": {
            "experiment": "rdf2vec_test",
            "tracking_uri": f"file://{tmpdir}",
        }
    }
elif tracker_name == "wandb":
    # Use offline mode so no account/network needed
    os.environ["WANDB_MODE"] = "offline"
    tracker_kwargs = {
        "wandb": {
            "project": "rdf2vec-test",
        }
    }
else:
    print(f"Unknown tracker: {tracker_name}")
    sys.exit(1)

config = RDF2VecConfig(
    walk_strategy="random",
    walk_depth=4,
    walk_number=10,
    embedding_model="skipgram",
    epochs=2,
    vector_size=64,
    window_size=3,
    min_count=1,
    negative_samples=5,
    learning_rate=0.001,
    random_state=42,
    tune_batch_size=False,
    batch_size=512,
    tracker=tracker_name,
    tracker_kwargs=tracker_kwargs,
    tracker_run_name="e2e_test_run",
)

with GPU_RDF2Vec(config) as model:
    edges = model.load_data(DATA)
    print(f"load_data OK: {edges.shape}")

    model.fit(edges)
    print("fit OK")

    emb = model.transform()
    print(f"transform OK: {emb.shape}")

print(f"\n=== {tracker_name.upper()} TRACKER E2E PASSED ===")
