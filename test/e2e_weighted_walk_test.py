"""E2E test: weighted random walks (single-GPU).

Creates a weighted version of the test graph, then runs the full
load_data -> fit -> transform pipeline with walk_weighted=True.
"""
import cudf
import numpy as np

DATA_UNWEIGHTED = "data/generated_graphs/barabasi_graph_100.parquet"
DATA_WEIGHTED = "data/generated_graphs/barabasi_graph_100_weighted.parquet"


def create_weighted_data():
    """Add a 'weights' column to the test parquet."""
    df = cudf.read_parquet(DATA_UNWEIGHTED)
    rng = np.random.default_rng(42)
    df["weights"] = rng.uniform(0.1, 5.0, size=len(df)).astype("float32")
    df.to_parquet(DATA_WEIGHTED, index=False)
    print(f"Created weighted data: {DATA_WEIGHTED} ({len(df)} rows)")
    return DATA_WEIGHTED


def test_weighted_single_gpu():
    """Weighted random walks on a single GPU."""
    path = create_weighted_data()

    from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

    config = RDF2VecConfig(
        walk_strategy="random",
        walk_depth=4,
        walk_number=5,
        walk_weighted=True,
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
    )

    model = GPU_RDF2Vec(config)
    edges = model.load_data(path)
    print(f"load_data OK: {edges.shape}")

    model.fit(edges)
    print("fit OK")

    emb = model.transform()
    print(f"transform OK: {emb.shape}")
    print(f"Cols: {list(emb.columns[:4])} ... {list(emb.columns[-4:])}")


def test_unweighted_still_works():
    """Verify unweighted walks still work after changes."""
    from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

    config = RDF2VecConfig(
        walk_strategy="random",
        walk_depth=4,
        walk_number=5,
        walk_weighted=False,
        embedding_model="skipgram",
        epochs=1,
        vector_size=32,
        window_size=3,
        min_count=1,
        negative_samples=5,
        learning_rate=0.001,
        random_state=42,
        tune_batch_size=False,
        batch_size=512,
    )

    model = GPU_RDF2Vec(config)
    edges = model.load_data(DATA_UNWEIGHTED)
    model.fit(edges)
    emb = model.transform()
    print(f"Unweighted still works: {emb.shape}")


if __name__ == "__main__":
    print("=== Test 1: Weighted random walks (single-GPU) ===")
    test_weighted_single_gpu()
    print("\n=== Test 2: Unweighted still works ===")
    test_unweighted_still_works()
    print("\n=== ALL WEIGHTED WALK TESTS PASSED ===")
