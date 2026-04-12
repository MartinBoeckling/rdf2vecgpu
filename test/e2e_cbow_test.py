"""E2E test: CBOW embedding model on a small graph."""
import sys

from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

DATA = "data/generated_graphs/barabasi_graph_100.parquet"

config = RDF2VecConfig(
    walk_strategy="random",
    walk_depth=4,
    walk_number=10,
    embedding_model="cbow",
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

with GPU_RDF2Vec(config) as model:
    edges = model.load_data(DATA)
    print(f"load_data OK: {edges.shape}")

    model.fit(edges)
    print("fit OK")

    emb = model.transform()
    print(f"transform OK: {emb.shape}")
    print(f"Cols: {list(emb.columns[:4])} ... {list(emb.columns[-4:])}")

print("\n=== CBOW E2E PASSED ===")
