"""E2E test: prepare a walk corpus once, then train skip-gram on it with DDP
across all visible GPUs via `fit_from_walks` (no Dask client).

Lightning starts each further GPU's process by running this script again; the
corpus already exists by then, so those processes only read it.
Run from the repository root: python test/e2e_fit_from_walks_multi_gpu_test.py
"""
import os
import tempfile

DATA = "data/generated_graphs/barabasi_graph_1000.parquet"
CORPUS = os.environ.setdefault(
    "RDF2VEC_E2E_CORPUS", os.path.join(tempfile.gettempdir(), f"rdf2vec_e2e_corpus_{os.getpid()}")
)


def main():
    import numpy as np
    import torch

    from rdf2vecgpu import GPU_RDF2Vec

    settings = dict(walk_number=10, walk_depth=4, epochs=2, vector_size=32, window_size=3, batch_size=4096)
    walks_path = os.path.join(CORPUS, "walks")
    vocab_path = os.path.join(CORPUS, "word2idx.parquet")
    if not os.path.exists(vocab_path):  # only in the first process
        preparer = GPU_RDF2Vec(**settings)
        preparer.prepare_corpus(preparer.load_data(DATA), CORPUS)

    model = GPU_RDF2Vec(multi_gpu=True, **settings)
    model.fit_from_walks(walks_path, vocab_path)
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    assert torch.cuda.device_count() >= 2, "needs two GPUs"
    emb = model.transform()
    vectors = emb[[f"embedding_{i}" for i in range(32)]].to_pandas().to_numpy()
    assert np.isfinite(vectors).all() and np.abs(vectors).sum() > 0
    print(f"transform OK: {emb.shape}")
    print("\n=== FIT_FROM_WALKS MULTI-GPU E2E PASSED ===")


if __name__ == "__main__":
    main()
