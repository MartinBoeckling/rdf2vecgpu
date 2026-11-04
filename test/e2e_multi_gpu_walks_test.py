"""E2E test: chunked multi-GPU walk writing (`MultiGPUWalkCorpus.write_walk_sequences`) and the
`walk_generation_sequences(out_dir=...)` path that uses it, with Dask LocalCUDACluster (2+ GPUs).

    python test/e2e_multi_gpu_walks_test.py
"""
import glob
import os
import tempfile


def _check_walks(walks_dir, edge_pd, starts, walk_number, walk_depth):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{walks_dir}/*.parquet"))
    tbl = pq.ParquetDataset(files).read()
    for col in ("vertices", "predicates"):
        t = tbl.schema.field(col).type
        assert pa.types.is_list(t) and t.value_type == pa.int64(), tbl.schema
    V = np.asarray(pc.list_flatten(tbl["vertices"])).reshape(tbl.num_rows, walk_depth + 1)
    P = np.asarray(pc.list_flatten(tbl["predicates"])).reshape(tbl.num_rows, walk_depth)
    first, counts = np.unique(V[:, 0], return_counts=True)
    assert set(first.tolist()) == set(starts) and (counts == walk_number).all(), "walks per start vertex"
    legit = set(zip(edge_pd.subject, edge_pd.predicate, edge_pd.object))
    m = V[:, 1:] >= 0
    steps = set(zip(V[:, :-1][m].tolist(), P[m].tolist(), V[:, 1:][m].tolist()))
    assert steps and steps <= legit, f"{len(steps - legit)} walk steps are not input triples"
    assert (P[~m] == 0).all(), "predicate padding after a walk's end"
    return len(files), tbl.num_rows


def main():
    DATA = "data/generated_graphs/barabasi_graph_1000.parquet"

    import cudf
    import cugraph
    import cugraph.dask.comms.comms as Comms
    import dask_cudf
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster()
    client = Client(cluster)
    n_workers = len(client.scheduler_info()["workers"])
    assert n_workers >= 2, f"needs 2+ GPUs, got {n_workers}"
    Comms.initialize(p2p=True)

    from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig
    from rdf2vecgpu._compat import apply_patches
    from rdf2vecgpu.corpus.walk_corpus import MultiGPUWalkCorpus, encode_predicates, predicate_code_table

    apply_patches()
    try:
        # 1. corpus level: predicate tokens above 2^24, several chunks and several files per task.
        edges = cudf.read_parquet(DATA)
        codes = {w: i for i, w in enumerate(edges["predicate"].unique().to_pandas())}
        vertices = {w: i for i, w in enumerate(cudf.concat([edges["subject"], edges["object"]]).unique().to_pandas())}
        edge_df = cudf.DataFrame({
            "subject": edges["subject"].map(vertices).astype("int64"),
            "predicate": (edges["predicate"].map(codes) + 300_000_001).astype("int64"),
            "object": edges["object"].map(vertices).astype("int64"),
        })
        tokens = edge_df["predicate"].unique().sort_values().values_host
        edge_df["pcode"] = encode_predicates(edge_df["predicate"], tokens)
        G = cugraph.MultiGraph(directed=True)
        G.from_dask_cudf_edgelist(
            dask_cudf.from_cudf(edge_df, npartitions=n_workers),
            source="subject", destination="object", edge_attr="pcode", renumber=False,
        )
        corpus = MultiGPUWalkCorpus(G, window_size=3, code_to_token=predicate_code_table(tokens))
        starts = sorted(edge_df["subject"].unique().to_pandas().tolist())
        with tempfile.TemporaryDirectory() as tmp:
            n = corpus.write_walk_sequences(
                cudf.Series(starts, dtype="int64"), walk_number=7, walk_depth=4, random_state=None,
                out_dir=f"{tmp}/walks", walks_per_worker_chunk=7 * 100, file_mb=0.004,
            )
            n_files, n_rows = _check_walks(f"{tmp}/walks", edge_df.to_pandas(), starts, 7, 4)
            assert n == n_rows == 7 * len(starts), (n, n_rows)
            print(f"write_walk_sequences OK: {n} walks in {n_files} files")

        # 2. library level: walk_generation_sequences(out_dir=...) writes through the same path.
        config = RDF2VecConfig(walk_strategy="random", walk_depth=4, walk_number=5, min_count=1,
                               random_state=42, multi_gpu=True)
        model = GPU_RDF2Vec(config, client=client)
        edges = model.load_data(DATA)
        with tempfile.TemporaryDirectory() as tmp:
            walks = model.walk_generation_sequences(edges, out_dir=f"{tmp}/walks")
            encoded = edges.compute() if hasattr(edges, "compute") else edges
            starts = sorted(model._walk_source_vertices.to_pandas().tolist())
            n_files, n_rows = _check_walks(walks.path, encoded.to_pandas(), starts, 5, 4)
            assert walks.n_walks == n_rows, (walks.n_walks, n_rows)
            print(f"walk_generation_sequences OK: {n_rows} walks in {n_files} files")

        # 3. gensim backend: fit trains from the GPU-written walks (and shuts the cluster down).
        try:
            import gensim  # noqa: F401
        except ImportError:
            print("gensim not installed: skipping the gensim fit")
        else:
            config = RDF2VecConfig(walk_strategy="random", walk_depth=4, walk_number=5, min_count=1,
                                   random_state=42, multi_gpu=True, backend="gensim", epochs=1,
                                   vector_size=16, window_size=3)
            model = GPU_RDF2Vec(config, client=client)
            model.fit(model.load_data(DATA))
            emb = model.transform()
            assert len(emb) > 0 and sum(c.startswith("embedding_") for c in emb.columns) == 16, emb.columns
            print(f"gensim fit OK: {emb.shape}")

        print("\n=== MULTI-GPU WALKS E2E PASSED ===")
    finally:
        if Comms.is_initialized():
            Comms.destroy()
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
