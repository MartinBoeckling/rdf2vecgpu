"""E2E test: walks through `load_data` on a generated KG whose vocabulary
exceeds 2^24 tokens, so predicate tokens do not survive cuGraph's float32
edge attribute, together with parallel edges, an exact duplicate triple, a
tie between equal-weight parallel edges, a self-loop and dead ends. Weighted
and unweighted, single GPU then multi-GPU (2+ GPUs, Dask LocalCUDACluster):

  * every walk step is an input triple, padding follows a walk's end, and
    walks stop early only at dead ends;
  * walks from the hub follow its edge weights (uniform over its edges when
    unweighted), checked on walk sequences and on skip-gram pairs;
  * BFS walks are root-to-leaf paths in order, one predicate per step, drawn
    uniformly over the hub's parallel edges;
  * single GPU: PyTorch fit + transform; multi-GPU: chunked walk writer and,
    with gensim installed, a weighted gensim fit.

    python test/e2e_walk_boundaries_test.py [--n-entities N]
"""
import argparse
import math
import os
import tempfile

import numpy as np

WALK_DEPTH, WALK_NUMBER = 4, 5
HUB_STARTS = 4_000  # x WALK_NUMBER walks from the hub
# x WALK_NUMBER walks from the self-loop vertex: each first step stays there with p = 1/2, so
# with few walks `check_sequences` would by chance see no self-loop step (1/32 at 5 walks)
LOOP_STARTS = 50
# hub -> hub_dst1: zz_a twice (duplicate triple, weight 1), zz_b and zz_d (tie, weight 3); hub -> hub_dst2: zz_c (4)
HUB_EDGES = [("zz_a", "hub_dst1", 1.0), ("zz_a", "hub_dst1", 1.0), ("zz_b", "hub_dst1", 3.0),
             ("zz_d", "hub_dst1", 3.0), ("zz_c", "hub_dst2", 4.0)]
SPECIAL_EDGES = HUB_EDGES + [("zz_self", "loop", 1.0, "loop"), ("zz_exit", "0", 1.0, "loop"),
                             ("zz_in", "sink", 1.0, "5")]


def expected_hub(weighted):
    w = {}
    for p, _, x in HUB_EDGES:
        w[p] = w.get(p, 0) + (x if weighted else 1)
    return {p: v / sum(w.values()) for p, v in w.items()}


def make_kg(path, n):
    """Entities "0".."n-1" in a ring (predicates zp00..zp63) with extra edges
    from the first n/10, random weights in [0.5, 2), plus SPECIAL_EDGES."""
    import cupy as cp
    import cudf

    rng = cp.random.default_rng(0)
    i = cp.arange(n, dtype=cp.int64)
    j = i[: n // 10]
    src = cp.concatenate([i, j])
    dst = cp.concatenate([(i + 1) % n, (j * 7919 + 13) % n])
    pid = cp.concatenate([i % 64, (j + 17) % 64])
    names = cudf.Series([f"zp{k:02d}" for k in range(64)])
    bulk = cudf.DataFrame({
        "subject": cudf.Series(src).astype(str),
        "predicate": names.take(pid).reset_index(drop=True),
        "object": cudf.Series(dst).astype(str),
        "weights": (0.5 + 1.5 * rng.random(len(src))).astype(cp.float32),
    })
    special = [(e[3] if len(e) == 4 else "hub", e[0], e[1], e[2]) for e in SPECIAL_EDGES]
    special = cudf.DataFrame({
        "subject": [s for s, _, _, _ in special], "predicate": [p for _, p, _, _ in special],
        "object": [o for _, _, o, _ in special],
        "weights": cudf.Series([w for _, _, _, w in special], dtype="float32"),
    })
    cudf.concat([bulk, special], ignore_index=True).to_parquet(path, index=False)


def tokens_of(model, words):
    w2i = model.word2idx
    sub = w2i[w2i["word"].isin(words)].to_pandas()
    found = dict(zip(sub["word"], sub["token"].astype(int)))
    assert set(found) == set(words), set(words) - set(found)
    return found


def local(df):
    return df.compute() if hasattr(df, "compute") else df


def check_predicate_precondition(edges, label):
    p = local(edges["predicate"]).unique().values_host.astype(np.int64)
    lossy = p[p.astype(np.float32).astype(np.int64) != p]
    assert len(lossy), f"{label}: no predicate token above float32 precision; raise --n-entities"
    print(f"{label}: {len(lossy)}/{len(p)} predicate tokens change in a float32 round trip (max {p.max():,})")


def check_sequences(walks, edges, depth, starts_count, t, weighted, label):
    """`walks` holds `vertices`/`predicates` lists; `edges` the encoded KG."""
    import cupy as cp
    import cudf

    walks = local(walks)
    n = len(walks)
    V = cp.asarray(walks["vertices"].list.leaves.values).reshape(n, depth + 1).astype(cp.int64)
    P = cp.asarray(walks["predicates"].list.leaves.values).reshape(n, depth).astype(cp.int64)
    first, counts = cp.unique(V[:, 0], return_counts=True)
    got = dict(zip(first.get().tolist(), counts.get().tolist()))
    assert got == starts_count, f"{label}: walks per start vertex differ"

    m = V[:, 1:] >= 0
    steps = cudf.DataFrame({"subject": V[:, :-1][m], "predicate": P[m], "object": V[:, 1:][m]})
    legit = local(edges[["subject", "predicate", "object"]]).astype("int64").drop_duplicates().assign(ok=1)
    joined = steps.merge(legit, on=["subject", "predicate", "object"], how="left")
    bad = int(joined["ok"].isnull().sum())
    assert bad == 0, f"{label}: {bad}/{len(steps)} walk steps are not input triples"
    assert bool((P[~m] == 0).all()), f"{label}: predicate padding after a walk's end"
    assert bool((V[:, 1:][V[:, :-1] < 0] < 0).all()), f"{label}: a walk resumes after its end"

    # Early stops only at dead ends.
    last = V[cp.arange(n), (V >= 0).sum(1) - 1][(V < 0).any(1)]
    has_out = cp.isin(last, cp.asarray(local(edges["subject"]).unique().values))
    assert not bool(has_out.any()), f"{label}: {int(has_out.sum())} walks stop at a vertex with out-edges"
    assert bool((V[V[:, 0] == t["sink"], 1:] == -1).all()), f"{label}: walks from the sink"
    loop = (V[:, :-1] == t["loop"]) & (V[:, 1:] == t["loop"]) & (P == t["zz_self"])
    assert bool(loop.any()), f"{label}: no self-loop step"

    hub = V[:, 0] == t["hub"]
    check_hub(P[hub, 0].get(), t, weighted, label)
    print(f"{label}: {n:,} walks, {len(steps):,} steps OK")


def check_hub(first_predicates, t, weighted, label):
    expected = {t[p]: q for p, q in expected_hub(weighted).items()}
    n = len(first_predicates)
    got = dict(zip(*np.unique(first_predicates, return_counts=True)))
    assert set(got) == set(expected), f"{label}: hub predicates {got} vs {expected}"
    for tok, q in expected.items():
        f = got[tok] / n
        assert abs(f - q) < 5 * math.sqrt(q * (1 - q) / n), f"{label}: hub predicate {tok} at {f:.4f}, expected {q:.4f}"


def check_hub_pairs(pairs, t, weighted, label):
    """Walks from the hub are [hub, p, hub_dst]: one predicate context per walk."""
    pairs = local(pairs)
    preds = [t[p] for p, _, _ in HUB_EDGES]
    ctx = pairs[(pairs["center"] == t["hub"]) & pairs["context"].isin(preds)]["context"].values_host
    assert len(ctx) == HUB_STARTS * WALK_NUMBER, f"{label}: {len(ctx)} hub pairs"
    check_hub(ctx, t, weighted, label)
    print(f"{label}: hub skip-gram pairs OK")


BFS_HUB_ROOTS = 800


def check_bfs(edges, ring_roots, t, label):
    """BFS walks from ring vertices, the self-loop and the hub (BFS_HUB_ROOTS times)."""
    import cudf
    from rdf2vecgpu.corpus.walk_corpus import _bfs_walk_steps

    roots = ring_roots + [t["loop"]] + [t["hub"]] * BFS_HUB_ROOTS
    steps = _bfs_walk_steps(local(edges), cudf.Series(roots, dtype="int32"), WALK_DEPTH, random_state=0)
    legit = local(edges[["subject", "predicate", "object"]]).astype("int64").drop_duplicates().assign(ok=1)
    joined = steps.astype("int64").rename(columns={"src": "subject", "dst": "object"}).merge(
        legit, on=["subject", "predicate", "object"], how="left")
    assert int(joined["ok"].isnull().sum()) == 0, f"{label}: BFS steps that are not input triples"
    s = steps.sort_values(["walk_id", "step"]).to_pandas()
    g = s.groupby("walk_id")
    assert (g["step"].count() == g["step"].max() + 1).all() and not s.duplicated(["walk_id", "step"]).any(), \
        f"{label}: BFS walks need one row per step"
    nxt = s.groupby("walk_id")["src"].shift(-1)
    assert (s["dst"][nxt.notna()] == nxt[nxt.notna()]).all(), f"{label}: BFS steps do not chain"
    first = s[s["step"] == 0]
    assert set(first["src"]) <= set(roots) and (g["step"].max() < WALK_DEPTH).all(), f"{label}: BFS roots/depth"
    hub = first[first["src"] == t["hub"]]
    assert len(hub) == 2 * BFS_HUB_ROOTS, f"{label}: {len(hub)} BFS walks from the hub"
    to_dst1 = hub[hub["dst"] == t["hub_dst1"]]["predicate"].to_numpy()
    # hub -> hub_dst1 carries zz_a twice, zz_b and zz_d: uniform over the four edges
    expected = {t["zz_a"]: 0.5, t["zz_b"]: 0.25, t["zz_d"]: 0.25}
    got = dict(zip(*np.unique(to_dst1, return_counts=True)))
    assert set(got) == set(expected), f"{label}: BFS hub predicates {got}"
    for tok, q in expected.items():
        f = got[tok] / len(to_dst1)
        assert abs(f - q) < 5 * math.sqrt(q * (1 - q) / len(to_dst1)), f"{label}: BFS hub predicate {tok} at {f:.3f}"
    print(f"{label}: {s['walk_id'].nunique():,} BFS walks OK")


def config(weighted, **kw):
    from rdf2vecgpu import RDF2VecConfig

    return RDF2VecConfig(
        walk_strategy="random", walk_depth=WALK_DEPTH, walk_number=WALK_NUMBER, walk_weighted=weighted,
        min_count=1, random_state=42, epochs=1, vector_size=16, window_size=2, negative_samples=2,
        learning_rate=0.001, tune_batch_size=False, batch_size=4096, **kw,
    )


def starts_for(model, n):
    """A sample of ring entities, the hub HUB_STARTS times, the self-loop LOOP_STARTS times and the sink."""
    import cudf

    rng = np.random.default_rng(0)
    words = [str(x) for x in rng.choice(n, 20_000, replace=False)] + ["hub", "loop", "sink"]
    t = tokens_of(model, words + ["hub_dst1", "zz_a", "zz_b", "zz_c", "zz_d", "zz_self"])
    starts = [t[w] for w in words[:-3]] + [t["hub"]] * HUB_STARTS + [t["loop"]] * LOOP_STARTS + [t["sink"]]
    counts = {}
    for s in starts:
        counts[s] = counts.get(s, 0) + WALK_NUMBER
    return cudf.Series(starts, dtype="int32"), counts, t


def run_single_gpu(path, n, weighted):
    import cudf
    from rdf2vecgpu import GPU_RDF2Vec

    label = f"single-GPU {'weighted' if weighted else 'uniform'}"
    model = GPU_RDF2Vec(config(weighted))
    edges = model.load_data(path)
    check_predicate_precondition(edges, label)
    starts, counts, t = starts_for(model, n)
    check_sequences(model.walk_generation_sequences(edges, walk_vertices=starts), edges, WALK_DEPTH, counts, t,
                    weighted, f"{label} sequences")
    hub = cudf.Series([t["hub"]] * HUB_STARTS, dtype="int32")
    check_hub_pairs(model.walk_generation(edges, walk_vertices=hub), t, weighted, f"{label} pairs")
    check_bfs(edges, starts.values_host[:20].tolist(), t, label)
    model.fit(edges, walk_vertices=starts)
    emb = model.transform()
    rows = emb[emb["word"].isin(["zz_a", "zz_b", "zz_c", "zz_d"])]
    assert len(rows) == 4 and sum(c.startswith("embedding_") for c in emb.columns) == 16
    print(f"{label}: fit + transform OK {emb.shape}")


def run_multi_gpu(path, n, weighted, client, fit_gensim):
    from rdf2vecgpu import GPU_RDF2Vec

    label = f"multi-GPU {'weighted' if weighted else 'uniform'}"
    model = GPU_RDF2Vec(config(weighted, multi_gpu=True), client=client)
    edges = model.load_data(path)
    check_predicate_precondition(edges, label)
    starts, counts, t = starts_for(model, n)
    check_sequences(model.walk_generation_sequences(edges, walk_vertices=starts), edges, WALK_DEPTH, counts, t,
                    weighted, f"{label} sequences")
    import cudf

    with tempfile.TemporaryDirectory() as tmp:
        written = model.walk_generation_sequences(edges, walk_vertices=starts, out_dir=f"{tmp}/walks")
        walks = cudf.read_parquet(written.path)
        assert written.n_walks == len(walks)
        check_sequences(walks, edges, WALK_DEPTH, counts, t, weighted, f"{label} chunked writer")
    hub = cudf.Series([t["hub"]] * HUB_STARTS, dtype="int32")
    check_hub_pairs(model.walk_generation(edges, walk_vertices=hub), t, weighted, f"{label} pairs")
    from rdf2vecgpu.corpus.walk_corpus import MultiGPUWalkCorpus

    bfs_pairs = MultiGPUWalkCorpus(model.knowledge_graph, window_size=2).bfs_walk(
        edges, cudf.Series(starts.values_host[:20].tolist() + [t["hub"]], dtype="int32"), WALK_DEPTH,
        0, "skipgram", 1)
    assert len(bfs_pairs) > 0
    print(f"{label}: BFS skip-gram pairs {len(bfs_pairs):,}")
    if fit_gensim:
        model = GPU_RDF2Vec(config(weighted, multi_gpu=True, backend="gensim"), client=client)
        edges = model.load_data(path)
        starts, _, _ = starts_for(model, n)
        model.fit(edges, walk_vertices=starts)  # shuts the cluster down
        emb = model.transform()
        assert emb["word"].isin(["zz_a", "zz_c"]).sum() == 2
        print(f"{label}: gensim fit + transform OK {emb.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-entities", type=int, default=25_000_000)
    args = ap.parse_args()
    import torch  # noqa: F401  (loads its CUDA libraries, incl. NVRTC, before CuPy compiles kernels)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kg.parquet")
        make_kg(path, args.n_entities)
        print(f"generated KG with {args.n_entities:,} ring entities")

        for weighted in (False, True):
            run_single_gpu(path, args.n_entities, weighted)

        import cugraph.dask.comms.comms as Comms
        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster
        from rdf2vecgpu._compat import apply_patches

        try:
            import gensim  # noqa: F401
            has_gensim = True
        except ImportError:
            has_gensim = False
            print("gensim not installed: skipping the gensim fit")
        cluster = LocalCUDACluster()
        client = Client(cluster)
        assert len(client.scheduler_info()["workers"]) >= 2, "needs 2+ GPUs"
        Comms.initialize(p2p=True)
        apply_patches()
        try:
            for weighted in (False, True):
                run_multi_gpu(path, args.n_entities, weighted, client, fit_gensim=weighted and has_gensim)
        finally:
            if Comms.is_initialized():
                Comms.destroy()
            client.close()
            cluster.close()
    print("\n=== WALK BOUNDARIES E2E PASSED ===")


if __name__ == "__main__":
    main()
