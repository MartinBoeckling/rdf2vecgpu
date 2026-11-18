"""Tests for the random-walk corpus path.

Covers the correctness fixes made when we stopped re-deriving predicates via
a (src, dst) merge and started consuming cuGraph's `edge_attrs` return value
directly. The merge-based recovery had two silent bugs on Wikidata-scale
graphs:
  1. On `cugraph.MultiGraph` with parallel edges, the (src, dst) merge is
     non-unique on the right side — cuDF picks an arbitrary matching
     predicate per row, so walks could end up with predicates the walker
     never traversed.
  2. The multi-GPU path's `_build_walk_df` did `shift(-1) + iloc[:-1]` per
     dask partition, which dropped the last row of every partition — a
     walk-boundary tail vertex when partition cuts crossed a walk.
  3. As a corollary of the multi-GPU partition-local `cp.arange(len(df))`
     walk-id assignment, walk_ids collided across partitions, poisoning the
     downstream skip-gram (walk_id, pos) merge.

The new `_build_walk_steps` reshape helper sidesteps all three; these tests
exercise its mechanics and the calling-side smoke against a single-GPU
`MultiGraph`.
"""
from __future__ import annotations

import cupy as cp
import cudf
import pytest

from src.rdf2vecgpu.corpus.walk_corpus import (
    SingleGPUWalkCorpus,
    _build_walk_steps,
    _walks_to_lists,
    encode_predicates,
    predicate_code_table,
)


@pytest.mark.gpu
@pytest.mark.parametrize("vdtype,pdtype", [("int32", "float32"), ("int64", "int64")])
def test_walks_to_lists_matches_tolist_reference(vdtype, pdtype):
    """The offsets-built list columns equal the old `cudf.Series(m.tolist())`
    output, values and dtypes, including cuGraph's -1 / 0 padding."""
    rng = cp.random.default_rng(0)
    v = rng.integers(-1, 1000, size=(257, 5)).astype(vdtype)
    p = rng.integers(0, 50, size=(257, 4)).astype(pdtype)
    out = _walks_to_lists(cudf.Series(v.ravel()), cudf.Series(p.ravel()), walk_length=4)
    assert out["vertices"].to_arrow().equals(cudf.Series(v.tolist()).to_arrow())
    assert out["predicates"].to_arrow().equals(cudf.Series(p.tolist()).to_arrow())


@pytest.mark.gpu
def test_build_walk_steps_basic_layout():
    """Two walks of length 3 → 6 step rows; src, predicate, dst align."""
    # walk 0: vertices [10, 11, 12, 13], predicates [100, 101, 102]
    # walk 1: vertices [20, 21, 22, 23], predicates [200, 201, 202]
    vertices = cudf.Series(
        cp.array([10, 11, 12, 13, 20, 21, 22, 23], dtype="int32")
    )
    edge_attrs = cudf.Series(
        cp.array([100, 101, 102, 200, 201, 202], dtype="int32")
    )
    out = _build_walk_steps(vertices, edge_attrs, walk_length=3)
    assert list(out.columns) == ["src", "predicate", "dst", "walk_id", "step"]
    assert len(out) == 6
    out_sorted = out.sort_values(["walk_id", "step"]).reset_index(drop=True)
    # Walk 0
    assert out_sorted["src"].iloc[0] == 10 and out_sorted["dst"].iloc[0] == 11
    assert out_sorted["predicate"].iloc[0] == 100
    assert out_sorted["src"].iloc[2] == 12 and out_sorted["dst"].iloc[2] == 13
    assert out_sorted["predicate"].iloc[2] == 102
    # Walk 1
    assert out_sorted["src"].iloc[3] == 20 and out_sorted["dst"].iloc[3] == 21
    assert out_sorted["predicate"].iloc[3] == 200
    assert out_sorted["walk_id"].to_pandas().tolist() == [0, 0, 0, 1, 1, 1]
    assert out_sorted["step"].to_pandas().tolist() == [0, 1, 2, 0, 1, 2]


@pytest.mark.gpu
def test_build_walk_steps_offset_yields_global_walk_ids():
    """`walk_id_offset` shifts the per-partition walk_ids globally.

    This is the exact mechanism the multi-GPU path relies on to avoid the
    partition-local walk_id collision that poisoned the previous code.
    """
    vertices = cudf.Series(cp.array([10, 11, 20, 21], dtype="int32"))
    edge_attrs = cudf.Series(cp.array([100, 200], dtype="int32"))
    out = _build_walk_steps(vertices, edge_attrs, walk_length=1, walk_id_offset=42)
    assert sorted(out["walk_id"].to_pandas().tolist()) == [42, 43]


@pytest.mark.gpu
def test_build_walk_steps_drops_padded_steps():
    """cuGraph's -1 sentinel marks early-terminating walks; those rows drop."""
    # walk 0: full length 3
    # walk 1: length 1 then padded with -1
    vertices = cudf.Series(
        cp.array([10, 11, 12, 13, 20, 21, -1, -1], dtype="int32")
    )
    edge_attrs = cudf.Series(
        cp.array([100, 101, 102, 200, -1, -1], dtype="int32")
    )
    out = _build_walk_steps(vertices, edge_attrs, walk_length=3)
    # 3 rows from walk 0 + 1 valid row from walk 1 = 4
    assert len(out) == 4
    assert (out["src"] != -1).all()
    assert (out["dst"] != -1).all()
    assert (out["predicate"] != -1).all()


@pytest.mark.gpu
def test_build_walk_steps_rejects_misaligned_input():
    """Stride-misaligned input fails loudly rather than silently truncating."""
    bad_vertices = cudf.Series(cp.array([1, 2, 3, 4, 5], dtype="int32"))  # not %4
    edge_attrs = cudf.Series(cp.array([10, 11, 12], dtype="int32"))
    with pytest.raises(ValueError, match="not a multiple of walk stride"):
        _build_walk_steps(bad_vertices, edge_attrs, walk_length=3)


@pytest.mark.gpu
def test_random_walk_predicate_matches_traversed_edge():
    """Walks over a `MultiGraph` with parallel edges emit predicates that
    actually exist between the (src, dst) pair the walker stepped on.

    Pre-fix, the merge-based predicate recovery would pick *any* of the
    parallel predicates per (src, dst) lookup; post-fix, the predicate is
    exactly the one cuGraph chose. The weaker invariant "predicate ∈
    edges_between(src, dst)" still holds in both — but if a future
    regression swaps in a different attribute (e.g. edge weight) the test
    will fail because the value won't be in the legitimate predicate set.
    """
    from cugraph import MultiGraph, uniform_random_walks

    # Parallel edges: (0,1) has predicates 100 and 101.
    edge_df = cudf.DataFrame(
        {
            "subject": cudf.Series([0, 0, 1, 2], dtype="int32"),
            "predicate": cudf.Series([100, 101, 200, 300], dtype="int32"),
            "object": cudf.Series([1, 1, 2, 0], dtype="int32"),
        }
    )
    graph = MultiGraph(directed=True)
    graph.from_cudf_edgelist(
        edge_df,
        source="subject",
        destination="object",
        edge_attr="predicate",
        renumber=False,
    )

    start = cudf.Series([0, 0, 0, 0], dtype="int32")
    vp, ea, max_len = uniform_random_walks(
        graph, start_vertices=start, max_depth=2, random_state=0
    )
    steps = _build_walk_steps(vp, ea, walk_length=int(max_len))

    # Legitimate (src, dst) → {predicates} map from the edge list.
    edge_pd = edge_df.to_pandas()
    legit: dict[tuple[int, int], set[int]] = {}
    for _, row in edge_pd.iterrows():
        legit.setdefault((int(row["subject"]), int(row["object"])), set()).add(
            int(row["predicate"])
        )

    steps_pd = steps.to_pandas()
    for _, row in steps_pd.iterrows():
        key = (int(row["src"]), int(row["dst"]))
        assert key in legit, f"walk traversed edge {key} that does not exist"
        assert int(row["predicate"]) in legit[key], (
            f"walk emitted predicate {int(row['predicate'])} for edge {key}; "
            f"only {legit[key]} are legitimate predicates between them"
        )

    # Smoke the public single-GPU corpus API end-to-end.
    corpus = SingleGPUWalkCorpus(graph, window_size=2, walk_weighted=False)
    pairs = corpus.random_walk(
        edge_df=edge_df,
        walk_vertices=start,
        walk_depth=2,
        random_state=0,
        word2vec_model="skipgram",
        min_count=1,
    )
    assert isinstance(pairs, cudf.DataFrame)
    assert {"center", "context"}.issubset(set(pairs.columns))


@pytest.mark.gpu
def test_walks_decode_predicate_tokens_above_float32_range():
    """Predicate tokens above 2^24 are rounded in cuGraph's float32 edge
    attribute; on `encode_predicates` codes the walks decode them exactly."""
    import numpy as np
    from cugraph import MultiGraph

    big = [16_777_217, 300_000_001, 392_999_999]  # none survive a float32 round trip
    assert all(int(np.float32(t)) != t for t in big)
    edge_df = cudf.DataFrame(
        {
            "subject": cudf.Series([0, 0, 1, 2], dtype="int32"),
            "predicate": cudf.Series([big[0], big[1], big[2], 7], dtype="int64"),
            "object": cudf.Series([1, 1, 2, 0], dtype="int32"),
        }
    )
    tokens = edge_df["predicate"].unique().sort_values().values_host
    edge_df["pcode"] = encode_predicates(edge_df["predicate"], tokens)
    graph = MultiGraph(directed=True)
    graph.from_cudf_edgelist(edge_df, source="subject", destination="object", edge_attr="pcode", renumber=False)
    corpus = SingleGPUWalkCorpus(graph, window_size=2, code_to_token=predicate_code_table(tokens))

    walks = corpus.random_walk_sequences(cudf.Series([0, 1, 2] * 20, dtype="int32"), walk_depth=3, random_state=0)
    assert str(walks["predicates"].dtype) == "list"
    legit = {(int(r.subject), int(r.object), int(r.predicate)) for r in edge_df.to_pandas().itertuples()}
    V = walks["vertices"].list.leaves.values_host.reshape(len(walks), 4)
    P = walks["predicates"].list.leaves.values_host.reshape(len(walks), 3)
    steps = {(int(v), int(o), int(p)) for v, o, p in zip(V[:, :-1].ravel(), V[:, 1:].ravel(), P.ravel()) if o >= 0}
    assert steps and steps <= legit


def _coded_graph():
    """A small MultiGraph on `encode_predicates` codes, with parallel edges and a
    dead end (vertex 3), plus the set of its (subject, object, predicate) edges."""
    from cugraph import MultiGraph

    edge_df = cudf.DataFrame(
        {
            "subject": cudf.Series([0, 0, 1, 2, 2], dtype="int32"),
            "predicate": cudf.Series([40, 41, 42, 43, 44], dtype="int64"),
            "object": cudf.Series([1, 1, 2, 0, 3], dtype="int32"),
        }
    )
    tokens = edge_df["predicate"].unique().sort_values().values_host
    edge_df["pcode"] = encode_predicates(edge_df["predicate"], tokens)
    graph = MultiGraph(directed=True)
    graph.from_cudf_edgelist(edge_df, source="subject", destination="object", edge_attr="pcode", renumber=False)
    legit = {(int(r.subject), int(r.object), int(r.predicate)) for r in edge_df.to_pandas().itertuples()}
    return SingleGPUWalkCorpus(graph, window_size=2, code_to_token=predicate_code_table(tokens)), legit


def _check_walks(V, P, starts, walk_number, legit):
    """Every start vertex begins `walk_number` walks, every step is an edge of
    the graph with its predicate, and ended walks are padded with -1."""
    import numpy as np

    assert sorted(V[:, 0].tolist()) == sorted(np.repeat(starts, walk_number).tolist())
    steps = [(int(v), int(o), int(p)) for v, o, p in zip(V[:, :-1].ravel(), V[:, 1:].ravel(), P.ravel()) if o >= 0]
    assert steps and set(steps) <= legit
    ended = V[:, 1:] < 0
    assert (V[:, 1:][np.cumsum(ended, axis=1) > 0] == -1).all()  # once ended, stays ended


@pytest.mark.gpu
def test_decode_predicates_returns_int32_tokens():
    """Predicate tokens are decoded straight to int32 (vocabulary tokens fit:
    cuDF caps the vocabulary at 2^31 rows), without int64 copies of the walks."""
    from src.rdf2vecgpu.corpus.walk_corpus import _decode_predicates

    table = predicate_code_table([5, 9, 300_000_001])
    out = _decode_predicates(cp.asarray([1, 0, 3, 2], dtype=cp.float32), table)
    assert out.dtype == cp.int32
    assert out.tolist() == [5, 0, 300_000_001, 9]
    with pytest.raises(ValueError):
        _decode_predicates(cp.asarray([1.5], dtype=cp.float32), table)
    with pytest.raises(ValueError):
        _decode_predicates(cp.asarray([4], dtype=cp.float32), table)


@pytest.mark.gpu
def test_single_gpu_walks_come_in_chunks_of_int32_matrices():
    """`walk_chunks` walks `walks_per_chunk` walks (whole start vertices) per
    cuGraph call and yields int32 (vertices, predicates) matrices."""
    corpus, legit = _coded_graph()
    starts = cudf.Series([0, 1, 2], dtype="int32")

    chunks = list(corpus.walk_chunks(starts, walk_number=20, walk_depth=3, random_state=0, walks_per_chunk=45))

    assert [len(v) for v, _ in chunks] == [40, 20]  # 2 start vertices x 20 walks, then 1
    for v, p in chunks:
        assert v.dtype == cp.int32 and p.dtype == cp.int32
        assert v.shape[1] == 4 and p.shape[1] == 3
    V = cp.concatenate([v for v, _ in chunks]).get()
    P = cp.concatenate([p for _, p in chunks]).get()
    _check_walks(V, P, [0, 1, 2], 20, legit)


@pytest.mark.gpu
def test_single_gpu_walk_matrices_hold_every_chunk():
    import torch

    corpus, legit = _coded_graph()
    vertices, predicates = corpus.walk_matrices(
        cudf.Series([0, 1, 2], dtype="int32"), walk_number=20, walk_depth=3, random_state=0, walks_per_chunk=20
    )
    assert vertices.is_cuda and vertices.dtype == torch.int32 and predicates.dtype == torch.int32
    assert vertices.shape == (60, 4) and predicates.shape == (60, 3)
    _check_walks(vertices.cpu().numpy(), predicates.cpu().numpy(), [0, 1, 2], 20, legit)


@pytest.mark.gpu
def test_single_gpu_write_walk_sequences_writes_each_chunk(tmp_path):
    """Walks written chunk by chunk to parquet, in the schema of the multi-GPU
    writer, so the corpus never has to fit on the GPU at once."""
    import numpy as np
    import pyarrow.parquet as pq

    corpus, legit = _coded_graph()
    out = tmp_path / "walks"
    n = corpus.write_walk_sequences(
        cudf.Series([0, 1, 2], dtype="int32"), walk_number=20, walk_depth=3, random_state=0,
        out_dir=str(out), walks_per_chunk=20,
    )
    assert n == 60
    files = sorted(out.glob("part.*.parquet"))
    assert len(files) == 3
    table = pq.read_table(out)
    assert table.num_rows == 60
    V = np.array(table.column("vertices").to_pylist())
    P = np.array(table.column("predicates").to_pylist())
    _check_walks(V, P, [0, 1, 2], 20, legit)
    with pytest.raises(ValueError, match="already holds parquet"):
        corpus.write_walk_sequences(cudf.Series([0], dtype="int32"), walk_number=1, walk_depth=3,
                                    random_state=0, out_dir=str(out))
