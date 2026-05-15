"""Unit tests for the walk-coverage rewrite — _walk_source_vertices plumbing
fix that routes around cuGraph .nodes() returning a fraction of the actual
source-vertex set. Spec:
docs/superpowers/specs/2026-05-15-walk-coverage-rewrite-design.md (§1, §2, D2, D5).

The synthetic graph has 6 entity tokens: 4 subjects {0, 1, 2, 3} that each
have an outgoing edge, and 2 object-only leaves {4, 5} that appear only on
the right side of a triple. After construct_graph, _walk_source_vertices
should contain exactly the 4 subjects (D5: leaves stay out of starts).
"""
import pytest
from unittest.mock import MagicMock


def _make_synthetic_kg_data():
    """5-triple toy: subjects {0,1,2,3}, objects {1,2,3,4,5} — leaves {4,5}."""
    import cudf
    return cudf.DataFrame({
        "subject":   [0, 0, 1, 2, 3],
        "predicate": [10, 11, 10, 11, 12],
        "object":    [1, 4, 2, 3, 5],
    })


def _make_bare_instance(multi_gpu: bool = False):
    """GPU_RDF2Vec instance with __init__ bypassed — for unit-test isolation
    from dask client / cuGraph setup. Mirrors the pattern in test_walked_mask.py.
    """
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec
    inst = GPU_RDF2Vec.__new__(GPU_RDF2Vec)
    inst.config = RDF2VecConfig(multi_gpu=multi_gpu)
    inst.tracker = MagicMock()
    inst.tracker.stage = MagicMock(return_value=MagicMock(
        __enter__=lambda s: None, __exit__=lambda *a: None
    ))
    inst.tracker.log_params = MagicMock()
    inst.tracker.log_metrics = MagicMock()
    inst.knowledge_graph = MagicMock()
    inst.knowledge_graph.from_cudf_edgelist = MagicMock()
    inst.knowledge_graph.from_dask_cudf_edgelist = MagicMock()
    inst.knowledge_graph.degree = MagicMock(return_value=MagicMock())
    inst.knowledge_graph.number_of_vertices = MagicMock(return_value=6)
    inst.knowledge_graph.number_of_edges = MagicMock(return_value=5)
    inst._walked_mask = None
    inst._walk_source_vertices = None
    return inst


@pytest.mark.gpu
def test_construct_graph_stashes_walk_source_vertices_single_gpu():
    """Spec §1: single-GPU mode stashes the unique-subject cudf.Series."""
    inst = _make_bare_instance(multi_gpu=False)
    inst.construct_graph(_make_synthetic_kg_data())

    assert inst._walk_source_vertices is not None
    actual = sorted(inst._walk_source_vertices.to_pandas().tolist())
    assert actual == [0, 1, 2, 3], (
        f"expected 4 subjects (object-only leaves excluded per D5); got {actual}"
    )


@pytest.mark.gpu
def test_resolve_walk_vertices_returns_stash_when_arg_is_none():
    """Spec §2: when walk_vertices=None, the helper returns _walk_source_vertices."""
    import cudf
    inst = _make_bare_instance(multi_gpu=False)
    inst._walk_source_vertices = cudf.Series([0, 1, 2, 3], name="subject")

    resolved = inst._resolve_walk_vertices(None)
    assert resolved is inst._walk_source_vertices


@pytest.mark.gpu
def test_resolve_walk_vertices_passes_explicit_arg_through():
    """Spec §2: explicit walk_vertices argument bypasses the stash."""
    import cudf
    inst = _make_bare_instance(multi_gpu=False)
    inst._walk_source_vertices = cudf.Series([0, 1, 2, 3], name="subject")

    explicit = cudf.Series([7, 8], name="subject")
    resolved = inst._resolve_walk_vertices(explicit)
    assert resolved is explicit  # same object, no stash substitution


def test_resolve_walk_vertices_unfitted_raises():
    """Spec §2: un-fitted instance + walk_vertices=None ⇒ RuntimeError."""
    inst = _make_bare_instance(multi_gpu=False)
    # _walk_source_vertices already None from _make_bare_instance

    with pytest.raises(RuntimeError, match=r"call fit\(\) first"):
        inst._resolve_walk_vertices(None)
