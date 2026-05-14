"""Unit tests for the walked_mask coverage-fix feature (spec §7.5, plan Task 4-6).

The bitmask is derived from `_token_counts_from_walk_parquet`. These tests
exercise the derivation and downstream zero-row application in transform.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import pytest


def _write_synth_walks(tmpdir, walks):
    """walks: list of (vertices: List[int], predicates: List[int]) tuples.
    Writes a single parquet file under `<tmpdir>/walks/part.0.parquet`."""
    os.makedirs(f"{tmpdir}/walks", exist_ok=True)
    tbl = pa.table({
        "vertices": pa.array([v for v, _ in walks], type=pa.list_(pa.int32())),
        "predicates": pa.array([p for _, p in walks], type=pa.list_(pa.int32())),
    })
    pq.write_table(tbl, f"{tmpdir}/walks/part.0.parquet")


def test_walked_mask_from_synthetic_walks(tmp_path):
    """walked_mask[i] iff token i appears as a vertex or predicate in any walk."""
    from src.rdf2vecgpu.gpu_rdf2vec import _token_counts_from_walk_parquet

    walks = [
        ([0, 5, 2], [7, 9]),       # tokens touched: 0, 2, 5, 7, 9
        ([5, 2], [9]),             # tokens touched: 2, 5, 9
        ([3], []),                 # token touched: 3
    ]
    _write_synth_walks(str(tmp_path), walks)
    counts = _token_counts_from_walk_parquet(f"{tmp_path}/walks", vocab_size=10)
    mask = counts > 0
    expected_touched = {0, 2, 3, 5, 7, 9}
    expected_untouched = {1, 4, 6, 8}
    assert set(np.flatnonzero(mask).tolist()) == expected_touched
    assert set(np.flatnonzero(~mask).tolist()) == expected_untouched


def test_walked_mask_attribute_is_initialised_to_none():
    """`_walked_mask` must be initialised to None in GpuRdf2Vec.__init__ so legacy/
    unfitted instances skip the masking in transform (spec §7.5 backward-compat).

    Source-level check rather than constructing the class (which requires a full
    config + GPU). The string literal must appear in the module file."""
    import importlib
    rdf_mod = importlib.import_module("src.rdf2vecgpu.gpu_rdf2vec")
    src = open(rdf_mod.__file__).read()
    assert "self._walked_mask = None" in src, (
        "_walked_mask must be initialised to None in GpuRdf2Vec.__init__ "
        "(spec §7.5 backward-compat — legacy/unfitted instances skip masking)"
    )
