import pytest


@pytest.mark.gpu
def test_token_counts_from_cudf_walks_skips_sentinels():
    import cudf
    from rdf2vecgpu.gpu_rdf2vec import _token_counts_from_cudf_walks

    df = cudf.DataFrame({
        "vertices":   [[0, 1, 2, -1], [1, 1, -1, -1]],
        "predicates": [[3, 3, -1], [3, -1, -1]],
    })
    counts = _token_counts_from_cudf_walks(df, vocab_size=4)
    # token 0 →1, token 1 →3 (two in vertices + one), token 2 →1, token 3 →3; -1 skipped.
    assert counts.tolist() == [1, 3, 1, 3]
    assert counts.dtype.name == "int64"
