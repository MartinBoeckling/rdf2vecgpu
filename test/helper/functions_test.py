from __future__ import annotations
from loguru import logger
import cudf
import dask.dataframe as dd
import dask_cudf
from dask_cuda import LocalCUDACluster
import torch

import importlib
import sys
import types
from typing import Any, List

import pytest
from src.rdf2vecgpu.helper.functions import (
    _generate_vocab,
    cudf_to_torch_tensor,
    torch_to_cudf,
)


def _install_stubs(monkeypatch):
    try:
        import cudf, dask_cudf, dask_cuda, torch
    except ModuleNotFoundError:
        raise ImportError(
            "Please install the required packages: cudf, dask_cudf, dask_cuda, torch"
        )


def _make_edge_df():
    return cudf.DataFrame(
        {
            "subject": cudf.Series(["A", "B"]),
            "predicate": cudf.Series(["likes", "likes"]),
            "object": cudf.Series(["B", "C"]),
        }
    )


def _make_cudf_df():
    return cudf.DataFrame(
        {"context": cudf.Series([0, 1, 2]), "word": cudf.Series([10, 20, 30])}
    )


def _make_dask_cudf_df():
    return dask_cudf.from_cudf(_make_edge_df(), npartitions=2)


def _make_tensor_data():
    return torch.tensor([[0, 1, 2], [3, 4, 5]]).to(device="cuda")


def test_generate_vocab_single_gpu():
    edge_df = _make_edge_df()
    edge_df, vocab = _generate_vocab(edge_df, multi_gpu=False)
    assert list(vocab.columns) == ["token", "word"]
    assert list(edge_df.columns) == ["subject", "predicate", "object"]
    assert isinstance(vocab, cudf.DataFrame)
    assert vocab.shape == (4, 2)


def test_generate_vocab_multi_gpu():
    edge_df = _make_dask_cudf_df()
    edge_df, vocab = _generate_vocab(edge_df, multi_gpu=True)
    assert isinstance(vocab, dask_cudf.DataFrame)
    assert isinstance(edge_df, dask_cudf.DataFrame)
    assert list(vocab.columns) == ["word", "token"]
    assert list(edge_df.columns) == ["subject", "predicate", "object"]
    assert vocab.compute().shape == (4, 2)


def _make_larger_dask_cudf_df(npartitions: int = 4):
    """A 12-row triple table chosen so the unique vocab is exactly 11 strings.

    Used to exercise the multi-GPU vocab build with multiple hash partitions.
    The vocab is intentionally larger than `npartitions` so the cumsum-offset
    arithmetic actually has multiple non-empty partitions to stitch together.
    """
    return dask_cudf.from_cudf(
        cudf.DataFrame(
            {
                "subject": cudf.Series(
                    ["A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F"]
                ),
                "predicate": cudf.Series(
                    ["p1", "p2", "p3", "p1", "p2", "p3", "p1", "p2", "p3", "p1", "p2", "p3"]
                ),
                "object": cudf.Series(
                    ["X", "Y", "Z", "X", "Y", "Z", "Y", "Z", "X", "Z", "X", "Y"]
                ),
            }
        ),
        npartitions=npartitions,
    )


@pytest.mark.gpu
def test_generate_vocab_multi_gpu_unique_tokens_contiguous_range():
    """Every (word, token) pair is unique and tokens form a contiguous [0, n).

    Catches off-by-one in the per-partition cumsum offsets — if any partition's
    `_assign_ids` started at the wrong offset, tokens would either collide
    (overlap) or skip ids (gap), breaking contiguity.
    """
    edge_df = _make_larger_dask_cudf_df(npartitions=4)
    _, vocab_dd = _generate_vocab(edge_df, multi_gpu=True)
    vocab = vocab_dd.compute()
    # Vocab is the union of {A..F, p1..p3, X..Z} = 12 unique strings.
    expected_words = {"A", "B", "C", "D", "E", "F", "p1", "p2", "p3", "X", "Y", "Z"}
    assert set(vocab["word"].to_pandas()) == expected_words
    # Token ids are unique.
    tokens = sorted(vocab["token"].to_pandas().tolist())
    assert len(tokens) == len(set(tokens)), "duplicate token ids"
    # Token ids form a contiguous [0, n) range.
    assert tokens == list(range(len(tokens)))


@pytest.mark.gpu
def test_generate_vocab_multi_gpu_round_trip_decodes_to_original_strings():
    """edge_df decoded back through word2idx equals the original triple set.

    Catches drift where the broadcast merges or the per-partition reshape
    silently lose rows (e.g. if `drop_duplicates` got re-rolled without
    persist, or a broadcast merge dropped non-matching rows).
    """
    original = _make_larger_dask_cudf_df(npartitions=4).compute().reset_index(drop=True)
    encoded_dd, vocab_dd = _generate_vocab(
        _make_larger_dask_cudf_df(npartitions=4), multi_gpu=True
    )
    encoded = encoded_dd.compute().reset_index(drop=True)
    vocab = vocab_dd.compute()

    # Build a token → word lookup from the materialized vocab.
    token_to_word = dict(
        zip(vocab["token"].to_pandas().tolist(), vocab["word"].to_pandas().tolist())
    )
    decoded_subject = encoded["subject"].to_pandas().map(token_to_word).tolist()
    decoded_predicate = encoded["predicate"].to_pandas().map(token_to_word).tolist()
    decoded_object = encoded["object"].to_pandas().map(token_to_word).tolist()

    decoded_triples = set(zip(decoded_subject, decoded_predicate, decoded_object))
    original_triples = set(
        zip(
            original["subject"].to_pandas().tolist(),
            original["predicate"].to_pandas().tolist(),
            original["object"].to_pandas().tolist(),
        )
    )
    assert decoded_triples == original_triples, (
        f"decoded triples differ from original; "
        f"missing={original_triples - decoded_triples}, "
        f"unexpected={decoded_triples - original_triples}"
    )
    # Row count preserved (no row loss from broadcast merges).
    assert len(encoded) == len(original)


def test_cudf_to_torch_tensor():
    cudf_df = _make_cudf_df()
    tensor = cudf_to_torch_tensor(cudf_df, "word")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3,)
    comparison_tensor = torch.tensor([10, 20, 30]).to(device="cuda")
    assert torch.equal(tensor, comparison_tensor)
    context_tensor = cudf_to_torch_tensor(cudf_df, "context")
    assert isinstance(context_tensor, torch.Tensor)
    assert context_tensor.shape == (3,)
    comparison_tensor = torch.tensor([0, 1, 2]).to(device="cuda")
    assert torch.equal(context_tensor, comparison_tensor)
    unavailable_column = "non_existent_column"
    with pytest.raises(ValueError) as excinfo:
        unavailable_tensor = cudf_to_torch_tensor(cudf_df, unavailable_column)


def test_tensor_to_cudf():
    tensor = _make_tensor_data()
    cudf_df = torch_to_cudf(tensor, multi_gpu=False)
    assert isinstance(cudf_df, cudf.DataFrame)
    assert cudf_df.shape == (2, 3)
    assert list(cudf_df.columns) == [0, 1, 2]
    assert cudf_df.iloc[0, 0] == 0
    assert cudf_df.iloc[1, 2] == 5
    with pytest.raises(NotImplementedError) as excinfo:
        multi_gpu_cudf_df = torch_to_cudf(tensor, multi_gpu=True)
