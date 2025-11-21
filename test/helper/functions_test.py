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
    return dask_cudf.from_cudf(_make_cudf_df(), npartitions=2)


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
    assert list(vocab.columns) == ["token", "word"]
    assert list(edge_df.columns) == ["subject", "predicate", "object"]
    assert vocab.compute().shape == (4, 2)


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
