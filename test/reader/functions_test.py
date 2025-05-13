from loguru import logger
import cudf
import dask.dataframe as dd
import dask_cudf
from dask_cuda import LocalCUDACluster
import torch
from __future__ import annotations

import importlib
import sys
import types
from typing import Any, List

import pytest


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

@pytest.fixture(scope="module")
def mod(monkeypatch):
    _install_stubs(monkeypatch)
    return importlib.import_module("src.reader.kg_reader")

@pytest.fixture(scope="module")
def test_generate_vocab_data_single_gpu(mod):
    edge_df = _make_edge_df()
    vocab = mod._generate_vocab_data(edge_df, multi_gpu=False)
    assert isinstance(vocab, cudf.DataFrame)
    assert list(vocab) == ["A", "B", "likes", "C"]
    assert vocab.shape == (3, 2)