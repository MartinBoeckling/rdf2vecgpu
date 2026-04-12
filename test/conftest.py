import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="run gpu tests")


def pytest_runtest_setup(item):
    if "gpu" in item.keywords:
        if not item.config.getoption("--gpu"):
            pytest.skip("need --gpu option to run")
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
