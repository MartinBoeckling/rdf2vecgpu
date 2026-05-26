"""Conftest shared by tests under test/embedders/.

Local-env shim for torchvision/torchaudio: the rdf2vecgpu env has a torch
2.12 install (pulled in by the torchrec extras) but torchvision 0.22 /
torchaudio 2.7 wheels built against torch 2.7. torchmetrics — a transitive
dep of `lightning` — probes torchvision at import time and trips a
circular-import error from that torch-version mismatch.

The project itself does not use torchvision or torchaudio. This shim
pre-registers benign module stubs so the probe succeeds and importing
gpu_rdf2vec (which does `import lightning as L` at module level) works
in the current env. It is a no-op when the env is clean.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types

for _name in ("torchvision", "torchaudio"):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__version__ = "0.0.0"
        _m.__file__ = f"<stub-{_name}>"
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules[_name] = _m
