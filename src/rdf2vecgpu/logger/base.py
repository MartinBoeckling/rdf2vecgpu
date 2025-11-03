from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, Any, Optional, Iterable, Union
import os
import json
import tempfile


class BaseTracker:
    def enabled(self) -> bool:
        return False

    def start_pipeline(
        self, run_name: Optional[str], tags: Optional[Dict[str, str]] = None
    ) -> None:
        return self

    @contextmanager
    def stage(self, name: str):
        yield self

    def log_params(self, params: Dict[str, Any]):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        pass

    def log_figure(self, figure: Any, artifact_path: str):
        pass

    def close(self):
        pass

    def log_model_pytorch(self, model, artifact_path: str):
        pass
