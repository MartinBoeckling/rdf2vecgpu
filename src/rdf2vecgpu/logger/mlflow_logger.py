from .base import BaseTracker
from loguru import logger
from typing import Dict, Any, Optional, Iterable, Union
import os
import tempfile

try:
    import mlflow
except ImportError as e:
    logger.exception("mlflow is not installed. Please install it to use MLflowLogger.")
    raise


class MlflowTracker(BaseTracker):
    def __init__(
        self, experiment: str, tracking_uri: str, registry_uri: Optional[str] = None
    ):
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        mlflow.set_experiment(experiment)

        self._parent_run = None
        self._active_stage_runs = []

    def enabled(self) -> bool:
        return True

    def start_pipeline(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> "MlflowTracker":
        if run_name is not None:
            self._parent_run = mlflow.start_run(run_name)
        else:
            self._parent_run = mlflow.start_run()
        if tags:
            mlflow.set_tags(tags)
        return self

    def stage(self, name: str):
        run = mlflow.start_run(run_name=name, nested=True)
        self._active_stage_runs.append(run)
        try:
            yield self
        finally:
            mlflow.end_run()
            self._active_stage_runs.pop()

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step:
            mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics)

    def set_tags(self, tags: Dict[str, str]):
        mlflow.set_tags(tags)

    def log_artifact(self, path, artifact_path=None):
        mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_figure(self, fig, artifact_file, artifact_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, artifact_file)
            fig.savefig(p)
            self.log_artifact(p, artifact_path=artifact_path)

    def log_model_pytorch(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)

    def close(self):
        while self._active_stage_runs:
            mlflow.end_run()
            self._active_stage_runs.pop()
        if self._parent_run:
            mlflow.end_run()
            self._parent_run = None


def make_tracker(
    experiment: str, tracking_uri: str, registry_uri: Optional[str] = None
) -> Union[MlflowTracker, None]:
    return MlflowTracker(
        experiment=experiment, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
