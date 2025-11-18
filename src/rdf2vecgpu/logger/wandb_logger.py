from .base import BaseTracker


class WandbTracker(BaseTracker):
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        import wandb

        self.wandb = wandb
        self.project = project
        self.run = None
        self.entity = entity
        self.kw = dict(
            project=project, entity=entity, name=run_name, config=config or {}
        )

    def enabled(self) -> bool:
        return True

    def start_pipeline(self, run_name, tags=None, resume_run_id=None):
        self.kw["name"] = self.kw.get("name") or run_name
        self.run = self.wandb.init(**self.kw)
        if tags:
            self.wandb.run.tags = list(
                {*(self.wandb.run.tags or []), *list(tags.values())}
            )
        return self

    def stage(self, name):
        self.wandb.run.log({"stage/start": name})
        try:
            yield self
        finally:
            self.wandb.run.log({"stage/end": name})

    def log_params(self, params):
        self.wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics, step=None):
        self.wandb.log(metrics if step is None else {**metrics, "step": step})

    def set_tags(self, tags):  # store as config + tags
        self.wandb.config.update(
            {"tags": {**self.wandb.config.get("tags", {}), **(tags or {})}},
            allow_val_change=True,
        )

    def log_artifact(self, path, artifact_path=None):
        at = self.wandb.Artifact(artifact_path or "artifacts", type="artifact")
        at.add_file(path)
        self.wandb.run.log_artifact(at)

    def log_dict(self, d, artifact_file, artifact_path=None):
        import json, tempfile, os

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, artifact_file)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)
            self.log_artifact(p, artifact_path or "data")

    def log_text(self, text, artifact_file, artifact_path=None):
        import tempfile, os

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, artifact_file)
            open(p, "w", encoding="utf-8").write(text)
            self.log_artifact(p, artifact_path or "notes")

    def close(self):
        try:
            self.wandb.finish()
        except Exception:
            pass
