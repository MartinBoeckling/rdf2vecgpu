from .base import BaseTracker, NoOpTracker
from loguru import logger


def build_tracker(spec: str | None, kwargs: dict | None = None) -> BaseTracker:
    """
    spec: "none", "mlflow", "wandb"
    kwargs: {"mlflow": {...}, "wandb": {...}}
    """
    spec = (spec or "none").strip().lower()
    if spec == "none":
        return NoOpTracker()
    elif spec == "mlflow":
        try:
            from .mlflow_logger import MlflowTracker

            mlflow_kwargs = kwargs or {}
            return MLflowTracker(**mlflow_kwargs.get("mlflow", {}))
        except ImportError:
            logger.exception(
                "mlflow is not installed. Please install it to use MlflowTracker."
            )
            raise
    elif spec == "wandb":
        try:
            from .wandb_logger import WandbTracker

            wandb_kwargs = kwargs or {}
            return WandbTracker(**wandb_kwargs.get("wandb", {}))
        except ImportError:
            logger.exception(
                "wandb is not installed. Please install it to use WandbTracker."
            )
            raise
    else:
        logger.exception(f"Unknown tracker specification: {spec}")
        raise ValueError(f"Unknown tracker specification: {spec}")
