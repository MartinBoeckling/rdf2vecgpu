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
        except ImportError:
            logger.exception(
                "mlflow is not installed. Please install it to use MlflowTracker."
            )
            raise
    elif spec == "wandb":
        try:
            from .wandb_logger import WandbTracker
        except ImportError:
            logger.exception(
                "wandb is not installed. Please install it to use WandbTracker."
            )
            raise
