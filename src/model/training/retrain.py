"""Retraining utilities with canary deployment and rollback.

This module offers a high level :func:`retrain_and_deploy` helper that wraps
model training, canary deployment, and automatic rollback.  It relies on
callables passed in by the caller so it can be integrated with different
training frameworks.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def retrain_and_deploy(
    train_fn: Callable[[], Path],
    evaluate_fn: Callable[[Path], float],
    *,
    prod_model_path: str,
    canary_dir: str,
    metric_threshold: float = 0.8,
) -> Path:
    """Retrain the model and roll out via a canary deployment.

    Args:
        train_fn: Callable that trains a model and returns the path to the
            saved artifact.
        evaluate_fn: Callable that evaluates a model and returns a scalar
            performance metric where higher is better.
        prod_model_path: Destination path for the production model.
        canary_dir: Directory where the canary model will be staged.
        metric_threshold: Minimum acceptable evaluation metric for promotion.

    Returns:
        Path to the model that ended up serving production traffic.  If the
        canary evaluation fails, the previous production model path is
        returned after rollback.
    """

    prod_path = Path(prod_model_path)
    canary_dir_path = Path(canary_dir)
    canary_dir_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting retraining...")
    new_model = train_fn()
    LOGGER.info("Training complete: %s", new_model)

    canary_path = canary_dir_path / new_model.name
    shutil.copy2(new_model, canary_path)
    LOGGER.info("Canary model staged at %s", canary_path)

    score = evaluate_fn(canary_path)
    LOGGER.info("Canary evaluation score: %.4f", score)

    backup_path = None
    if prod_path.exists():
        backup_path = prod_path.with_suffix(prod_path.suffix + ".bak")
        shutil.copy2(prod_path, backup_path)
        LOGGER.info("Backed up existing production model to %s", backup_path)

    if score >= metric_threshold:
        shutil.copy2(canary_path, prod_path)
        LOGGER.info("Promoted canary to production: %s", prod_path)
        if backup_path and backup_path.exists():
            backup_path.unlink()
        return prod_path

    LOGGER.warning(
        "Canary score %.4f below threshold %.4f; rolling back", score, metric_threshold
    )
    if backup_path and backup_path.exists():
        shutil.copy2(backup_path, prod_path)
        backup_path.unlink()
        LOGGER.info("Restored previous production model from backup")
    return prod_path


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description="Retrain model with canary deployment")
    parser.add_argument("train", help="Dotted path to training function returning model Path")
    parser.add_argument("evaluate", help="Dotted path to evaluation function")
    parser.add_argument("prod_model_path", help="Path to production model file")
    parser.add_argument("canary_dir", help="Directory for canary deployment")
    parser.add_argument("--threshold", type=float, default=0.8, help="Metric threshold for promotion")
    args = parser.parse_args()

    def _load_callable(dotted: str) -> Callable:
        module, name = dotted.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)

    retrain_and_deploy(
        _load_callable(args.train),
        _load_callable(args.evaluate),
        prod_model_path=args.prod_model_path,
        canary_dir=args.canary_dir,
        metric_threshold=args.threshold,
    )
