"""Utilities for deterministic experiment setup."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:  # torch is optional for documentation building environments
    import torch
except Exception:  # pragma: no cover - torch may be unavailable for docs builds
    torch = None  # type: ignore


def seed_everything(seed: int, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (if available).

    Args:
        seed: Seed value to apply across libraries.
        deterministic_cudnn: If ``True`` and PyTorch is installed, enable
            deterministic cuDNN operations which may have a performance impact.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_env_seed(default: int = 0) -> int:
    """Read a seed from the ``GLOBAL_SEED`` environment variable."""

    value = os.environ.get("GLOBAL_SEED")
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError("GLOBAL_SEED must be an integer") from exc


__all__ = ["seed_everything", "get_env_seed"]
