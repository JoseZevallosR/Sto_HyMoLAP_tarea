"""Utilidades de reproducibilidad: semillas, entorno y metadatos de ejecución."""
from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np

from .logging import get_logger

_log = get_logger("utils.reproducibility")


def seed_everything(seed: int = 42) -> np.random.Generator:
    """Fija la semilla global y devuelve un ``numpy.random.Generator``."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # pragma: no cover - depende del entorno
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass

    try:  # pragma: no cover - depende del entorno
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    _log.debug("Semilla global fijada en %d", seed)
    return np.random.default_rng(seed)


def child_rng(seed: int, offset: int) -> np.random.Generator:
    """Devuelve un generador derivado e independiente."""
    return np.random.default_rng(seed + offset)


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return out or None
    except Exception:
        return None


def _find_project_root() -> Path:
    """Busca una raíz razonable del proyecto para consultar git."""
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    return here


def package_versions(packages: Optional[list[str]] = None) -> dict[str, str | None]:
    """Devuelve versiones de dependencias relevantes si están instaladas."""
    packages = packages or [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pyyaml",
        "joblib",
        "xgboost",
        "tensorflow",
        "torch",
    ]
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def runtime_metadata() -> dict[str, object]:
    """Metadatos mínimos para reproducibilidad científica.

    Se guarda dentro de ``config_used.yaml`` en cada experimento.
    """
    root = _find_project_root()
    status = _git_value(["status", "--short"], root)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "working_directory": str(Path.cwd().resolve()),
        "git_root": str(root),
        "git_commit": _git_value(["rev-parse", "HEAD"], root),
        "git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "git_is_dirty": bool(status),
        "git_status_short": status or "",
        "package_versions": package_versions(),
    }
