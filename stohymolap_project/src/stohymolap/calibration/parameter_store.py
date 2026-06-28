"""Persistencia de parametros calibrados (best + top-K).

Convierte los resultados de calibracion en CSV reproducibles dentro de la
carpeta del experimento.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .monte_carlo_search import CalibrationResult
from ..utils.logging import get_logger

_log = get_logger("calibration.parameter_store")

_TOP_COLUMNS = [
    "mu", "lambda", "sigma", "alpha", "beta", "c_r", "k_b", "S0_b",
    "nse", "J", "KGE", "PBIAS", "rank", "selected",
]


def save_best_parameters(result: CalibrationResult, path: str | Path) -> None:
    """Guarda la mejor combinacion de parametros como CSV de una fila."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.best_params]).to_csv(path, index=False, encoding="utf-8")
    _log.info("Mejores parametros guardados en %s", path)


def save_top_k(result: CalibrationResult, path: str | Path) -> None:
    """Guarda la tabla top-K de parametros."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.top_k_table, columns=_TOP_COLUMNS).to_csv(
        path, index=False, encoding="utf-8"
    )
    _log.info("Top-K parametros guardados en %s", path)


def load_best_parameters(path: str | Path) -> Dict[str, float]:
    """Carga la mejor combinacion de parametros desde CSV."""
    df = pd.read_csv(path)
    return df.iloc[0].to_dict()
