"""Features de incertidumbre derivadas del ensemble Monte Carlo.

Convierte el ``summary`` de un ``EnsembleResult`` en un DataFrame de columnas
listas para alimentar a los modelos hibridos (E4-E7):

    Qmean, q01, q05, q25, q50, q75, q95, q99,
    width_q95_q05, width_q75_q25, std_ensemble,
    Qbase, Qfast  (y skew opcional).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

_log = get_logger("features.uncertainty_features")

_BASE_COLUMNS = [
    "Qmean", "q01", "q05", "q25", "q50", "q75", "q95", "q99",
    "width_q95_q05", "width_q75_q25", "std_ensemble", "Qbase", "Qfast",
]


def ensemble_summary_to_frame(
    summary: Dict[str, np.ndarray],
    n: int,
    include_skew: bool = False,
    ensemble: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Construye un DataFrame de features de incertidumbre desde el summary."""
    data = {}
    for col in _BASE_COLUMNS:
        if col in summary:
            data[col] = np.asarray(summary[col], dtype=float)
        else:
            data[col] = np.full(n, np.nan)

    if include_skew and ensemble is not None:
        data["skew_ensemble"] = _ensemble_skew(ensemble)

    return pd.DataFrame(data)


def _ensemble_skew(ensemble: np.ndarray) -> np.ndarray:
    """Asimetria por paso de tiempo (sin depender de scipy)."""
    ens = np.asarray(ensemble, dtype=float)
    mean = np.nanmean(ens, axis=1, keepdims=True)
    std = np.nanstd(ens, axis=1, keepdims=True)
    std = np.where(std == 0, np.nan, std)
    m3 = np.nanmean(((ens - mean) / std) ** 3, axis=1)
    return m3
