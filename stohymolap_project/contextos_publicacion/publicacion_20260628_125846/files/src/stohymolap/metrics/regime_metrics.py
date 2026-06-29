"""Metricas por regimen hidrologico (caudales bajos, medios y altos).

Los regimenes se definen por percentiles de Qobs calculados *sobre el periodo
de referencia* (idealmente el de calibracion, para no usar informacion de
validacion en la definicion de umbrales):

    bajos : Qobs <= P25
    medios: P25 < Qobs <= P75
    altos : Qobs > P75

Se calculan NSE, KGE, RMSE, MAE y PBIAS por regimen, y opcionalmente la
cobertura (PICP) por regimen cuando hay intervalos.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .deterministic import nse, kge, rmse, mae, pbias
from .uncertainty import picp


def regime_thresholds(qobs_reference: np.ndarray) -> Tuple[float, float]:
    """Devuelve (P25, P75) de la serie de referencia."""
    q = np.asarray(qobs_reference, dtype=float)
    q = q[np.isfinite(q)]
    return float(np.percentile(q, 25)), float(np.percentile(q, 75))


def regime_masks(qobs: np.ndarray, p25: float, p75: float) -> Dict[str, np.ndarray]:
    """Mascaras booleanas por regimen."""
    q = np.asarray(qobs, dtype=float)
    return {
        "low": q <= p25,
        "mid": (q > p25) & (q <= p75),
        "high": q > p75,
    }


def metrics_by_regime(
    obs: np.ndarray,
    sim: np.ndarray,
    p25: float,
    p75: float,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Tabla de metricas por regimen.

    Devuelve un DataFrame con una fila por regimen (low/mid/high) y columnas
    NSE, KGE, RMSE, MAE, PBIAS, n (y PICP si se entregan intervalos).
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    masks = regime_masks(obs, p25, p75)

    rows = []
    for name, m in masks.items():
        if m.sum() == 0:
            row = {"regime": name, "n": 0, "NSE": np.nan, "KGE": np.nan,
                   "RMSE": np.nan, "MAE": np.nan, "PBIAS": np.nan}
        else:
            row = {
                "regime": name,
                "n": int(m.sum()),
                "NSE": nse(obs[m], sim[m]),
                "KGE": kge(obs[m], sim[m]),
                "RMSE": rmse(obs[m], sim[m]),
                "MAE": mae(obs[m], sim[m]),
                "PBIAS": pbias(obs[m], sim[m]),
            }
        if lower is not None and upper is not None and m.sum() > 0:
            row["PICP"] = picp(obs[m], np.asarray(lower)[m], np.asarray(upper)[m])
        rows.append(row)
    return pd.DataFrame(rows)
