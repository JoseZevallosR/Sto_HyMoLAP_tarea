"""Metricas de incertidumbre para predicciones probabilisticas.

Evaluan la calidad de los intervalos de prediccion producidos por el ensemble
estocastico (E2) o por los modelos cuantilicos hibridos.

* PICP   : Prediction Interval Coverage Probability (cobertura empirica).
* PINAW  : Prediction Interval Normalized Average Width (ancho normalizado).
* MPIW   : Mean Prediction Interval Width (ancho medio absoluto).
* Winkler: Winkler / interval score para un nivel (1 - alpha) dado.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _clean3(obs, lower, upper):
    obs = np.asarray(obs, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(lower) & np.isfinite(upper)
    return obs[mask], lower[mask], upper[mask]


def picp(obs, lower, upper) -> float:
    """Proporcion de observaciones dentro del intervalo [lower, upper]."""
    obs, lower, upper = _clean3(obs, lower, upper)
    if len(obs) == 0:
        return np.nan
    inside = (obs >= lower) & (obs <= upper)
    return float(np.mean(inside))


def mpiw(obs, lower, upper) -> float:
    """Ancho medio absoluto del intervalo."""
    obs, lower, upper = _clean3(obs, lower, upper)
    if len(obs) == 0:
        return np.nan
    return float(np.mean(upper - lower))


def pinaw(obs, lower, upper) -> float:
    """Ancho medio normalizado por el rango de las observaciones."""
    obs, lower, upper = _clean3(obs, lower, upper)
    if len(obs) == 0:
        return np.nan
    rng = np.max(obs) - np.min(obs)
    if rng == 0:
        return np.nan
    return float(np.mean(upper - lower) / rng)


def winkler_score(obs, lower, upper, alpha: float = 0.05) -> float:
    """Winkler/interval score promedio para un intervalo (1-alpha).

    Penaliza el ancho del intervalo y, adicionalmente, las observaciones que
    caen fuera de el. Menor es mejor.
    """
    obs, lower, upper = _clean3(obs, lower, upper)
    if len(obs) == 0:
        return np.nan
    width = upper - lower
    score = width.copy()
    below = obs < lower
    above = obs > upper
    score[below] += (2.0 / alpha) * (lower[below] - obs[below])
    score[above] += (2.0 / alpha) * (obs[above] - upper[above])
    return float(np.mean(score))


def all_uncertainty(obs, lower, upper, alpha: float = 0.05) -> Dict[str, float]:
    """Set completo de metricas de incertidumbre como dict."""
    return {
        "PICP": picp(obs, lower, upper),
        "PINAW": pinaw(obs, lower, upper),
        "MPIW": mpiw(obs, lower, upper),
        "Winkler": winkler_score(obs, lower, upper, alpha=alpha),
    }
