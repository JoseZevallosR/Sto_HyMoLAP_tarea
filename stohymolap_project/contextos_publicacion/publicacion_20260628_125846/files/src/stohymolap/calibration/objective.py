"""Funcion objetivo multiobjetivo para calibracion.

Reemplaza la optimizacion exclusiva de NSE del codigo original por una funcion
escalar que combina ajuste deterministico y calidad de la incertidumbre:

    J = w_nse  * (1 - NSE)
      + w_kge  * (1 - KGE)
      + w_pbias* |PBIAS / 100|
      + w_cov  * |target_coverage - PICP|

Menor J es mejor. Si el experimento no tiene incertidumbre, el termino de
cobertura se omite (PICP = NaN) y los pesos se renormalizan sobre los terminos
disponibles.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..metrics.deterministic import nse, kge, pbias, nse_vectorized, kge_vectorized, pbias_vectorized
from ..metrics.uncertainty import picp

DEFAULT_WEIGHTS = {"nse": 0.40, "kge": 0.30, "pbias": 0.15, "coverage": 0.15}
TARGET_COVERAGE = 0.95


def objective_value(
    obs: np.ndarray,
    sim_mean: np.ndarray,
    *,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    target_coverage: float = TARGET_COVERAGE,
) -> Dict[str, float]:
    """Calcula J y sus componentes.

    Returns
    -------
    dict con claves: J, NSE, KGE, PBIAS, PICP (PICP = NaN si no hay intervalos).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    nse_v = nse(obs, sim_mean)
    kge_v = kge(obs, sim_mean)
    pbias_v = pbias(obs, sim_mean)

    has_uncertainty = lower is not None and upper is not None
    picp_v = picp(obs, lower, upper) if has_uncertainty else np.nan

    terms = {
        "nse": w["nse"] * (1.0 - (nse_v if np.isfinite(nse_v) else -1.0)),
        "kge": w["kge"] * (1.0 - (kge_v if np.isfinite(kge_v) else -1.0)),
        "pbias": w["pbias"] * abs((pbias_v if np.isfinite(pbias_v) else 100.0) / 100.0),
    }
    active_weight = w["nse"] + w["kge"] + w["pbias"]
    if has_uncertainty and np.isfinite(picp_v):
        terms["coverage"] = w["coverage"] * abs(target_coverage - picp_v)
        active_weight += w["coverage"]

    # Renormaliza por el peso activo para que J sea comparable entre
    # experimentos con y sin termino de cobertura.
    J = sum(terms.values()) / active_weight if active_weight > 0 else np.nan

    return {
        "J": float(J),
        "NSE": float(nse_v),
        "KGE": float(kge_v),
        "PBIAS": float(pbias_v),
        "PICP": float(picp_v) if np.isfinite(picp_v) else np.nan,
    }


def objective_value_vectorized(
    obs: np.ndarray,
    sim_matrix: np.ndarray,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Calcula J sin cobertura para muchas simulaciones.

    Esta funcion se usa dentro de la calibracion para escoger el mejor
    candidato de parametros con la misma funcion objetivo que se usa para
    ordenar trayectorias. Antes se escogia el candidato por NSE puro y solo
    despues se calculaba J; eso podia desalinear la calibracion cuando los
    pesos daban importancia a KGE o PBIAS.
    """
    sims = np.asarray(sim_matrix, dtype=float)
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    nse_v = nse_vectorized(obs, sims)
    kge_v = kge_vectorized(obs, sims)
    pbias_v = pbias_vectorized(obs, sims)

    nse_for_j = np.where(np.isfinite(nse_v), nse_v, -1.0)
    kge_for_j = np.where(np.isfinite(kge_v), kge_v, -1.0)
    pbias_for_j = np.where(np.isfinite(pbias_v), pbias_v, 100.0)

    active_weight = float(w["nse"] + w["kge"] + w["pbias"])
    if active_weight <= 0:
        raise ValueError("Los pesos nse+kge+pbias deben sumar más que cero para calibrar candidatos.")

    J = (
        w["nse"] * (1.0 - nse_for_j)
        + w["kge"] * (1.0 - kge_for_j)
        + w["pbias"] * np.abs(pbias_for_j / 100.0)
    ) / active_weight

    nonfinite_sim = ~np.all(np.isfinite(sims[:, np.isfinite(np.asarray(obs, dtype=float))]), axis=1)
    J[nonfinite_sim] = np.inf

    return {
        "J": J.astype(float),
        "NSE": nse_v.astype(float),
        "KGE": kge_v.astype(float),
        "PBIAS": pbias_v.astype(float),
    }
