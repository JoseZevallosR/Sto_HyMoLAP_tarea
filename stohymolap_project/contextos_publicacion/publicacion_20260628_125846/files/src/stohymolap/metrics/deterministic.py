"""Metricas deterministicas: NSE, KGE, RMSE, MAE, PBIAS, R2.

Extiende el ``metrics.py`` original (que solo tenia NSE, KGE, RMSE, PBIAS y
NSE vectorizado) anadiendo MAE y R2, y una funcion ``all_deterministic`` que
devuelve el set completo como diccionario.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _clean(obs: np.ndarray, sim: np.ndarray):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / denom


def nse_vectorized(obs: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:
    """NSE para muchas simulaciones (n_sim, n_tiempos).

    Fase 1: permite ``NaN`` en ``obs`` y evalúa solo contra observaciones
    finitas. Esto evita rellenar ``Qobs`` con ``Qsim`` para calibrar.
    """
    obs = np.asarray(obs, dtype=float)
    sim_matrix = np.asarray(sim_matrix, dtype=float)
    obs_mask = np.isfinite(obs)
    if obs_mask.sum() == 0:
        return np.full(sim_matrix.shape[0], -np.inf)
    obs_valid = obs[obs_mask]
    sim_valid = sim_matrix[:, obs_mask]
    denom = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
    if denom == 0:
        return np.full(sim_matrix.shape[0], -np.inf)
    finite_rows = np.all(np.isfinite(sim_valid), axis=1)
    sse = np.sum((sim_valid - obs_valid[None, :]) ** 2, axis=1)
    out = 1.0 - sse / denom
    out[~finite_rows] = -np.inf
    return out


def kge_vectorized(obs: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:
    """KGE para muchas simulaciones (n_sim, n_tiempos).

    Evalua solo contra observaciones finitas y marca con ``-inf`` las filas
    simuladas no finitas o degeneradas. Usa la misma formulacion que ``kge``.
    """
    obs = np.asarray(obs, dtype=float)
    sim_matrix = np.asarray(sim_matrix, dtype=float)
    obs_mask = np.isfinite(obs)
    if obs_mask.sum() < 2:
        return np.full(sim_matrix.shape[0], -np.inf)

    obs_valid = obs[obs_mask]
    sim_valid = sim_matrix[:, obs_mask]
    finite_rows = np.all(np.isfinite(sim_valid), axis=1)

    mean_obs = np.mean(obs_valid)
    std_obs = np.std(obs_valid, ddof=1)
    if std_obs == 0 or mean_obs == 0:
        return np.full(sim_matrix.shape[0], -np.inf)

    mean_sim = np.mean(sim_valid, axis=1)
    std_sim = np.std(sim_valid, axis=1, ddof=1)
    centered_obs = obs_valid - mean_obs
    centered_sim = sim_valid - mean_sim[:, None]
    cov = np.sum(centered_sim * centered_obs[None, :], axis=1) / (len(obs_valid) - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / (std_sim * std_obs)
        alpha = std_sim / std_obs
        beta = mean_sim / mean_obs
        out = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    bad = (~finite_rows) | (~np.isfinite(out)) | (std_sim == 0)
    out[bad] = -np.inf
    return out


def pbias_vectorized(obs: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:
    """PBIAS (%) para muchas simulaciones (n_sim, n_tiempos)."""
    obs = np.asarray(obs, dtype=float)
    sim_matrix = np.asarray(sim_matrix, dtype=float)
    obs_mask = np.isfinite(obs)
    if obs_mask.sum() == 0:
        return np.full(sim_matrix.shape[0], np.inf)
    obs_valid = obs[obs_mask]
    sim_valid = sim_matrix[:, obs_mask]
    denom = np.sum(obs_valid)
    if denom == 0:
        return np.full(sim_matrix.shape[0], np.inf)
    finite_rows = np.all(np.isfinite(sim_valid), axis=1)
    out = 100.0 * np.sum(sim_valid - obs_valid[None, :], axis=1) / denom
    out[~finite_rows] = np.inf
    return out


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    return float(np.mean(np.abs(obs - sim)))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    denom = np.sum(obs)
    if denom == 0:
        return np.nan
    return float(100.0 * np.sum(sim - obs) / denom)


def r2(obs: np.ndarray, sim: np.ndarray) -> float:
    """Coeficiente de determinacion (cuadrado del coef. de correlacion)."""
    obs, sim = _clean(obs, sim)
    if len(obs) < 2:
        return np.nan
    if np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    return float(r ** 2) if np.isfinite(r) else np.nan


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _clean(obs, sim)
    if len(obs) < 2:
        return np.nan
    std_obs = np.std(obs, ddof=1)
    std_sim = np.std(sim, ddof=1)
    if std_obs == 0 or std_sim == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    if not np.isfinite(r):
        return np.nan
    alpha = std_sim / std_obs
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def all_deterministic(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Devuelve el set completo de metricas deterministicas como dict."""
    return {
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "RMSE": rmse(obs, sim),
        "MAE": mae(obs, sim),
        "PBIAS": pbias(obs, sim),
        "R2": r2(obs, sim),
    }
