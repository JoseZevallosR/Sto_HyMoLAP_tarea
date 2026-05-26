import numpy as np
from typing import Callable, Dict


# ============================================================
# METRICS
# ============================================================

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan

    return 1.0 - np.sum((obs - sim) ** 2) / denom


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    return np.sqrt(np.mean((obs - sim) ** 2))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    return np.mean(np.abs(obs - sim))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    denom = np.sum(obs)
    if denom == 0:
        return np.nan

    return 100.0 * np.sum(sim - obs) / denom


def log_nse(obs: np.ndarray, sim: np.ndarray, eps: float = 1e-6) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    if np.any(obs < 0) or np.any(sim < 0):
        return np.nan

    obs_log = np.log(obs + eps)
    sim_log = np.log(sim + eps)

    denom = np.sum((obs_log - np.mean(obs_log)) ** 2)
    if denom == 0:
        return np.nan

    return 1.0 - np.sum((obs_log - sim_log) ** 2) / denom


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 2:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    if not np.isfinite(r):
        return np.nan

    s_obs = np.std(obs, ddof=1)
    s_sim = np.std(sim, ddof=1)
    m_obs = np.mean(obs)
    m_sim = np.mean(sim)

    alpha = s_sim / s_obs if s_obs > 0 else np.nan
    beta = m_sim / m_obs if m_obs != 0 else np.nan

    if not (np.isfinite(alpha) and np.isfinite(beta)):
        return np.nan

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


# ============================================================
# METRIC REGISTRY
# ============================================================

METRIC_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "nse": {
        "func": nse,
        "sense": "max"
    },
    "lognse": {
        "func": log_nse,
        "sense": "max"
    },
    "kge": {
        "func": kge,
        "sense": "max"
    },
    "rmse": {
        "func": rmse,
        "sense": "min"
    },
    "mae": {
        "func": mae,
        "sense": "min"
    },
    "pbias_abs": {
        "func": lambda obs, sim: abs(pbias(obs, sim)),
        "sense": "min"
    }
}