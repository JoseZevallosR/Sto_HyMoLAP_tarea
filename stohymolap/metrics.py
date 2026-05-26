import numpy as np


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


def nse_vectorized(obs: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula NSE para muchas simulaciones simultáneamente.

    sim_matrix debe tener forma:
    (n_simulaciones, n_tiempos)
    """
    obs = np.asarray(obs, dtype=float)
    sim_matrix = np.asarray(sim_matrix, dtype=float)

    denom = np.sum((obs - np.mean(obs)) ** 2)

    if denom == 0:
        return np.full(sim_matrix.shape[0], -np.inf)

    finite_rows = np.all(np.isfinite(sim_matrix), axis=1)

    sse = np.sum((sim_matrix - obs[None, :]) ** 2, axis=1)
    out = 1.0 - sse / denom

    out[~finite_rows] = -np.inf

    return out


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    return np.sqrt(np.mean((obs - sim) ** 2))


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


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

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

    return 1.0 - np.sqrt(
        (r - 1.0) ** 2 +
        (alpha - 1.0) ** 2 +
        (beta - 1.0) ** 2
    )

