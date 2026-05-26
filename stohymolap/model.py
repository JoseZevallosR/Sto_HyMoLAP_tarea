import numpy as np


def state_basin_scalar(mu: float, lambda_: float, peff: np.ndarray) -> np.ndarray:
    peff = np.asarray(peff, dtype=float)
    n = len(peff)
    x = np.zeros(n, dtype=float)
    x[0] = peff[0]

    decay = mu / lambda_          # tasa de vaciado por paso

    for i in range(1, n):
        x[i] = x[i - 1] * (1.0 - decay) + peff[i]   # ← CORREGIDO

    return x


def state_basin_vectorized(
    mu: np.ndarray,
    lambda_: np.ndarray,
    peff: np.ndarray
) -> np.ndarray:
    mu      = np.asarray(mu,      dtype=float)
    lambda_ = np.asarray(lambda_, dtype=float)
    peff    = np.asarray(peff,    dtype=float)

    n_params = len(mu)
    n        = len(peff)
    decay    = mu / lambda_          # shape (n_params,)

    x       = np.zeros((n_params, n), dtype=float)
    x[:, 0] = peff[0]

    for i in range(1, n):
        x[:, i] = x[:, i - 1] * (1.0 - decay) + peff[i]   # ← CORREGIDO

    return x
