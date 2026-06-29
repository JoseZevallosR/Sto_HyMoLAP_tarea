"""Nucleo RAMIS / StoHyMoLAP: ecuacion de estado y caudal rapido (Qfast).

Consolida y generaliza la logica que en el codigo original vivia en
``model.py``, ``calibration.py`` (``simulate_candidates_paper``) y
``validation.py`` (``simulate_fixed_params_ensemble``). Se separa el caudal
*rapido* (la respuesta directa del modelo) del caudal *base* (reservorio
lineal en ``baseflow.py``), de modo que la memoria hidrologica se pueda
activar o desactivar de forma independiente.

Parametros del modelo:
    mu, lambda_ : controlan el estado de la cuenca y el vaciado.
    sigma       : amplitud del ruido estocastico (Levy).
    alpha_area  : factor de escala Peff (mm/dia) -> unidades de Q.

El ruido Levy se inyecta externamente (modulo ``stochastic.levy``) para que la
misma maquinaria sirva tanto al caso deterministico (sigma=0 o sin ruido) como
al estocastico.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.logging import get_logger

_log = get_logger("hydro.ramis")


def validate_ramis_parameters(
    mu,
    lambda_,
    sigma=None,
    alpha_area: Optional[float] = None,
    *,
    require_positive_exponent: bool = True,
) -> None:
    """Valida rangos numericos/estructurales del nucleo RAMIS.

    Condiciones usadas por el modelo diario:
    - ``lambda`` debe ser positiva.
    - ``mu`` debe ser positiva y, por defecto, mayor que 0.5 para que
      ``2*mu-1`` sea positivo.
    - ``0 < mu/lambda < 1`` para que el estado de cuenca tenga memoria
      estable y no cambie de signo por el factor ``1 - mu/lambda``.
    - ``sigma`` y ``alpha_area`` no pueden ser negativos.
    """
    mu_a = np.asarray(mu, dtype=float)
    lam_a = np.asarray(lambda_, dtype=float)
    if mu_a.shape != lam_a.shape and mu_a.size != 1 and lam_a.size != 1:
        raise ValueError("mu y lambda_ deben ser escalares o broadcast compatibles.")
    if not np.all(np.isfinite(mu_a)) or not np.all(np.isfinite(lam_a)):
        raise ValueError("mu y lambda_ deben ser finitos.")
    if np.any(mu_a <= 0):
        raise ValueError("mu debe ser positivo.")
    if require_positive_exponent and np.any(mu_a <= 0.5):
        raise ValueError("mu debe ser > 0.5 para mantener exponente 2*mu-1 positivo.")
    if np.any(lam_a <= 0):
        raise ValueError("lambda_ debe ser positivo.")
    decay = mu_a / lam_a
    if np.any(decay <= 0) or np.any(decay >= 1):
        raise ValueError("RAMIS requiere 0 < mu/lambda < 1 para memoria estable.")
    if sigma is not None:
        sig_a = np.asarray(sigma, dtype=float)
        if not np.all(np.isfinite(sig_a)) or np.any(sig_a < 0):
            raise ValueError("sigma debe ser finito y no negativo.")
    if alpha_area is not None:
        aa = float(alpha_area)
        if not np.isfinite(aa) or aa <= 0:
            raise ValueError("alpha_area debe ser finito y positivo.")


def validate_ramis_bounds(bounds: dict) -> None:
    """Valida que los rangos de calibracion no permitan RAMIS inestable."""
    mu_min, mu_max = bounds["mu"]
    lam_min, lam_max = bounds["lambda"]
    sig_min, sig_max = bounds.get("sigma", (0.0, 0.0))
    validate_ramis_parameters(
        np.array([mu_min, mu_max]),
        np.array([lam_min, lam_max]),
        sigma=np.array([sig_min, sig_max]),
    )
    if mu_max / lam_min >= 1.0:
        raise ValueError(
            "Bounds RAMIS inestables: mu_max/lambda_min debe ser < 1."
        )


def state_basin(mu: float, lambda_: float, peff: np.ndarray) -> np.ndarray:
    """Estado de la cuenca x_t (version escalar de un parametro).

    x_t = x_{t-1} * (1 - mu/lambda) + Peff_t
    """
    validate_ramis_parameters(mu, lambda_)
    peff = np.asarray(peff, dtype=float)
    n = len(peff)
    x = np.zeros(n, dtype=float)
    x[0] = peff[0]
    decay = mu / lambda_
    for i in range(1, n):
        x[i] = x[i - 1] * (1.0 - decay) + peff[i]
    return x


def state_basin_vectorized(
    mu: np.ndarray, lambda_: np.ndarray, peff: np.ndarray
) -> np.ndarray:
    """Estado de la cuenca para muchos juegos de parametros a la vez.

    Devuelve matriz (n_params, n_tiempos).
    """
    validate_ramis_parameters(mu, lambda_)
    mu = np.asarray(mu, dtype=float)
    lambda_ = np.asarray(lambda_, dtype=float)
    peff = np.asarray(peff, dtype=float)

    n_params = len(mu)
    n = len(peff)
    decay = mu / lambda_

    x = np.zeros((n_params, n), dtype=float)
    x[:, 0] = peff[0]
    for i in range(1, n):
        x[:, i] = x[:, i - 1] * (1.0 - decay) + peff[i]
    return x


def simulate_fast_candidates(
    mu: np.ndarray,
    lambda_: np.ndarray,
    sigma: np.ndarray,
    peff: np.ndarray,
    q0: float,
    levy_values: np.ndarray,
    alpha_area: float,
    clamp_negative_q: bool = True,
) -> np.ndarray:
    """Qfast para muchos candidatos (1 realizacion Levy compartida).

    Reproduce ``simulate_candidates_paper`` del codigo original. Devuelve
    matriz (n_params, n_tiempos).
    """
    validate_ramis_parameters(mu, lambda_, sigma=sigma, alpha_area=alpha_area)
    mu = np.asarray(mu, dtype=float)
    lambda_ = np.asarray(lambda_, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    peff = np.asarray(peff, dtype=float)
    levy_values = np.asarray(levy_values, dtype=float)

    n_params = len(mu)
    n = len(peff)
    q0_safe = max(float(q0), 1e-6)

    qsim = np.zeros((n_params, n), dtype=float)
    qsim[:, 0] = q0_safe

    x_state = state_basin_vectorized(mu, lambda_, peff)

    ratio = mu / lambda_
    inv_lambda = 1.0 / lambda_
    exponent = 2.0 * mu - 1.0
    dlev = levy_values[1:] - levy_values[:-1]

    for k in range(1, n):
        q_prev = qsim[:, k - 1]
        if clamp_negative_q:
            q_prev = np.maximum(q_prev, 0.0)
            qsim[:, k - 1] = q_prev

        q_norm = q_prev / q0_safe
        q_power = np.power(np.maximum(q_norm, 0.0), exponent) * q0_safe

        qsim[:, k] = (
            q_prev
            - ratio * q_power
            + inv_lambda * x_state[:, k - 1] * alpha_area
            + sigma * q_prev * dlev[k - 1]
        )
        bad = ~np.isfinite(qsim[:, k])
        if np.any(bad):
            qsim[bad, k] = np.nan

    if clamp_negative_q:
        qsim[:, -1] = np.maximum(qsim[:, -1], 0.0)
    return qsim


def simulate_fast_ensemble(
    mu: float,
    lambda_: float,
    sigma: float,
    peff: np.ndarray,
    q0: float,
    levy_matrix: np.ndarray,
    alpha_area: float,
    clamp_negative_q: bool = True,
) -> np.ndarray:
    """Qfast para parametros fijos y un ensemble de realizaciones Levy.

    ``levy_matrix`` tiene forma (n_traj, n_tiempos). Devuelve matriz
    (n_tiempos, n_traj) para ser consistente con el codigo original de
    validacion.
    """
    validate_ramis_parameters(mu, lambda_, sigma=sigma, alpha_area=alpha_area)
    peff = np.asarray(peff, dtype=float)
    levy_matrix = np.asarray(levy_matrix, dtype=float)
    n_traj, n = levy_matrix.shape
    q0_safe = max(float(q0), 1e-6)

    x_state = state_basin(mu, lambda_, peff)
    ratio = mu / lambda_
    inv_lambda = 1.0 / lambda_
    exponent = 2.0 * mu - 1.0

    s = np.zeros((n_traj, n), dtype=float)
    s[:, 0] = q0_safe
    for k in range(1, n):
        s_prev = s[:, k - 1]
        if clamp_negative_q:
            s_prev = np.maximum(s_prev, 0.0)
            s[:, k - 1] = s_prev
        dlev = levy_matrix[:, k] - levy_matrix[:, k - 1]
        q_norm = s_prev / q0_safe
        q_power = np.power(np.maximum(q_norm, 0.0), exponent) * q0_safe
        s[:, k] = (
            s_prev
            - ratio * q_power
            + inv_lambda * x_state[k - 1] * alpha_area
            + sigma * s_prev * dlev
        )
        bad = ~np.isfinite(s[:, k])
        if np.any(bad):
            s[bad, k] = np.nan
    if clamp_negative_q:
        s[:, -1] = np.maximum(s[:, -1], 0.0)
    return s.T  # (n_tiempos, n_traj)


def simulate_fast_deterministic(
    mu: float,
    lambda_: float,
    peff: np.ndarray,
    q0: float,
    alpha_area: float,
    clamp_negative_q: bool = True,
) -> np.ndarray:
    """Qfast deterministico (sin ruido Levy, sigma=0).

    Caso usado por E1. Devuelve un vector (n_tiempos,).
    """
    n = len(peff)
    zero_levy = np.zeros((1, n), dtype=float)
    out = simulate_fast_ensemble(
        mu, lambda_, 0.0, peff, q0, zero_levy, alpha_area, clamp_negative_q
    )
    return out[:, 0]
