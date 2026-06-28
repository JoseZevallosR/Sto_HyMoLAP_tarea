"""Calibracion Monte Carlo de RAMIS (mu, lambda, sigma, alpha, beta + baseflow).

Refactorizacion de ``calibrate_paper_heuristic`` del codigo original con dos
extensiones:

1. Reservorio de baseflow opcional y calibrable (c_r, k_b, S0_b): cada
   trayectoria sortea sus parametros de baseflow y el caudal total
   (Qfast + Qbase) es el que se evalua contra Qobs.
2. Seleccion de parametros finales por objetivo multiobjetivo: el top-K de
   trayectorias se elige por el menor ``J`` (no por NSE puro), de modo que la
   eleccion respete tambien KGE, PBIAS y, cuando aplica, la cobertura.

La pantalla por-candidato dentro de cada trayectoria sigue usando NSE
vectorizado (rapido) para escoger el mejor mu/lambda/sigma; el costo de evaluar
J completo se reserva para el ranking entre trayectorias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .objective import objective_value
from ..hydro.baseflow import BaseflowParams, add_baseflow
from ..hydro.ramis import simulate_fast_candidates
from ..metrics.deterministic import nse_vectorized
from ..stochastic.levy import stable_rvs_cms
from ..utils.logging import get_logger

_log = get_logger("calibration.monte_carlo_search")


@dataclass
class CalibrationResult:
    """Contenedor de resultados de calibracion."""

    best_params: Dict[str, float]
    top_k_table: "np.ndarray"
    qq_best: np.ndarray            # (n, T) mejor candidato por trayectoria (Qtotal)
    mean_trajectory: np.ndarray
    inf_trajectory: np.ndarray
    sup_trajectory: np.ndarray
    alpha_area: float
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]


def _alpha_area(discharge: np.ndarray, peff: np.ndarray) -> float:
    q = np.asarray(discharge, dtype=float)
    q = q[np.isfinite(q)]
    peff = np.asarray(peff, dtype=float)
    peff_mean = np.mean(peff[peff > 0]) if np.any(peff > 0) else 1.0
    q_mean = np.mean(q) if q.size else 1.0
    return float(q_mean / peff_mean)


def calibrate(
    discharge: np.ndarray,
    peff: np.ndarray,
    *,
    bounds: Dict[str, Any],
    n_traj: int,
    n_param_samples: int,
    top_frac: float,
    seed: int,
    use_baseflow: bool = False,
    calibrate_baseflow: bool = False,
    objective_weights: Optional[Dict[str, float]] = None,
    clamp_negative_q: bool = True,
) -> CalibrationResult:
    """Ejecuta la calibracion Monte Carlo.

    ``bounds`` debe contener: mu, lambda, sigma, alpha, beta y (si aplica)
    c_r, k_b, S0_b, cada uno como tupla (min, max).
    """
    discharge = np.asarray(discharge, dtype=float)
    peff = np.asarray(peff, dtype=float)
    if len(discharge) != len(peff):
        raise ValueError(f"discharge y peff deben tener la misma longitud: {len(discharge)} vs {len(peff)}")
    valid_obs = np.isfinite(discharge)
    if valid_obs.sum() < 20:
        raise ValueError(
            "Muy pocos Qobs finitos para calibrar RAMIS "
            f"({int(valid_obs.sum())} registros)."
        )
    n = len(discharge)
    q0 = float(discharge[np.argmax(valid_obs)])
    rng = np.random.default_rng(seed)
    alpha_area = _alpha_area(discharge, peff)

    T = int(n_traj)
    store = {
        k: np.zeros(T) for k in
        ["mu", "lambda", "sigma", "alpha", "beta", "c_r", "k_b", "S0_b", "nse", "J"]
    }
    qq_best = np.zeros((n, T), dtype=float)

    _log.info("Calibracion MC: %d trayectorias x %d candidatos | baseflow=%s",
              T, n_param_samples, use_baseflow)

    for traj in range(T):
        alpha_traj = rng.uniform(*bounds["alpha"])
        beta_traj = rng.uniform(*bounds["beta"])
        lev = stable_rvs_cms(alpha=alpha_traj, beta=beta_traj, loc=0.0,
                             scale=1.0, size=n, rng=rng)

        mu_c = rng.uniform(*bounds["mu"], size=n_param_samples)
        lam_c = rng.uniform(*bounds["lambda"], size=n_param_samples)
        sig_c = rng.uniform(*bounds["sigma"], size=n_param_samples)

        q_fast = simulate_fast_candidates(
            mu=mu_c, lambda_=lam_c, sigma=sig_c, peff=peff, q0=q0,
            levy_values=lev, alpha_area=alpha_area, clamp_negative_q=clamp_negative_q,
        )  # (n_param_samples, n)

        # Baseflow opcional (mismos params para todos los candidatos de la traj).
        if use_baseflow:
            if calibrate_baseflow:
                bf = BaseflowParams(
                    c_r=rng.uniform(*bounds["c_r"]),
                    k_b=rng.uniform(*bounds["k_b"]),
                    S0_b=rng.uniform(*bounds["S0_b"]),
                )
            else:
                bf = BaseflowParams()
            # add_baseflow espera (n,) o (n, n_traj); transponemos a (n, cand).
            q_total_T, q_base = add_baseflow(
                q_fast.T, peff, bf, q0_obs=q0, clamp_negative=clamp_negative_q
            )
            q_total = q_total_T.T  # (n_param_samples, n)
        else:
            bf = BaseflowParams(c_r=0.0, k_b=0.001, S0_b=0.0)
            q_total = q_fast

        scores = nse_vectorized(discharge, q_total)
        if np.all(~np.isfinite(scores)):
            raise RuntimeError(f"Todas las simulaciones fallaron (traj {traj}).")
        idx = int(np.nanargmax(scores))

        best_series = q_total[idx, :]
        qq_best[:, traj] = best_series

        # J por trayectoria (sin termino de cobertura: serie unica).
        jval = objective_value(discharge, best_series, weights=objective_weights)

        store["mu"][traj] = mu_c[idx]
        store["lambda"][traj] = lam_c[idx]
        store["sigma"][traj] = sig_c[idx]
        store["alpha"][traj] = alpha_traj
        store["beta"][traj] = beta_traj
        store["c_r"][traj] = bf.c_r
        store["k_b"][traj] = bf.k_b
        store["S0_b"][traj] = bf.S0_b if bf.S0_b is not None else np.nan
        store["nse"][traj] = scores[idx]
        store["J"][traj] = jval["J"]

        if (traj + 1) % max(1, T // 10) == 0 or traj == 0:
            _log.info("  traj %d/%d | NSE=%.4f | J=%.4f", traj + 1, T,
                      store["nse"][traj], store["J"][traj])

    # Seleccion top-K por menor J (multiobjetivo).
    top_frac = min(max(float(top_frac), 0.0), 1.0)
    k = max(1, int(round(T * top_frac)))
    finite = np.isfinite(store["J"])
    order = np.argsort(np.where(finite, store["J"], np.inf))  # menor J primero
    top_idx = order[:k]

    best_params = {
        "mu": float(np.mean(store["mu"][top_idx])),
        "lambda": float(np.mean(store["lambda"][top_idx])),
        "sigma": float(np.mean(store["sigma"][top_idx])),
        "alpha": float(np.mean(store["alpha"][top_idx])),
        "beta": float(np.mean(store["beta"][top_idx])),
        "c_r": float(np.mean(store["c_r"][top_idx])),
        "k_b": float(np.mean(store["k_b"][top_idx])),
        "S0_b": float(np.nanmean(store["S0_b"][top_idx])),
        "n_top": int(k),
    }

    # Bandas/resumen con TODAS las trayectorias (espectro completo).
    mean_traj = np.nanmean(qq_best, axis=1)
    inf_traj = np.nanpercentile(qq_best, 2.5, axis=1)
    sup_traj = np.nanpercentile(qq_best, 97.5, axis=1)

    metrics = objective_value(
        discharge, mean_traj, lower=inf_traj, upper=sup_traj,
        weights=objective_weights,
    )

    top_table = np.column_stack([
        store["mu"][top_idx], store["lambda"][top_idx], store["sigma"][top_idx],
        store["alpha"][top_idx], store["beta"][top_idx], store["c_r"][top_idx],
        store["k_b"][top_idx], store["S0_b"][top_idx], store["nse"][top_idx],
        store["J"][top_idx],
    ])

    return CalibrationResult(
        best_params=best_params,
        top_k_table=top_table,
        qq_best=qq_best,
        mean_trajectory=mean_traj,
        inf_trajectory=inf_traj,
        sup_trajectory=sup_traj,
        alpha_area=alpha_area,
        metrics=metrics,
        diagnostics={"store": store, "top_idx": top_idx},
    )
