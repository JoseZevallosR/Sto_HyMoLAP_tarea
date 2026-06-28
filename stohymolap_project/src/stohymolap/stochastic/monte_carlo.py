"""Monte Carlo: ensemble estocastico RAMIS-Levy y resumen probabilistico.

Genera ``n_traj`` trayectorias de caudal total (Qfast + baseflow opcional) y
las resume en media, mediana, cuantiles y bandas. Es el insumo de los
experimentos estocasticos (E2) e hibridos (E4-E7), cuyas features de
incertidumbre se derivan de este ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .levy import stable_rvs_cms
from ..hydro.baseflow import BaseflowParams, quickflow_initial_from_total
from ..hydro.ramis import simulate_fast_ensemble
from ..hydro.water_balance import assemble_total_discharge
from ..utils.logging import get_logger

_log = get_logger("stochastic.monte_carlo")

# Cuantiles estandar reportados por el ensemble.
QUANTILE_LEVELS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


@dataclass
class EnsembleResult:
    """Resultado de un ensemble Monte Carlo."""

    ensemble: np.ndarray            # (n_tiempos, n_traj) caudal total
    q_base: np.ndarray              # (n_tiempos,) caudal base
    summary: Dict[str, np.ndarray] = field(default_factory=dict)


def summarize_ensemble(ensemble: np.ndarray, q_base: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Resume un ensemble (n_tiempos, n_traj) en estadisticos por paso de tiempo."""
    ens = np.asarray(ensemble, dtype=float)
    summary: Dict[str, np.ndarray] = {}
    summary["Qmean"] = np.nanmean(ens, axis=1)
    summary["std_ensemble"] = np.nanstd(ens, axis=1)

    qs = np.nanpercentile(ens, [q * 100 for q in QUANTILE_LEVELS], axis=1)
    for level, row in zip(QUANTILE_LEVELS, qs):
        tag = f"q{int(round(level * 100)):02d}"
        summary[tag] = row
    summary["q50"] = summary["q50"]  # alias mediana ya incluido

    # Bandas de incertidumbre.
    summary["width_q95_q05"] = summary["q95"] - summary["q05"]
    summary["width_q75_q25"] = summary["q75"] - summary["q25"]
    # Bandas para PICP al 95% (q2.5 - q97.5).
    band = np.nanpercentile(ens, [2.5, 97.5], axis=1)
    summary["q025"] = band[0]
    summary["q975"] = band[1]

    if q_base is not None:
        summary["Qbase"] = np.asarray(q_base, dtype=float)
        summary["Qfast"] = summary["Qmean"] - summary["Qbase"]
    return summary


def run_monte_carlo(
    *,
    mu: float,
    lambda_: float,
    sigma: float,
    peff: np.ndarray,
    q0: float,
    alpha_levy: float,
    beta_levy: float,
    alpha_area: float,
    n_traj: int,
    rng: np.random.Generator,
    use_baseflow: bool = False,
    baseflow_params: Optional[BaseflowParams] = None,
    q0_obs: Optional[float] = None,
    chunk_size: int = 2000,
    clamp_negative_q: bool = True,
) -> EnsembleResult:
    """Ejecuta el ensemble Monte Carlo RAMIS-Levy (+ baseflow opcional).

    Procesa por bloques (``chunk_size``) para limitar el uso de memoria con
    ensembles grandes.
    """
    peff = np.asarray(peff, dtype=float)
    n = len(peff)
    ensemble = np.zeros((n, n_traj), dtype=float)

    written = 0
    while written < n_traj:
        current = min(chunk_size, n_traj - written)
        levy_matrix = stable_rvs_cms(
            alpha=alpha_levy, beta=beta_levy, loc=0.0, scale=1.0,
            size=(current, n), rng=rng,
        )
        q0_fast = quickflow_initial_from_total(
            q0, baseflow_params, use_baseflow=use_baseflow
        )
        q_fast = simulate_fast_ensemble(
            mu=mu, lambda_=lambda_, sigma=sigma, peff=peff, q0=q0_fast,
            levy_matrix=levy_matrix, alpha_area=alpha_area,
            clamp_negative_q=clamp_negative_q,
        )  # (n, current)
        q_total, q_base = assemble_total_discharge(
            q_fast, peff,
            use_baseflow=use_baseflow, baseflow_params=baseflow_params,
            q0_obs=q0_obs, alpha_area=alpha_area, clamp_negative=clamp_negative_q,
        )
        ensemble[:, written:written + current] = q_total
        written += current
        _log.info("Monte Carlo: %d/%d trayectorias", written, n_traj)

    summary = summarize_ensemble(ensemble, q_base=q_base)
    return EnsembleResult(ensemble=ensemble, q_base=q_base, summary=summary)
