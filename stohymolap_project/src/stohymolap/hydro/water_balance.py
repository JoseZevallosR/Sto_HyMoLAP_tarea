"""Ensamblaje del balance hidrico: Qfast + Qbase -> Qtotal.

Capa fina de conveniencia que decide, segun la configuracion del experimento,
si se anade el reservorio de baseflow al caudal rapido del modelo RAMIS.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .baseflow import BaseflowParams, add_baseflow
from ..utils.logging import get_logger

_log = get_logger("hydro.water_balance")


def assemble_total_discharge(
    q_fast: np.ndarray,
    peff: np.ndarray,
    *,
    use_baseflow: bool,
    baseflow_params: Optional[BaseflowParams] = None,
    q0_obs: Optional[float] = None,
    alpha_area: float = 1.0,
    clamp_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (Q_total, Q_base).

    Si ``use_baseflow`` es False, Q_base es un vector de ceros y
    Q_total = Q_fast (recortado a no negativos si corresponde).
    """
    q_fast = np.asarray(q_fast, dtype=float)
    if use_baseflow:
        params = baseflow_params or BaseflowParams()
        q_total, q_base = add_baseflow(
            q_fast, peff, params, q0_obs=q0_obs, alpha_area=alpha_area,
            clamp_negative=clamp_negative
        )
        return q_total, q_base

    q_base = np.zeros(q_fast.shape[0], dtype=float)
    q_total = np.maximum(q_fast, 0.0) if clamp_negative else q_fast
    return q_total, q_base
