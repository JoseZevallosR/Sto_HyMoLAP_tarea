"""Reservorio lineal de caudal base (memoria hidrologica).

Implementa un reservorio lineal diario que agrega persistencia/memoria a la
respuesta rapida del modelo RAMIS. Es el aporte conceptual central de las
ablaciones A1/A2 y se usa en E1, E2, E4-E7.

Ecuaciones (paso diario):

    Ep_t     = max(P_t - PET_t, 0)        # precipitacion efectiva
    R_t      = c_r * Ep_t                  # recarga al reservorio
    S_b[t+1] = max(0, S_b[t] + R_t - k_b * S_b[t])
    Q_b[t]   = k_b * S_b[t]                # caudal base
    Q_total  = max(Q_fast + Q_b, 0)        # caudal total

Parametros:
    c_r  : coeficiente de recarga,  rango [0, 1].
    k_b  : coeficiente de descarga, rango [0.001, 0.5].
    S0_b : almacenamiento inicial. Si es None se inicializa como una fraccion
           del caudal observado inicial (Q0 / k_b en equilibrio aproximado).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

_log = get_logger("hydro.baseflow")


@dataclass
class BaseflowParams:
    """Parametros del reservorio lineal de baseflow."""

    c_r: float = 0.3
    k_b: float = 0.05
    S0_b: Optional[float] = None

    def clipped(self) -> "BaseflowParams":
        """Devuelve una copia con parametros recortados a sus rangos validos."""
        return BaseflowParams(
            c_r=float(np.clip(self.c_r, 0.0, 1.0)),
            k_b=float(np.clip(self.k_b, 0.001, 0.5)),
            S0_b=self.S0_b,
        )


def linear_reservoir(
    peff: np.ndarray,
    params: BaseflowParams,
    q0_obs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simula el reservorio lineal y devuelve (Q_base, S_storage).

    Parameters
    ----------
    peff:
        Precipitacion efectiva Ep_t = max(P - PET, 0), longitud n.
    params:
        Parametros del reservorio.
    q0_obs:
        Caudal observado inicial, usado para inicializar S0_b si no se entrega.

    Returns
    -------
    (Q_base, S) : tuple de np.ndarray
        Caudal base y almacenamiento, ambos de longitud n.
    """
    p = params.clipped()
    peff = np.asarray(peff, dtype=float)
    n = len(peff)

    if p.S0_b is not None:
        s0 = float(p.S0_b)
    elif q0_obs is not None and p.k_b > 0:
        # En equilibrio Q_b = k_b * S  =>  S0 ~ Q0_base / k_b. Usamos una
        # fraccion conservadora (20%) del caudal observado inicial como base.
        s0 = max(0.0, 0.2 * float(q0_obs) / p.k_b)
    else:
        s0 = 0.0

    S = np.zeros(n, dtype=float)
    Q_base = np.zeros(n, dtype=float)
    S[0] = s0
    Q_base[0] = p.k_b * S[0]

    for t in range(1, n):
        R = p.c_r * peff[t - 1]
        S[t] = max(0.0, S[t - 1] + R - p.k_b * S[t - 1])
        Q_base[t] = p.k_b * S[t]

    return Q_base, S


def add_baseflow(
    q_fast: np.ndarray,
    peff: np.ndarray,
    params: BaseflowParams,
    q0_obs: Optional[float] = None,
    clamp_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combina Qfast con el caudal base del reservorio.

    Acepta ``q_fast`` 1D (n,) o 2D (n, n_traj) y suma el caudal base
    (broadcasting sobre las trayectorias en el caso 2D).

    Returns
    -------
    (Q_total, Q_base)
    """
    q_fast = np.asarray(q_fast, dtype=float)
    q_base, _ = linear_reservoir(peff, params, q0_obs=q0_obs)

    if q_fast.ndim == 1:
        q_total = q_fast + q_base
    elif q_fast.ndim == 2:
        q_total = q_fast + q_base[:, None]
    else:
        raise ValueError("q_fast debe ser 1D (n,) o 2D (n, n_traj).")

    if clamp_negative:
        q_total = np.maximum(q_total, 0.0)
    return q_total, q_base
