"""Generador de variables alpha-estables (Levy) por Chambers-Mallows-Stuck.

Migrado sin cambios funcionales desde ``levy.py``. Reemplaza a
``scipy.stats.levy_stable.rvs`` usando solo NumPy.
"""
import numpy as np


def stable_rvs_cms(alpha, beta, loc=0.0, scale=1.0, size=None, rng=None):
    """Variables aleatorias alpha-estables (metodo CMS)."""
    if rng is None:
        rng = np.random.default_rng()

    alpha = float(alpha)
    beta = float(beta)
    scale = float(scale)

    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha debe estar en (0, 2].")
    if not (-1.0 <= beta <= 1.0):
        raise ValueError("beta debe estar en [-1, 1].")
    if scale <= 0:
        raise ValueError("scale debe ser positivo.")

    if np.isclose(alpha, 2.0):
        x = np.sqrt(2.0) * rng.normal(0.0, 1.0, size=size)
        return loc + scale * x

    u = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=size)
    w = rng.exponential(1.0, size=size)

    if np.isclose(alpha, 1.0):
        part1 = (np.pi / 2.0 + beta * u) * np.tan(u)
        part2 = beta * np.log(
            (np.pi / 2.0 * w * np.cos(u)) / (np.pi / 2.0 + beta * u)
        )
        x = (2.0 / np.pi) * (part1 - part2)
        return loc + scale * x

    zeta = beta * np.tan(np.pi * alpha / 2.0)
    b_alpha = np.arctan(zeta) / alpha
    s_alpha = (1.0 + zeta ** 2) ** (1.0 / (2.0 * alpha))

    numerator = np.sin(alpha * (u + b_alpha))
    denominator = np.cos(u) ** (1.0 / alpha)
    factor = (np.cos(u - alpha * (u + b_alpha)) / w) ** ((1.0 - alpha) / alpha)

    x = s_alpha * numerator / denominator * factor
    return loc + scale * x
