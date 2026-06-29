"""Diagrama de Taylor (correlacion, desviacion estandar, RMSD).

Implementacion ligera en coordenadas polares: el angulo codifica la
correlacion y el radio la desviacion estandar normalizada respecto a la
observacion.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_taylor(obs, sim, title, output_path, label="Qsim"):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[m], sim[m]
    if len(obs) < 2:
        return

    std_obs = np.std(obs)
    std_sim = np.std(sim)
    r = np.corrcoef(obs, sim)[0, 1]
    r = float(np.clip(r, -1.0, 1.0)) if np.isfinite(r) else 0.0

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    theta = np.arccos(r)
    radius = std_sim / std_obs if std_obs > 0 else 0.0

    # Punto de referencia (observacion): r=1, std normalizada=1.
    ax.plot(0, 1.0, "ko", markersize=10, label="Observacion")
    ax.plot(theta, radius, "rs", markersize=10, label=label)

    rs_ticks = np.array([0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99])
    ax.set_xticks(np.arccos(rs_ticks))
    ax.set_xticklabels([f"{v:.2f}" for v in rs_ticks])
    ax.set_ylim(0, max(1.5, radius * 1.2))
    ax.set_title(title + "\n(angulo=correlacion, radio=std normalizada)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
