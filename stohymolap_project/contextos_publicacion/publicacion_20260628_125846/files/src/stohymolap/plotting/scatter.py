"""Diagrama de dispersion Qobs vs Qsim con linea 1:1."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_scatter(obs, sim, title, output_path):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[m], sim[m]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs, sim, s=10, alpha=0.4, color="tab:blue")
    lim = [0, max(obs.max(), sim.max()) * 1.05] if len(obs) else [0, 1]
    ax.plot(lim, lim, "k--", linewidth=1, label="1:1")
    ax.set_xlabel("Qobs (m3/s)")
    ax.set_ylabel("Qsim (m3/s)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
