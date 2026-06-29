"""Banda de incertidumbre y residuos por regimen."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_uncertainty_band(obs, mean, lower, upper, title, output_path, dates=None):
    obs = np.asarray(obs, dtype=float)
    t = pd.to_datetime(np.asarray(dates)) if dates is not None else np.arange(len(obs))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(t, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float),
                    color="orange", alpha=0.30, label="IC 95%")
    ax.plot(t, obs, color="blue", linewidth=1.1, label="Qobs")
    ax.plot(t, np.asarray(mean, dtype=float), color="red", linewidth=1.2, label="Qmean")
    ax.set_ylabel("Caudal (m3/s)"); ax.set_title(title)
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_by_regime(obs, sim, p25, p75, title, output_path):
    obs = np.asarray(obs, dtype=float); sim = np.asarray(sim, dtype=float)
    res = sim - obs
    groups = {
        "bajos": res[obs <= p25],
        "medios": res[(obs > p25) & (obs <= p75)],
        "altos": res[obs > p75],
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [g[np.isfinite(g)] for g in groups.values()]
    labels = list(groups.keys())
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        # Matplotlib < 3.9 uses ``labels``; Matplotlib >= 3.11 removes it.
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Residuo (Qsim - Qobs)")
    ax.set_title(title); ax.grid(alpha=0.25)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
