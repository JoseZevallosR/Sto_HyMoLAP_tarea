"""Curva de duracion de caudales (flow duration curve)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _fdc(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    xs = np.sort(x)[::-1]
    exceed = np.arange(1, len(xs) + 1) / (len(xs) + 1) * 100.0
    return exceed, xs


def plot_flow_duration(obs, sim, title, output_path, extra=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    e, v = _fdc(obs); ax.plot(e, v, color="blue", label="Qobs", linewidth=1.5)
    e, v = _fdc(sim); ax.plot(e, v, color="red", label="Qsim", linewidth=1.3)
    if extra:
        for name, series in extra.items():
            e, v = _fdc(series); ax.plot(e, v, label=name, linewidth=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("% de tiempo excedido")
    ax.set_ylabel("Caudal (m3/s, log)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.25, which="both")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
