"""Hidrogramas con y sin banda de incertidumbre.

Migra ``plot_hydrograph_with_ci`` del codigo original y anade una version
simple para predicciones puntuales (E1/E3). Usa backend no interactivo (Agg)
para que funcione en ejecucion batch sin display.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_hydrograph(
    discharge: np.ndarray,
    qsim: np.ndarray,
    title: str,
    output_path: str | Path,
    precip: Optional[np.ndarray] = None,
    qinf: Optional[np.ndarray] = None,
    qsup: Optional[np.ndarray] = None,
    dates=None,
) -> None:
    """Hidrograma Qobs vs Qsim, con precipitacion y banda IC opcionales."""
    discharge = np.asarray(discharge, dtype=float)
    qsim = np.asarray(qsim, dtype=float)

    if dates is not None:
        t = pd.to_datetime(np.asarray(dates))
        usar_fechas = True
        bar_width = 1.0
    else:
        t = np.arange(len(discharge))
        usar_fechas = False
        bar_width = 0.8

    fig, ax1 = plt.subplots(figsize=(13, 6))

    if precip is not None:
        ax1.bar(t, np.asarray(precip, dtype=float), width=bar_width,
                color="black", alpha=0.45, label="Precipitacion")
        ax1.set_ylabel("Precipitacion (mm)")
        ax1.invert_yaxis()
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(t, discharge, color="blue", linewidth=1.2, label="Qobs")
    if qinf is not None and qsup is not None:
        ax2.fill_between(t, np.asarray(qinf, dtype=float), np.asarray(qsup, dtype=float),
                         color="gray", alpha=0.30, label="IC 95%")
    ax2.plot(t, qsim, color="red", linewidth=1.3, label="Qsim")
    ax2.set_ylabel("Caudal (m3/s)")

    if usar_fechas:
        ax1.set_xlabel("Fecha")
        locator = mdates.AutoDateLocator()
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate()
    else:
        ax1.set_xlabel("Paso de tiempo")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
