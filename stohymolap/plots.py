import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_hydrograph_with_ci(
    discharge: np.ndarray,
    precip: np.ndarray,
    qmean: np.ndarray,
    qinf: np.ndarray,
    qsup: np.ndarray,
    title: str,
    output_path: str,
    dates=None,
):
    discharge = np.asarray(discharge, dtype=float)
    precip = np.asarray(precip, dtype=float)
    qmean = np.asarray(qmean, dtype=float)
    qinf = np.asarray(qinf, dtype=float)
    qsup = np.asarray(qsup, dtype=float)

    # Eje X: fechas si se entregan, si no índice entero (paso de tiempo).
    if dates is not None:
        t = pd.to_datetime(np.asarray(dates))
        usar_fechas = True
        bar_width = 1.0          # ancho en días para datos diarios
    else:
        t = np.arange(len(discharge))
        usar_fechas = False
        bar_width = 0.8

    fig, ax1 = plt.subplots(figsize=(13, 6))

    ax1.bar(t, precip, width=bar_width, color="black", alpha=0.50,
            label="Precipitación")
    ax1.set_ylabel("Precipitación")
    ax1.invert_yaxis()
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(t, discharge, color="blue", linewidth=1.2, label="Qobs")
    ax2.fill_between(
        t,
        qinf,
        qsup,
        color="gray",
        alpha=0.30,
        label="IC 95%"
    )
    ax2.plot(t, qmean, color="red", linewidth=1.4, label="Qmean/Qsim")

    ax2.set_ylabel("Caudal")

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
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
