import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from metrics import METRIC_REGISTRY


def add_distance_to_ideal(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    df = df.copy()
    dist2 = np.zeros(len(df), dtype=float)

    for m in metrics:
        metric_name = m.lower()

        if metric_name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{metric_name}'. Available: {list(METRIC_REGISTRY.keys())}"
            )

        sense = METRIC_REGISTRY[metric_name]["sense"]

        if sense == "max":
            target = 1.0
            dist2 += (target - df[metric_name].to_numpy(dtype=float)) ** 2
        elif sense == "min":
            target = 0.0
            dist2 += (df[metric_name].to_numpy(dtype=float) - target) ** 2
        else:
            raise ValueError(f"Invalid sense for metric '{metric_name}'")

    df["distance_to_ideal"] = np.sqrt(dist2)
    return df


def plot_pareto_front(all_df: pd.DataFrame, pareto_df: pd.DataFrame, metrics: List[str]) -> None:
    if len(metrics) != 2:
        print("Pareto plot only implemented for exactly 2 metrics.")
        return

    m1, m2 = [m.lower() for m in metrics]

    plt.figure(figsize=(7, 5))

    plt.scatter(
        all_df[m1],
        all_df[m2],
        alpha=0.7,
        label="Weighted solutions"
    )

    plt.scatter(
        pareto_df[m1],
        pareto_df[m2],
        s=80,
        label="Pareto front"
    )

    plt.xlabel(m1.upper())
    plt.ylabel(m2.upper())
    plt.title(f"Pareto front: {m1.upper()} vs {m2.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.show()