"""Comparacion entre experimentos.

Lee las salidas por experimento (metrics_validation.csv, metrics_by_regime.csv,
uncertainty_metrics.csv) desde ``outputs/experiments/<id>/`` y construye en
``outputs/comparison/``:

* leaderboard.csv                  (ranking por NSE/KGE de validacion)
* experiment_matrix.csv            (que componentes activa cada experimento)
* validation_metrics_comparison.csv
* regime_metrics_comparison.csv
* uncertainty_comparison.csv
* best_model_summary.md
* figures/metrics_barplot.png
* figures/regime_rmse_comparison.png
* figures/hydrograph_best_models.png
* figures/flow_duration_comparison.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

_log = get_logger("experiments.comparison")

# Mapa estatico de componentes por experimento (para experiment_matrix.csv).
_COMPONENTS = {
    "E1_RAMIS_DET_BF":      dict(fisica=True,  estocastico=False, baseflow=True,  ml="-",        secuencial=False),
    "E2_RAMIS_LEVY_BF":     dict(fisica=True,  estocastico=True,  baseflow=True,  ml="-",        secuencial=False),
    "E3_ML_PURE":           dict(fisica=False, estocastico=False, baseflow=False, ml="xgboost",  secuencial=False),
    "E4_RAMIS_XGB_MEAN":    dict(fisica=True,  estocastico=True,  baseflow=True,  ml="xgboost",  secuencial=False),
    "E5_RAMIS_XGB_QUANTILE":dict(fisica=True,  estocastico=True,  baseflow=True,  ml="xgboost",  secuencial=False),
    "E6_RAMIS_GRU_MEAN":    dict(fisica=True,  estocastico=True,  baseflow=True,  ml="gru",      secuencial=True),
    "E7_RAMIS_GRU_QUANTILE":dict(fisica=True,  estocastico=True,  baseflow=True,  ml="gru",      secuencial=True),
    "A1_RAMIS_LEVY_NOBF":   dict(fisica=True,  estocastico=True,  baseflow=False, ml="-",        secuencial=False),
    "A2_RAMIS_LEVY_BF":     dict(fisica=True,  estocastico=True,  baseflow=True,  ml="-",        secuencial=False),
}


def _read_single_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def collect_results(exp_root: Path, experiment_ids: List[str]) -> pd.DataFrame:
    """Reune las metricas de validacion de cada experimento en una tabla."""
    rows = []
    for eid in experiment_ids:
        d = exp_root / eid
        val = _read_single_row(d / "metrics_validation.csv")
        if val is None:
            _log.warning("Sin metricas de validacion para %s (omitido).", eid)
            continue
        row = {"experiment": eid}
        row.update({k: val.get(k) for k in ["NSE", "KGE", "RMSE", "MAE", "PBIAS", "R2"]})
        unc = _read_single_row(d / "uncertainty_metrics.csv")
        if unc:
            row.update({k: unc.get(k) for k in ["PICP", "PINAW", "MPIW", "Winkler"]})
        rows.append(row)
    return pd.DataFrame(rows)


def build_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """Ordena los experimentos por desempeno (NSE desc, luego KGE)."""
    if results.empty:
        return results
    lb = results.sort_values(["NSE", "KGE"], ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    return lb


def build_experiment_matrix(experiment_ids: List[str]) -> pd.DataFrame:
    rows = []
    for eid in experiment_ids:
        comp = _COMPONENTS.get(eid, {})
        rows.append({"experiment": eid, **comp})
    return pd.DataFrame(rows)


def build_regime_comparison(exp_root: Path, experiment_ids: List[str]) -> pd.DataFrame:
    frames = []
    for eid in experiment_ids:
        p = exp_root / eid / "metrics_by_regime.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "experiment" not in df.columns:
                df.insert(0, "experiment", eid)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_uncertainty_comparison(results: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["experiment", "PICP", "PINAW", "MPIW", "Winkler"] if c in results.columns]
    sub = results[cols].dropna(subset=[c for c in cols if c != "experiment"], how="all")
    return sub.reset_index(drop=True)


# ----------------------------------------------------------------- figuras
def _fig_metrics_barplot(leaderboard: pd.DataFrame, out: Path) -> None:
    if leaderboard.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(leaderboard))
    ax.bar(x - 0.2, leaderboard["NSE"], width=0.4, label="NSE")
    ax.bar(x + 0.2, leaderboard["KGE"], width=0.4, label="KGE")
    ax.set_xticks(x)
    ax.set_xticklabels(leaderboard["experiment"], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Puntaje")
    ax.set_title("Comparacion NSE / KGE (validacion)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_regime_rmse(regime: pd.DataFrame, out: Path) -> None:
    if regime.empty or "regime" not in regime.columns:
        return
    piv = regime.pivot_table(index="experiment", columns="regime", values="RMSE")
    order = [c for c in ["low", "mid", "high"] if c in piv.columns]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE por regimen de caudal (validacion)")
    ax.set_xticklabels(piv.index, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_hydrograph_best(exp_root: Path, best_ids: List[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    for eid in best_ids:
        p = exp_root / eid / "predictions_validation.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not plotted:
            ax.plot(df["Qobs"].to_numpy(), color="black", lw=1.4, label="Qobs")
            plotted = True
        ax.plot(df["Qsim"].to_numpy(), lw=1.0, alpha=0.85, label=eid)
    if plotted:
        ax.set_xlabel("Paso temporal (validacion)")
        ax.set_ylabel("Caudal")
        ax.set_title("Hidrogramas: mejores modelos vs observado")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
    plt.close(fig)


def _fig_fdc_comparison(exp_root: Path, best_ids: List[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = False
    for eid in best_ids:
        p = exp_root / eid / "predictions_validation.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not plotted:
            obs = np.sort(df["Qobs"].to_numpy())[::-1]
            ex = np.arange(1, len(obs) + 1) / (len(obs) + 1) * 100
            ax.plot(ex, obs, color="black", lw=1.5, label="Qobs")
            plotted = True
        sim = np.sort(df["Qsim"].to_numpy())[::-1]
        ex = np.arange(1, len(sim) + 1) / (len(sim) + 1) * 100
        ax.plot(ex, sim, lw=1.0, alpha=0.85, label=eid)
    if plotted:
        ax.set_yscale("log")
        ax.set_xlabel("Probabilidad de excedencia (%)")
        ax.set_ylabel("Caudal (log)")
        ax.set_title("Curvas de duracion de caudales (validacion)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
    plt.close(fig)


def _write_summary_md(leaderboard: pd.DataFrame, matrix: pd.DataFrame, out: Path) -> None:
    lines = ["# Resumen de comparacion de modelos\n"]
    if leaderboard.empty:
        lines.append("No se encontraron resultados de validacion.\n")
        out.write_text("\n".join(lines), encoding="utf-8")
        return
    best = leaderboard.iloc[0]
    lines.append(f"**Mejor modelo:** `{best['experiment']}`  ")
    lines.append(
        f"NSE={best['NSE']:.3f}, KGE={best['KGE']:.3f}, "
        f"RMSE={best['RMSE']:.3f}, PBIAS={best['PBIAS']:.2f}%\n"
    )
    lines.append("## Leaderboard (validacion)\n")
    lines.append(leaderboard.to_markdown(index=False))
    lines.append("\n## Matriz de componentes\n")
    lines.append(matrix.to_markdown(index=False))

    # Lectura de ablaciones baseflow (A1 vs A2) si existen.
    ab = leaderboard.set_index("experiment")
    if "A1_RAMIS_LEVY_NOBF" in ab.index and "A2_RAMIS_LEVY_BF" in ab.index:
        d_nse = ab.loc["A2_RAMIS_LEVY_BF", "NSE"] - ab.loc["A1_RAMIS_LEVY_NOBF", "NSE"]
        verdict = "mejora" if d_nse > 0 else "no mejora"
        lines.append(
            f"\n## Ablacion del flujo base (A2 con BF vs A1 sin BF)\n"
            f"Delta NSE = {d_nse:+.3f} -> el reservorio de flujo base **{verdict}** "
            f"el desempeno en validacion.\n"
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    exp_root: str | Path,
    comparison_root: str | Path,
    experiment_ids: List[str],
    n_best_for_plots: int = 4,
) -> pd.DataFrame:
    """Construye todas las tablas y figuras de comparacion. Devuelve el leaderboard."""
    exp_root = Path(exp_root)
    comparison_root = Path(comparison_root)
    fig_dir = comparison_root / "figures"
    comparison_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(exp_root, experiment_ids)
    leaderboard = build_leaderboard(results)
    matrix = build_experiment_matrix(experiment_ids)
    regime = build_regime_comparison(exp_root, experiment_ids)
    uncertainty = build_uncertainty_comparison(results)

    leaderboard.to_csv(comparison_root / "leaderboard.csv", index=False)
    matrix.to_csv(comparison_root / "experiment_matrix.csv", index=False)
    results.to_csv(comparison_root / "validation_metrics_comparison.csv", index=False)
    regime.to_csv(comparison_root / "regime_metrics_comparison.csv", index=False)
    uncertainty.to_csv(comparison_root / "uncertainty_comparison.csv", index=False)

    _fig_metrics_barplot(leaderboard, fig_dir / "metrics_barplot.png")
    _fig_regime_rmse(regime, fig_dir / "regime_rmse_comparison.png")
    best_ids = leaderboard["experiment"].head(n_best_for_plots).tolist() if not leaderboard.empty else []
    _fig_hydrograph_best(exp_root, best_ids, fig_dir / "hydrograph_best_models.png")
    _fig_fdc_comparison(exp_root, best_ids, fig_dir / "flow_duration_comparison.png")
    _write_summary_md(leaderboard, matrix, comparison_root / "best_model_summary.md")

    _log.info("Comparacion escrita en %s", comparison_root)
    return leaderboard
