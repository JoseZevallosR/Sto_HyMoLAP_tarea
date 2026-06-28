"""Comparacion entre experimentos.

Lee las salidas por experimento desde ``outputs/experiments/<id>/`` y construye
en ``outputs/comparison/``:

* leaderboard.csv
* experiment_matrix.csv
* validation_metrics_comparison.csv
* regime_metrics_comparison.csv
* uncertainty_comparison.csv
* ablation_effects.csv
* evaluation_window_comparison.csv
* best_model_summary.md
* figures/*.png
"""
from __future__ import annotations

from pathlib import Path
import json
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
    "E0_RAMIS_DET_NOBF": dict(role="minimal", family="physical", fisica=True,  estocastico=False, baseflow=False, ml="-",       secuencial=False, features="RAMIS deterministico"),
    "E1_RAMIS_DET_BF":   dict(role="minimal", family="physical", fisica=True,  estocastico=False, baseflow=True,  ml="-",       secuencial=False, features="RAMIS deterministico + Qbase"),
    "E2_RAMIS_LEVY_NOBF":dict(role="minimal", family="physical", fisica=True,  estocastico=True,  baseflow=False, ml="-",       secuencial=False, features="RAMIS Levy ensemble"),
    "E3_RAMIS_LEVY_BF":  dict(role="minimal", family="physical", fisica=True,  estocastico=True,  baseflow=True,  ml="-",       secuencial=False, features="RAMIS Levy ensemble + Qbase"),
    "E4_ML_PURE_XGB":    dict(role="minimal", family="ml",       fisica=False, estocastico=False, baseflow=False, ml="xgboost", secuencial=False, features="P, PET, Tmin, Tmax + lags"),
    "E5_HYB_XGB_MEAN":   dict(role="minimal", family="hybrid",   fisica=True,  estocastico=True,  baseflow=True,  ml="xgboost", secuencial=False, features="Qmean + lags"),
    "E6_HYB_XGB_QUANTILES": dict(role="minimal", family="hybrid", fisica=True,  estocastico=True,  baseflow=True,  ml="xgboost", secuencial=False, features="Qmean + quantiles + widths + lags"),
    "E7_HYB_GRU_MEAN":   dict(role="extended", family="hybrid",  fisica=True,  estocastico=True,  baseflow=True,  ml="gru",     secuencial=True,  features="sequence(Qmean)"),
    "E8_HYB_GRU_QUANTILES": dict(role="extended", family="hybrid", fisica=True, estocastico=True, baseflow=True,  ml="gru",     secuencial=True,  features="sequence(Qmean + quantiles + Qbase)"),
}

_ABLATION_PAIRS = [
    ("deterministic_baseflow", "E0_RAMIS_DET_NOBF", "E1_RAMIS_DET_BF", "Efecto del reservorio baseflow en RAMIS deterministico"),
    ("stochastic_baseflow", "E2_RAMIS_LEVY_NOBF", "E3_RAMIS_LEVY_BF", "Efecto del reservorio baseflow en RAMIS-Levy"),
    ("levy_given_baseflow", "E1_RAMIS_DET_BF", "E3_RAMIS_LEVY_BF", "Efecto del ruido Levy manteniendo baseflow"),
    ("pure_ml_vs_physical", "E3_RAMIS_LEVY_BF", "E4_ML_PURE_XGB", "ML puro frente al fisico-estocastico completo"),
    ("hybrid_mean_vs_physical", "E3_RAMIS_LEVY_BF", "E5_HYB_XGB_MEAN", "Hibrido XGB-media frente al fisico-estocastico completo"),
    ("hybrid_quantiles_vs_physical", "E3_RAMIS_LEVY_BF", "E6_HYB_XGB_QUANTILES", "Hibrido XGB-cuantiles frente al fisico-estocastico completo"),
    ("quantiles_vs_mean_xgb", "E5_HYB_XGB_MEAN", "E6_HYB_XGB_QUANTILES", "Aporte de cuantiles frente a solo Qmean en XGB"),
    ("gru_mean_vs_xgb_mean", "E5_HYB_XGB_MEAN", "E7_HYB_GRU_MEAN", "Extension GRU-media frente a XGB-media"),
    ("gru_quantiles_vs_xgb_quantiles", "E6_HYB_XGB_QUANTILES", "E8_HYB_GRU_QUANTILES", "Extension GRU-cuantiles frente a XGB-cuantiles"),
]


def _read_single_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _prediction_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path, usecols=["Qobs", "Qsim"])))
    except Exception:  # noqa: BLE001
        return None


def collect_results(exp_root: Path, experiment_ids: List[str]) -> pd.DataFrame:
    """Reune metricas de validacion y n_eval por experimento."""
    rows = []
    for eid in experiment_ids:
        d = exp_root / eid
        val = _read_single_row(d / "metrics_validation.csv")
        if val is None:
            _log.warning("Sin metricas de validacion para %s (omitido).", eid)
            continue
        row = {"experiment": eid, "n_eval": _prediction_count(d / "predictions_validation.csv")}
        row.update({k: val.get(k) for k in ["NSE", "KGE", "RMSE", "MAE", "PBIAS", "R2"]})
        unc = _read_single_row(d / "uncertainty_metrics.csv")
        if unc:
            row.update({k: unc.get(k) for k in ["PICP", "PINAW", "MPIW", "Winkler"]})
        rows.append(row)
    return pd.DataFrame(rows)


def build_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """Ordena experimentos por desempeno de validacion (NSE desc, luego KGE)."""
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



def build_evaluation_window_comparison(exp_root: Path, experiment_ids: List[str]) -> pd.DataFrame:
    """Reune metadatos de ventana comun por experimento."""
    rows = []
    for eid in experiment_ids:
        p = exp_root / eid / "evaluation_window.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        val = data.get("validation", {}) or {}
        tr = data.get("train", {}) or {}
        rows.append({
            "experiment": eid,
            "enabled": data.get("enabled"),
            "mode": data.get("mode"),
            "start_offset": data.get("start_offset"),
            "end_trim": data.get("end_trim"),
            "validation_start_date": val.get("start_date"),
            "validation_end_date": val.get("end_date"),
            "n_validation_before_window": val.get("n_before"),
            "n_validation_after_window": val.get("n_after"),
            "train_start_date": tr.get("start_date"),
            "train_end_date": tr.get("end_date"),
            "n_train_before_window": tr.get("n_before"),
            "n_train_after_window": tr.get("n_after"),
        })
    return pd.DataFrame(rows)

def build_uncertainty_comparison(results: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["experiment", "PICP", "PINAW", "MPIW", "Winkler"] if c in results.columns]
    if not cols:
        return pd.DataFrame()
    sub = results[cols].dropna(subset=[c for c in cols if c != "experiment"], how="all")
    return sub.reset_index(drop=True)


def build_ablation_effects(results: pd.DataFrame) -> pd.DataFrame:
    """Calcula deltas candidato - baseline para pares de ablacion predefinidos."""
    if results.empty:
        return pd.DataFrame()
    idx = results.set_index("experiment")
    rows = []
    for ablation, baseline, candidate, question in _ABLATION_PAIRS:
        if baseline not in idx.index or candidate not in idx.index:
            continue
        b = idx.loc[baseline]
        c = idx.loc[candidate]
        rows.append({
            "ablation": ablation,
            "baseline": baseline,
            "candidate": candidate,
            "question": question,
            "delta_NSE": c.get("NSE", np.nan) - b.get("NSE", np.nan),
            "delta_KGE": c.get("KGE", np.nan) - b.get("KGE", np.nan),
            "delta_RMSE": c.get("RMSE", np.nan) - b.get("RMSE", np.nan),
            "delta_MAE": c.get("MAE", np.nan) - b.get("MAE", np.nan),
            "delta_abs_PBIAS_improvement": abs(b.get("PBIAS", np.nan)) - abs(c.get("PBIAS", np.nan)),
            "n_eval_baseline": b.get("n_eval", np.nan),
            "n_eval_candidate": c.get("n_eval", np.nan),
        })
    return pd.DataFrame(rows)


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


def _write_summary_md(
    leaderboard: pd.DataFrame,
    matrix: pd.DataFrame,
    ablations: pd.DataFrame,
    out: Path,
) -> None:
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

    if not ablations.empty:
        lines.append("\n## Efectos de ablacion\n")
        lines.append(ablations.to_markdown(index=False))
        bf = ablations[ablations["ablation"] == "stochastic_baseflow"]
        if not bf.empty:
            row = bf.iloc[0]
            verdict = "mejora" if row["delta_NSE"] > 0 else "no mejora"
            lines.append(
                f"\n**Lectura baseflow estocastico:** E3 - E2 da "
                f"Delta NSE={row['delta_NSE']:+.3f}; el reservorio de flujo base "
                f"**{verdict}** el desempeno global de validacion.\n"
            )
    out.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(
    exp_root: str | Path,
    comparison_root: str | Path,
    experiment_ids: List[str],
    n_best_for_plots: int = 4,
) -> pd.DataFrame:
    """Construye tablas y figuras de comparacion. Devuelve el leaderboard."""
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
    ablations = build_ablation_effects(results)
    eval_windows = build_evaluation_window_comparison(exp_root, experiment_ids)

    leaderboard.to_csv(comparison_root / "leaderboard.csv", index=False)
    matrix.to_csv(comparison_root / "experiment_matrix.csv", index=False)
    results.to_csv(comparison_root / "validation_metrics_comparison.csv", index=False)
    regime.to_csv(comparison_root / "regime_metrics_comparison.csv", index=False)
    uncertainty.to_csv(comparison_root / "uncertainty_comparison.csv", index=False)
    ablations.to_csv(comparison_root / "ablation_effects.csv", index=False)
    eval_windows.to_csv(comparison_root / "evaluation_window_comparison.csv", index=False)

    _fig_metrics_barplot(leaderboard, fig_dir / "metrics_barplot.png")
    _fig_regime_rmse(regime, fig_dir / "regime_rmse_comparison.png")
    best_ids = leaderboard["experiment"].head(n_best_for_plots).tolist() if not leaderboard.empty else []
    _fig_hydrograph_best(exp_root, best_ids, fig_dir / "hydrograph_best_models.png")
    _fig_fdc_comparison(exp_root, best_ids, fig_dir / "flow_duration_comparison.png")
    _write_summary_md(leaderboard, matrix, ablations, comparison_root / "best_model_summary.md")

    _log.info("Comparacion escrita en %s", comparison_root)
    return leaderboard
