#!/usr/bin/env python3
"""Generate paper-ready Phase 3.4B validation figures.

The script is intentionally lightweight: it only depends on pandas, numpy and
matplotlib. It reads the already-produced ``outputs/experiments`` and
``outputs/comparison`` artefacts, aligns all selected experiments on the common
validation intersection, and writes a machine-readable figure manifest.

Design principles for Phase 3.4B:
- never assume E7/E8 are real GRU models; use ``ml_backend.json`` when present;
- degrade gracefully when optional columns are absent;
- keep all warnings in the manifest instead of failing the whole generation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_SELECTED_EXPERIMENTS = [
    "E0_RAMIS_DET_NOBF",
    "E1_RAMIS_DET_BF",
    "E3_RAMIS_LEVY_BF",
    "E4_ML_PURE_XGB",
    "E6_HYB_XGB_QUANTILES",
    "E8_HYB_GRU_QUANTILES",
]

DEFAULT_STOCHASTIC_REFERENCES = [
    "E2_RAMIS_LEVY_NOBF",
    "E3_RAMIS_LEVY_BF",
]

QSIM_LOWER_CANDIDATES = [
    "Qsim_lower",
    "Qsim_p05",
    "Qsim_q05",
    "Qsim_q025",
    "Qinf",
    "qinf",
    "lower",
    "q_lower",
]
QSIM_UPPER_CANDIDATES = [
    "Qsim_upper",
    "Qsim_p95",
    "Qsim_q95",
    "Qsim_q975",
    "Qsup",
    "qsup",
    "upper",
    "q_upper",
]
PHYSICAL_COMPONENT_SETS = [
    ("Qfast", ["Qfast", "qfast", "Q_fast", "q_fast"]),
    ("Qbase", ["Qbase", "qbase", "Q_base", "q_base"]),
    ("Qtotal", ["Qtotal", "qtotal", "Q_total", "q_total"]),
]


@dataclass
class ManifestEntry:
    figure_id: str
    status: str
    path: str = ""
    experiments: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: str = ""


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _write_manifest(entries: list[ManifestEntry], figures_dir: Path) -> tuple[Path, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        row = asdict(entry)
        # CSV-friendly representation; JSON keeps native arrays below.
        row["experiments"] = ";".join(row["experiments"])
        row["required_columns"] = ";".join(row["required_columns"])
        row["missing_columns"] = ";".join(row["missing_columns"])
        row["warnings"] = " | ".join(row["warnings"])
        rows.append(row)
    csv_path = figures_dir / "phase34_figure_manifest.csv"
    json_path = figures_dir / "phase34_figure_manifest.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return csv_path, json_path


def _load_predictions(exp_root: Path, experiment: str) -> tuple[pd.DataFrame | None, str | None]:
    path = exp_root / experiment / "predictions_validation.csv"
    if not path.exists():
        return None, f"missing file: {path}"
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive path
        return None, f"could not read {path}: {exc}"
    missing = [col for col in ["date", "Qobs", "Qsim"] if col not in df.columns]
    if missing:
        return None, f"missing required columns in {path}: {missing}"
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, None


def _load_backend_note(exp_root: Path, experiment: str) -> str:
    path = exp_root / experiment / "ml_backend.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "backend metadata could not be decoded"
    ml_model = data.get("ml_model", "")
    backend_label = data.get("backend_label") or data.get("backend", "")
    is_real_gru = bool(data.get("is_real_gru", False))
    if ml_model == "gru" and not is_real_gru:
        return f"declared GRU uses backend={backend_label}; not a real GRU in this environment"
    if backend_label:
        return f"backend={backend_label}"
    return ""


def _common_window_from_comparison(comparison_dir: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None, list[str]]:
    warnings: list[str] = []
    path = comparison_dir / "leaderboard_common_intersection.csv"
    if not path.exists():
        warnings.append("leaderboard_common_intersection.csv not found; using date intersection from available predictions")
        return None, None, warnings
    df = pd.read_csv(path)
    required = {"common_start_date", "common_end_date"}
    if not required.issubset(df.columns):
        warnings.append("common_start_date/common_end_date absent in leaderboard; using prediction date intersection")
        return None, None, warnings
    starts = pd.to_datetime(df["common_start_date"].dropna().unique())
    ends = pd.to_datetime(df["common_end_date"].dropna().unique())
    if len(starts) == 0 or len(ends) == 0:
        warnings.append("empty common window in leaderboard; using prediction date intersection")
        return None, None, warnings
    if len(starts) > 1 or len(ends) > 1:
        warnings.append("multiple common windows found; using max(start) and min(end)")
    return max(starts), min(ends), warnings


def _align_common_predictions(
    prediction_map: dict[str, pd.DataFrame],
    comparison_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    start, end, warnings = _common_window_from_comparison(comparison_dir)
    if not prediction_map:
        return pd.DataFrame(), warnings + ["no prediction files were available"]

    prepared: list[pd.DataFrame] = []
    for exp, df in prediction_map.items():
        keep = df.copy()
        if start is not None:
            keep = keep[keep["date"] >= start]
        if end is not None:
            keep = keep[keep["date"] <= end]
        keep = keep[["date", "Qobs", "Qsim"]].rename(columns={"Qsim": f"Qsim__{exp}", "Qobs": f"Qobs__{exp}"})
        keep = keep[np.isfinite(keep[f"Qsim__{exp}"])]
        prepared.append(keep)

    aligned = prepared[0]
    for other in prepared[1:]:
        aligned = aligned.merge(other, on="date", how="inner")

    qobs_cols = [col for col in aligned.columns if col.startswith("Qobs__")]
    if qobs_cols:
        # Keep the exact common intersection used for metrics: all selected models must
        # have finite Qobs on the same dates. The first Qobs column is then used as reference.
        finite_qobs = np.ones(len(aligned), dtype=bool)
        for col in qobs_cols:
            finite_qobs &= np.isfinite(aligned[col].to_numpy(dtype=float))
        dropped = int((~finite_qobs).sum())
        if dropped:
            warnings.append(f"dropped {dropped} rows with missing Qobs within common window")
        aligned = aligned.loc[finite_qobs].copy()
        aligned["Qobs"] = aligned[qobs_cols[0]]
        for col in qobs_cols[1:]:
            diff = np.nanmax(np.abs(aligned[col].to_numpy(dtype=float) - aligned["Qobs"].to_numpy(dtype=float))) if len(aligned) else 0.0
            if np.isfinite(diff) and diff > 1e-9:
                warnings.append(f"Qobs differs across experiments; using {qobs_cols[0]} as reference")
                break
    aligned = aligned.sort_values("date").reset_index(drop=True)
    return aligned, warnings


def _fdc(series: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    sorted_values = np.sort(values)[::-1]
    exceedance = np.arange(1, sorted_values.size + 1) / (sorted_values.size + 1) * 100.0
    return exceedance, sorted_values


def _get_thresholds(exp_root: Path, experiment: str, qobs: pd.Series) -> tuple[float, float, list[str]]:
    path = exp_root / experiment / "regime_thresholds.json"
    warnings: list[str] = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            p25 = data.get("p25") or data.get("q25") or data.get("low_threshold")
            p75 = data.get("p75") or data.get("q75") or data.get("high_threshold")
            if p25 is not None and p75 is not None:
                return float(p25), float(p75), warnings
            warnings.append(f"{experiment}: regime_thresholds.json exists but p25/p75 were not found")
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"{experiment}: could not decode regime thresholds: {exc}")
    finite = pd.to_numeric(qobs, errors="coerce").dropna()
    if finite.empty:
        warnings.append(f"{experiment}: no finite Qobs for regime fallback; using p25=0, p75=0")
        return 0.0, 0.0, warnings
    warnings.append(f"{experiment}: using validation Qobs quantiles as fallback regime thresholds")
    return float(finite.quantile(0.25)), float(finite.quantile(0.75)), warnings


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def plot_hydrograph_common(aligned: pd.DataFrame, experiments: list[str], out: Path, warnings: list[str], backend_notes: dict[str, str]) -> ManifestEntry:
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.plot(aligned["date"], aligned["Qobs"], linewidth=1.25, label="Qobs")
    for exp in experiments:
        col = f"Qsim__{exp}"
        if col in aligned.columns:
            ax.plot(aligned["date"], aligned[col], linewidth=0.95, alpha=0.88, label=exp)
    ax.set_title("Fase 3.4B - Hidrograma de validación en ventana común")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Caudal (m³/s)")
    ax.grid(alpha=0.25)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(ncols=2, fontsize=8)
    notes = [note for note in backend_notes.values() if note]
    if notes:
        ax.text(0.01, -0.23, "Notas: " + " | ".join(notes), transform=ax.transAxes, fontsize=8, va="top")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ManifestEntry(
        figure_id="hydrograph_validation_common",
        status="generated",
        path=str(out),
        experiments=experiments,
        required_columns=["date", "Qobs", "Qsim"],
        warnings=warnings + notes,
        notes=f"n_common={len(aligned)}",
    )


def plot_fdc_common(aligned: pd.DataFrame, experiments: list[str], out: Path) -> ManifestEntry:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    exceed, values = _fdc(aligned["Qobs"])
    ax.plot(exceed, values, linewidth=1.35, label="Qobs")
    for exp in experiments:
        col = f"Qsim__{exp}"
        if col in aligned.columns:
            exceed, values = _fdc(aligned[col])
            ax.plot(exceed, values, linewidth=1.0, alpha=0.88, label=exp)
    ax.set_yscale("log")
    ax.set_xlabel("Tiempo excedido (%)")
    ax.set_ylabel("Caudal (m³/s, escala log)")
    ax.set_title("Fase 3.4B - Curva de duración de caudales")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ManifestEntry(
        figure_id="fdc_common",
        status="generated",
        path=str(out),
        experiments=experiments,
        required_columns=["date", "Qobs", "Qsim"],
        notes=f"n_common={len(aligned)}",
    )


def plot_scatter_per_model(aligned: pd.DataFrame, experiment: str, out: Path) -> ManifestEntry:
    obs = aligned["Qobs"].to_numpy(dtype=float)
    sim = aligned[f"Qsim__{experiment}"].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(obs, sim, s=8, alpha=0.35)
    upper = float(np.nanmax([np.nanmax(obs) if obs.size else 0.0, np.nanmax(sim) if sim.size else 0.0]))
    upper = upper * 1.05 if upper > 0 else 1.0
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1.0, label="1:1")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Qobs (m³/s)")
    ax.set_ylabel("Qsim (m³/s)")
    ax.set_title(f"Qobs-Qsim | {experiment}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ManifestEntry(
        figure_id=f"scatter_{experiment}",
        status="generated",
        path=str(out),
        experiments=[experiment],
        required_columns=["date", "Qobs", "Qsim"],
        notes=f"n={int(mask.sum())}",
    )


def plot_residuals_by_regime(aligned: pd.DataFrame, experiment: str, out: Path, p25: float, p75: float, warnings: list[str]) -> ManifestEntry:
    obs = aligned["Qobs"].to_numpy(dtype=float)
    sim = aligned[f"Qsim__{experiment}"].to_numpy(dtype=float)
    residual = sim - obs
    groups = {
        "low": residual[obs <= p25],
        "mid": residual[(obs > p25) & (obs <= p75)],
        "high": residual[obs > p75],
    }
    data = [values[np.isfinite(values)] for values in groups.values()]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    try:
        ax.boxplot(data, tick_labels=list(groups), showfliers=False)
    except TypeError:  # Matplotlib < 3.9
        ax.boxplot(data, labels=list(groups), showfliers=False)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Residuo Qsim - Qobs (m³/s)")
    ax.set_title(f"Residuos por régimen | {experiment}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    counts = {name: int(len(values)) for name, values in zip(groups, data)}
    return ManifestEntry(
        figure_id=f"residuals_by_regime_{experiment}",
        status="generated",
        path=str(out),
        experiments=[experiment],
        required_columns=["date", "Qobs", "Qsim", "regime_thresholds.json:p25", "regime_thresholds.json:p75"],
        warnings=warnings,
        notes=f"p25={p25:.6g}; p75={p75:.6g}; counts={counts}",
    )


def plot_physical_components(prediction_map: dict[str, pd.DataFrame], experiments: list[str], figures_dir: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for exp in experiments:
        df = prediction_map.get(exp)
        if df is None:
            entries.append(ManifestEntry(
                figure_id=f"physical_components_{exp}",
                status="skipped",
                experiments=[exp],
                required_columns=[name for name, _ in PHYSICAL_COMPONENT_SETS],
                missing_columns=[name for name, _ in PHYSICAL_COMPONENT_SETS],
                warnings=["prediction file not available"],
            ))
            continue
        found: dict[str, str] = {}
        missing: list[str] = []
        for canonical, candidates in PHYSICAL_COMPONENT_SETS:
            col = _find_first_column(df, candidates)
            if col is None:
                missing.append(canonical)
            else:
                found[canonical] = col
        if missing:
            entries.append(ManifestEntry(
                figure_id=f"physical_components_{exp}",
                status="skipped",
                experiments=[exp],
                required_columns=[name for name, _ in PHYSICAL_COMPONENT_SETS],
                missing_columns=missing,
                warnings=["physical component columns are absent in predictions_validation.csv"],
            ))
            continue
        out = figures_dir / f"physical_components_{exp}.png"
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        for label, col in found.items():
            ax.plot(df["date"], df[col], linewidth=1.0, label=label)
        ax.set_title(f"Componentes físicos | {exp}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Caudal (m³/s)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        entries.append(ManifestEntry(
            figure_id=f"physical_components_{exp}",
            status="generated",
            path=str(out),
            experiments=[exp],
            required_columns=[name for name, _ in PHYSICAL_COMPONENT_SETS],
            notes="; ".join(f"{key}={value}" for key, value in found.items()),
        ))
    return entries


def _uncertainty_warning(comparison_dir: Path, experiment: str) -> str:
    path = comparison_dir / "uncertainty_comparison.csv"
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path)
    except Exception:
        return ""
    if "experiment" not in df.columns or "PICP" not in df.columns:
        return ""
    row = df[df["experiment"] == experiment]
    if row.empty:
        return ""
    picp = float(row.iloc[0]["PICP"])
    if np.isfinite(picp) and picp < 0.5:
        return f"{experiment}: PICP={picp:.3f}; warning: stochastic band is strongly underdispersive"
    return f"{experiment}: PICP={picp:.3f}"


def plot_stochastic_reference(
    prediction_map: dict[str, pd.DataFrame],
    stochastic_refs: list[str],
    comparison_dir: Path,
    figures_dir: Path,
) -> ManifestEntry:
    available = [exp for exp in stochastic_refs if exp in prediction_map]
    missing_exp = [exp for exp in stochastic_refs if exp not in prediction_map]
    if not available:
        return ManifestEntry(
            figure_id="stochastic_reference_E2_E3",
            status="skipped",
            experiments=stochastic_refs,
            required_columns=["date", "Qobs", "Qsim", "Qsim_lower", "Qsim_upper"],
            missing_columns=["predictions_validation.csv"],
            warnings=[f"missing experiments: {missing_exp}"],
        )

    # Try full band figure first.
    band_specs = []
    missing_band_cols: list[str] = []
    for exp in available:
        df = prediction_map[exp]
        lower = _find_first_column(df, QSIM_LOWER_CANDIDATES)
        upper = _find_first_column(df, QSIM_UPPER_CANDIDATES)
        if lower and upper:
            band_specs.append((exp, lower, upper))
        else:
            missing_band_cols.append(f"{exp}:lower/upper")

    warnings = [w for exp in available if (w := _uncertainty_warning(comparison_dir, exp))]
    if missing_exp:
        warnings.append(f"missing stochastic reference experiments: {missing_exp}")

    out = figures_dir / "stochastic_reference_E2_E3.png"
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    first_df = prediction_map[available[0]]
    ax.plot(first_df["date"], first_df["Qobs"], linewidth=1.15, label="Qobs")

    if band_specs and len(band_specs) == len(available):
        for exp, lower_col, upper_col in band_specs:
            df = prediction_map[exp]
            ax.fill_between(df["date"], df[lower_col], df[upper_col], alpha=0.22, label=f"{exp} band")
            ax.plot(df["date"], df["Qsim"], linewidth=1.0, label=f"{exp} Qsim")
        status = "generated"
        missing_columns: list[str] = []
        notes = "stochastic bands generated from lower/upper columns"
    else:
        for exp in available:
            df = prediction_map[exp]
            ax.plot(df["date"], df["Qsim"], linewidth=1.0, label=f"{exp} Qsim")
        status = "fallback"
        missing_columns = missing_band_cols
        warnings.append("lower/upper stochastic band columns are absent; generated point-reference plot only")
        notes = "fallback: no uncertainty band columns in predictions_validation.csv"

    ax.set_title("Referencia estocástica E2/E3")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Caudal (m³/s)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    if warnings:
        ax.text(0.01, -0.22, " | ".join(warnings), transform=ax.transAxes, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ManifestEntry(
        figure_id="stochastic_reference_E2_E3",
        status=status,
        path=str(out),
        experiments=available,
        required_columns=["date", "Qobs", "Qsim", "lower stochastic band", "upper stochastic band"],
        missing_columns=missing_columns,
        warnings=warnings,
        notes=notes,
    )


def generate_phase34_figures(
    output_root: Path = Path("outputs"),
    figures_dir: Path | None = None,
    selected_experiments: list[str] | None = None,
    stochastic_references: list[str] | None = None,
) -> dict[str, Path | list[ManifestEntry]]:
    selected = selected_experiments or DEFAULT_SELECTED_EXPERIMENTS
    stochastic_refs = stochastic_references or DEFAULT_STOCHASTIC_REFERENCES
    output_root = Path(output_root)
    exp_root = output_root / "experiments"
    comparison_dir = output_root / "comparison"
    figures_dir = Path(figures_dir) if figures_dir is not None else output_root / "figures" / "phase34"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[ManifestEntry] = []
    prediction_map: dict[str, pd.DataFrame] = {}
    load_warnings: list[str] = []

    for exp in sorted(set(selected + stochastic_refs)):
        df, warning = _load_predictions(exp_root, exp)
        if df is not None:
            prediction_map[exp] = df
        elif warning:
            load_warnings.append(f"{exp}: {warning}")

    selected_available = [exp for exp in selected if exp in prediction_map]
    missing_selected = [exp for exp in selected if exp not in prediction_map]
    if missing_selected:
        manifest.append(ManifestEntry(
            figure_id="selected_prediction_inputs",
            status="warning",
            experiments=selected,
            required_columns=["date", "Qobs", "Qsim"],
            missing_columns=["predictions_validation.csv"],
            warnings=load_warnings,
            notes=f"missing selected experiments: {missing_selected}",
        ))

    aligned, common_warnings = _align_common_predictions({exp: prediction_map[exp] for exp in selected_available}, comparison_dir)
    if aligned.empty or not selected_available:
        manifest.append(ManifestEntry(
            figure_id="common_alignment",
            status="skipped",
            experiments=selected_available,
            required_columns=["date", "Qobs", "Qsim"],
            warnings=common_warnings + load_warnings,
            notes="no common finite validation intersection available",
        ))
    else:
        backend_notes = {exp: _load_backend_note(exp_root, exp) for exp in selected_available}
        manifest.append(plot_hydrograph_common(
            aligned,
            selected_available,
            figures_dir / "hydrograph_validation_common.png",
            common_warnings,
            backend_notes,
        ))
        manifest.append(plot_fdc_common(
            aligned,
            selected_available,
            figures_dir / "fdc_common.png",
        ))
        for exp in selected_available:
            manifest.append(plot_scatter_per_model(
                aligned,
                exp,
                figures_dir / f"scatter_{exp}.png",
            ))
            p25, p75, threshold_warnings = _get_thresholds(exp_root, exp, aligned["Qobs"])
            manifest.append(plot_residuals_by_regime(
                aligned,
                exp,
                figures_dir / f"residuals_by_regime_{exp}.png",
                p25,
                p75,
                threshold_warnings,
            ))

    manifest.extend(plot_physical_components(prediction_map, selected, figures_dir))
    manifest.append(plot_stochastic_reference(prediction_map, stochastic_refs, comparison_dir, figures_dir))

    csv_path, json_path = _write_manifest(manifest, figures_dir)
    return {"manifest_csv": csv_path, "manifest_json": json_path, "entries": manifest}


def _parse_csv_arg(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 3.4B paper-ready figures.")
    parser.add_argument("--output-root", default="outputs", help="Root containing experiments/ and comparison/.")
    parser.add_argument("--figures-dir", default=None, help="Output directory for Phase 3.4B figures.")
    parser.add_argument(
        "--selected-experiments",
        default=None,
        help="Comma-separated experiment IDs for hydrograph/FDC/scatter/residual figures.",
    )
    parser.add_argument(
        "--stochastic-references",
        default=None,
        help="Comma-separated stochastic reference IDs for E2/E3 band/fallback figure.",
    )
    args = parser.parse_args()
    selected = _parse_csv_arg(args.selected_experiments, DEFAULT_SELECTED_EXPERIMENTS)
    stochastic_refs = _parse_csv_arg(args.stochastic_references, DEFAULT_STOCHASTIC_REFERENCES)
    result = generate_phase34_figures(
        output_root=Path(args.output_root),
        figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        selected_experiments=selected,
        stochastic_references=stochastic_refs,
    )
    print(f"Manifest CSV: {result['manifest_csv']}")
    print(f"Manifest JSON: {result['manifest_json']}")
    entries = result["entries"]
    generated = sum(1 for entry in entries if entry.status == "generated")
    fallback = sum(1 for entry in entries if entry.status == "fallback")
    skipped = sum(1 for entry in entries if entry.status == "skipped")
    print(f"Figures generated: {generated}; fallback: {fallback}; skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
