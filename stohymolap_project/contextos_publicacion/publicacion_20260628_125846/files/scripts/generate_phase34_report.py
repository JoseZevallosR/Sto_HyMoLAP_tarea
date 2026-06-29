#!/usr/bin/env python3
"""Build a paper-ready Phase 3.4C/3.5 result package.

This script does not re-run calibration, ML training or figure generation. It
collects the validated artefacts already produced by Phase 3.4A/3.4B and writes
compact, manuscript-oriented tables plus a Markdown report.

The report is deliberately conservative:
- it uses the common validation intersection when available;
- it surfaces audit WARN/FAIL status instead of hiding it;
- it records figure fallback/skipped states from the figure manifest;
- it prevents over-claiming GRU experiments when the actual backend is a
  flattened-sequence fallback MLP.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_SELECTED_EXPERIMENTS = [
    "E0_RAMIS_DET_NOBF",
    "E1_RAMIS_DET_BF",
    "E3_RAMIS_LEVY_BF",
    "E4_ML_PURE_XGB",
    "E6_HYB_XGB_QUANTILES",
    "E8_HYB_GRU_QUANTILES",
]

CORE_METRIC_COLUMNS = ["NSE", "KGE", "RMSE", "MAE", "PBIAS", "R2"]


@dataclass
class PackageArtifact:
    artifact_id: str
    status: str
    path: str = ""
    rows: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: str = ""


def _read_csv(path: Path, *, required: bool = False, warnings: list[str] | None = None) -> pd.DataFrame:
    if warnings is None:
        warnings = []
    if not path.exists():
        message = f"missing input: {path}"
        if required:
            warnings.append(message)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive I/O branch
        warnings.append(f"could not read {path}: {exc}")
        return pd.DataFrame()


def _read_text_status(path: Path, default: str = "MISSING") -> str:
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8").strip()
    return text or default


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _fmt(value: object, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)
    return f"{number:.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: Iterable[str] | None = None, *, max_rows: int = 20) -> str:
    if df.empty:
        return "_No disponible._"
    keep = df.copy()
    if columns is not None:
        keep = keep[[col for col in columns if col in keep.columns]]
    if len(keep) > max_rows:
        keep = keep.head(max_rows)
    for col in keep.columns:
        if pd.api.types.is_float_dtype(keep[col]) or pd.api.types.is_integer_dtype(keep[col]):
            if col in {"rank", "n", "n_eval", "rows"}:
                keep[col] = keep[col].map(lambda x: "" if pd.isna(x) else str(int(x)))
            else:
                keep[col] = keep[col].map(_fmt)
        else:
            keep[col] = keep[col].map(lambda x: "" if pd.isna(x) else str(x))
    try:
        return keep.to_markdown(index=False)
    except Exception:  # pragma: no cover - fallback if tabulate is missing
        return keep.to_string(index=False)


def _write_table(df: pd.DataFrame, path: Path, artifacts: list[PackageArtifact], *, notes: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    artifacts.append(PackageArtifact(path.stem, "generated", str(path), int(len(df)), notes=notes))


def _clean_ranking(leaderboard: pd.DataFrame, backend: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    ranking = leaderboard.copy()
    if "rank" not in ranking.columns:
        ranking = ranking.sort_values(["NSE", "KGE"], ascending=[False, False]).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    if not backend.empty and "experiment" in backend.columns:
        merge_cols = [col for col in ["experiment", "model_type", "ml_model", "backend", "backend_label", "feature_source", "postprocess_method"] if col in backend.columns]
        ranking = ranking.merge(backend[merge_cols], on="experiment", how="left")
    preferred = [
        "rank",
        "experiment",
        "n_eval",
        "common_start_date",
        "common_end_date",
        "NSE",
        "KGE",
        "RMSE",
        "MAE",
        "PBIAS",
        "R2",
        "model_type",
        "ml_model",
        "backend_label",
        "feature_source",
    ]
    return ranking[[col for col in preferred if col in ranking.columns]]


def _selected_metrics(ranking: pd.DataFrame, selected_experiments: list[str]) -> pd.DataFrame:
    if ranking.empty or "experiment" not in ranking.columns:
        return pd.DataFrame()
    order = {exp: i for i, exp in enumerate(selected_experiments)}
    selected = ranking[ranking["experiment"].isin(selected_experiments)].copy()
    selected["selected_order"] = selected["experiment"].map(order)
    selected = selected.sort_values("selected_order").drop(columns=["selected_order"])
    return selected


def _figure_availability(figure_manifest: pd.DataFrame) -> pd.DataFrame:
    if figure_manifest.empty:
        return pd.DataFrame(columns=["status", "count"])
    if "status" not in figure_manifest.columns:
        return pd.DataFrame(columns=["status", "count"])
    summary = figure_manifest.groupby("status", dropna=False).size().reset_index(name="count")
    return summary.sort_values(["status"])


def _backend_notes(backend: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if backend.empty:
        return pd.DataFrame(), ["ml_backend_summary.csv not found; backend claims could not be audited"]
    keep_cols = [col for col in ["experiment", "model_type", "ml_model", "backend", "backend_label", "feature_source", "postprocess_method"] if col in backend.columns]
    notes = backend[keep_cols].copy()
    if "ml_model" in notes.columns:
        gru_mask = notes["ml_model"].fillna("").str.lower().eq("gru")
    else:
        gru_mask = pd.Series(False, index=notes.index)
    backend_label = notes.get("backend_label", pd.Series("", index=notes.index)).fillna("").astype(str)
    backend_raw = notes.get("backend", pd.Series("", index=notes.index)).fillna("").astype(str)
    fallback_mask = gru_mask & (
        backend_label.str.contains("fallback", case=False, regex=False)
        | backend_raw.str.contains("sklearn", case=False, regex=False)
        | backend_label.str.contains("flattened", case=False, regex=False)
    )
    notes["manuscript_claim"] = np.where(
        fallback_mask,
        "GRU-labelled experiment; report actual backend as fallback MLP on flattened sequences",
        "Report according to backend metadata",
    )
    if fallback_mask.any():
        experiments = ";".join(notes.loc[fallback_mask, "experiment"].astype(str).tolist())
        warnings.append(f"GRU over-claim risk: {experiments} use fallback backend, not real recurrent GRU")
    return notes, warnings


def _reproducibility_checklist(
    comparison_dir: Path,
    figures_dir: Path,
    leakage_status: str,
    physical_status: str,
    ranking: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    backend_warnings: list[str],
) -> pd.DataFrame:
    rows = []

    def add(item: str, status: str, evidence: str, action: str = "") -> None:
        rows.append({"item": item, "status": status, "evidence": evidence, "action": action})

    add(
        "Common validation intersection",
        "PASS" if not ranking.empty and {"common_start_date", "common_end_date"}.issubset(ranking.columns) else "WARN",
        "leaderboard_common_intersection.csv" if (comparison_dir / "leaderboard_common_intersection.csv").exists() else "missing leaderboard_common_intersection.csv",
        "Use only common-window metrics in the manuscript.",
    )
    add(
        "Anti-leakage audit",
        "PASS" if leakage_status == "PASS" else ("WARN" if leakage_status == "WARN" else "FAIL"),
        f"leakage_audit_status.txt={leakage_status}",
        "Do not submit if status becomes FAIL; document WARN items.",
    )
    add(
        "Physical audit",
        "PASS" if physical_status == "PASS" else ("WARN" if physical_status == "WARN" else "FAIL"),
        f"physical_audit_status.txt={physical_status}",
        "Discuss physical limitations when WARN is present.",
    )
    add(
        "Figure manifest",
        "PASS" if not figure_manifest.empty else "WARN",
        str(figures_dir / "phase34_figure_manifest.csv"),
        "Regenerate figures before final manuscript package if missing.",
    )
    add(
        "GRU wording",
        "WARN" if backend_warnings else "PASS",
        " | ".join(backend_warnings) if backend_warnings else "no fallback GRU warning detected",
        "Use backend-aware wording for E7/E8.",
    )
    add(
        "Optional physical components in figures",
        "WARN" if (not figure_manifest.empty and (figure_manifest.get("status") == "skipped").any()) else "PASS",
        "skipped entries in phase34_figure_manifest.csv" if not figure_manifest.empty else "figure manifest unavailable",
        "Only claim Qfast/Qbase/Qtotal figures if columns are exported.",
    )
    return pd.DataFrame(rows)


def _summarize_best_model(ranking: pd.DataFrame) -> dict[str, str]:
    if ranking.empty:
        return {}
    best = ranking.sort_values("rank").iloc[0] if "rank" in ranking.columns else ranking.iloc[0]
    return {
        "experiment": str(best.get("experiment", "")),
        "n_eval": _fmt(best.get("n_eval"), 0),
        "window": f"{best.get('common_start_date', '')} to {best.get('common_end_date', '')}",
        "NSE": _fmt(best.get("NSE")),
        "KGE": _fmt(best.get("KGE")),
        "RMSE": _fmt(best.get("RMSE")),
        "MAE": _fmt(best.get("MAE")),
        "PBIAS": _fmt(best.get("PBIAS")),
    }


def _build_markdown_report(
    *,
    output_root: Path,
    report_dir: Path,
    artifacts: list[PackageArtifact],
    warnings: list[str],
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    ablation: pd.DataFrame,
    regime: pd.DataFrame,
    uncertainty: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    figure_summary: pd.DataFrame,
    backend_notes: pd.DataFrame,
    checklist: pd.DataFrame,
    leakage_status: str,
    physical_status: str,
) -> str:
    best = _summarize_best_model(ranking)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines: list[str] = []
    lines.append("# StoHyMoLAP Phase 3.4C/3.5 — Paper-ready result package")
    lines.append("")
    lines.append(f"Generated at: `{generated_at}`")
    lines.append(f"Output root: `{output_root}`")
    lines.append("")
    lines.append("## 1. Status")
    lines.append("")
    lines.append(f"- Anti-leakage audit status: `{leakage_status}`")
    lines.append(f"- Physical audit status: `{physical_status}`")
    if warnings:
        lines.append(f"- Package warnings: `{len(warnings)}`")
    else:
        lines.append("- Package warnings: `0`")
    lines.append("")
    lines.append("## 2. Best common-window model")
    lines.append("")
    if best:
        lines.append(
            "The best model on the common validation intersection is "
            f"`{best['experiment']}` with NSE={best['NSE']}, KGE={best['KGE']}, "
            f"RMSE={best['RMSE']}, MAE={best['MAE']} and PBIAS={best['PBIAS']} "
            f"over n={best['n_eval']} samples (`{best['window']}`)."
        )
    else:
        lines.append("_No common-window ranking was available._")
    lines.append("")
    lines.append("## 3. Final ranking table")
    lines.append("")
    lines.append(_markdown_table(ranking, ["rank", "experiment", "n_eval", "common_start_date", "common_end_date", *CORE_METRIC_COLUMNS, "backend_label"]))
    lines.append("")
    lines.append("## 4. Selected model metrics for figures")
    lines.append("")
    lines.append(_markdown_table(selected, ["experiment", "NSE", "KGE", "RMSE", "MAE", "PBIAS", "R2", "backend_label"]))
    lines.append("")
    lines.append("## 5. Ablation effects")
    lines.append("")
    lines.append(_markdown_table(ablation, ["ablation", "baseline", "candidate", "delta_NSE", "delta_KGE", "delta_RMSE", "delta_MAE", "delta_abs_PBIAS_improvement"]))
    lines.append("")
    lines.append("## 6. Regime metrics")
    lines.append("")
    if regime.empty:
        lines.append("_No regime metrics were available._")
    else:
        lines.append(_markdown_table(regime, ["experiment", "regime", "n", "NSE", "KGE", "RMSE", "MAE", "PBIAS"], max_rows=40))
    lines.append("")
    lines.append("## 7. Uncertainty and stochastic references")
    lines.append("")
    if uncertainty.empty:
        lines.append("_No uncertainty table was available._")
    else:
        lines.append(_markdown_table(uncertainty, ["experiment", "PICP", "PINAW", "MPIW", "Winkler"]))
        if "PICP" in uncertainty.columns:
            low_picp = uncertainty[pd.to_numeric(uncertainty["PICP"], errors="coerce") < 0.8]
            if not low_picp.empty:
                exp_list = ", ".join(low_picp["experiment"].astype(str).tolist())
                lines.append("")
                lines.append(f"Note: `{exp_list}` should be described as underdispersive stochastic references, not calibrated uncertainty bands.")
    lines.append("")
    lines.append("## 8. Figure availability")
    lines.append("")
    lines.append(_markdown_table(figure_summary, ["status", "count"]))
    lines.append("")
    if not figure_manifest.empty:
        lines.append(_markdown_table(figure_manifest, ["figure_id", "status", "path", "warnings"], max_rows=40))
    lines.append("")
    lines.append("## 9. Backend and manuscript wording")
    lines.append("")
    lines.append(_markdown_table(backend_notes, ["experiment", "ml_model", "backend", "backend_label", "feature_source", "manuscript_claim"], max_rows=20))
    lines.append("")
    lines.append("Use backend-aware wording: when E7/E8 are declared as GRU but run with `fallback_mlp_on_flattened_sequences`, describe them as GRU-labelled hybrid experiments executed with a flattened-sequence MLP fallback, not as real recurrent GRU models.")
    lines.append("")
    lines.append("## 10. Reproducibility checklist")
    lines.append("")
    lines.append(_markdown_table(checklist, ["item", "status", "evidence", "action"], max_rows=20))
    lines.append("")
    lines.append("## 11. Generated package artefacts")
    lines.append("")
    artifact_df = pd.DataFrame([asdict(item) for item in artifacts])
    lines.append(_markdown_table(artifact_df, ["artifact_id", "status", "rows", "path", "notes"], max_rows=50))
    lines.append("")
    if warnings:
        lines.append("## 12. Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append("## 13. Suggested manuscript framing")
    lines.append("")
    lines.append("- Use the common-window leaderboard as the main quantitative result table.")
    lines.append("- Present E8 as the best current model only under the exact backend used in this environment.")
    lines.append("- Discuss E2/E3 stochastic bands as references if PICP is low, not as reliable predictive uncertainty.")
    lines.append("- Do not claim physical component plots until Qfast/Qbase/Qtotal are exported by the experiment runners.")
    lines.append("- Keep audit WARN items visible in supplementary reproducibility material.")
    lines.append("")
    return "\n".join(lines)


def generate_phase34_report(
    output_root: str | Path = "outputs",
    selected_experiments: Iterable[str] | None = None,
) -> dict[str, Path | int | list[str]]:
    output_root = Path(output_root)
    selected_experiments = list(selected_experiments or DEFAULT_SELECTED_EXPERIMENTS)
    comparison_dir = output_root / "comparison"
    figures_dir = output_root / "figures" / "phase34"
    report_dir = output_root / "reports" / "phase34"
    report_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    artifacts: list[PackageArtifact] = []

    leaderboard = _read_csv(comparison_dir / "leaderboard_common_intersection.csv", required=True, warnings=warnings)
    if leaderboard.empty:
        leaderboard = _read_csv(comparison_dir / "validation_metrics_common_intersection.csv", required=True, warnings=warnings)
    backend = _read_csv(comparison_dir / "ml_backend_summary.csv", warnings=warnings)
    ablation = _read_csv(comparison_dir / "ablation_effects_common_intersection.csv", warnings=warnings)
    regime = _read_csv(comparison_dir / "regime_metrics_comparison.csv", warnings=warnings)
    uncertainty = _read_csv(comparison_dir / "uncertainty_comparison.csv", warnings=warnings)
    figure_manifest = _read_csv(figures_dir / "phase34_figure_manifest.csv", warnings=warnings)

    leakage_status = _read_text_status(comparison_dir / "leakage_audit_status.txt")
    physical_status = _read_text_status(comparison_dir / "physical_audit_status.txt")

    ranking = _clean_ranking(leaderboard, backend)
    selected = _selected_metrics(ranking, selected_experiments)
    figure_summary = _figure_availability(figure_manifest)
    backend_note_table, backend_warnings = _backend_notes(backend)
    warnings.extend(backend_warnings)
    checklist = _reproducibility_checklist(
        comparison_dir=comparison_dir,
        figures_dir=figures_dir,
        leakage_status=leakage_status,
        physical_status=physical_status,
        ranking=ranking,
        figure_manifest=figure_manifest,
        backend_warnings=backend_warnings,
    )

    # Keep regime table focused on the models used in the paper-ready figures.
    if not regime.empty and "experiment" in regime.columns:
        regime_selected = regime[regime["experiment"].isin(selected_experiments)].copy()
    else:
        regime_selected = regime

    _write_table(ranking, report_dir / "table_final_ranking_common_window.csv", artifacts, notes="Main quantitative result table")
    _write_table(selected, report_dir / "table_selected_models_common_window.csv", artifacts, notes="Models used in Phase 3.4B figures")
    _write_table(ablation, report_dir / "table_ablation_effects_common_window.csv", artifacts, notes="Ablation deltas on common validation window")
    _write_table(regime_selected, report_dir / "table_regime_metrics_selected.csv", artifacts, notes="Regime metrics for selected models")
    _write_table(uncertainty, report_dir / "table_uncertainty_stochastic_references.csv", artifacts, notes="Stochastic reference diagnostics")
    _write_table(figure_summary, report_dir / "table_figure_status_summary.csv", artifacts, notes="Generated/fallback/skipped figure counts")
    _write_table(backend_note_table, report_dir / "table_backend_manuscript_notes.csv", artifacts, notes="Backend-aware wording notes")
    _write_table(checklist, report_dir / "table_reproducibility_checklist.csv", artifacts, notes="Submission-readiness checks")

    manifest_path = report_dir / "phase34_report_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": [asdict(item) for item in artifacts], "warnings": warnings}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts.append(PackageArtifact("phase34_report_manifest", "generated", str(manifest_path), notes="Machine-readable report package manifest"))

    report_md = _build_markdown_report(
        output_root=output_root,
        report_dir=report_dir,
        artifacts=artifacts,
        warnings=warnings,
        ranking=ranking,
        selected=selected,
        ablation=ablation,
        regime=regime_selected,
        uncertainty=uncertainty,
        figure_manifest=figure_manifest,
        figure_summary=figure_summary,
        backend_notes=backend_note_table,
        checklist=checklist,
        leakage_status=leakage_status,
        physical_status=physical_status,
    )
    report_path = report_dir / "phase34_results_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "report_path": report_path,
        "manifest_path": manifest_path,
        "report_dir": report_dir,
        "n_artifacts": len(artifacts),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 3.4C/3.5 paper-ready result package.")
    parser.add_argument("--output-root", default="outputs", help="Root directory containing comparison, experiments and figures outputs.")
    parser.add_argument(
        "--selected-experiments",
        nargs="*",
        default=DEFAULT_SELECTED_EXPERIMENTS,
        help="Experiments to emphasize in selected-model and regime tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_phase34_report(output_root=args.output_root, selected_experiments=args.selected_experiments)
    print(f"Report Markdown: {result['report_path']}")
    print(f"Report manifest: {result['manifest_path']}")
    print(f"Report artefacts: {result['n_artifacts']}")
    print(f"Warnings: {len(result['warnings'])}")


if __name__ == "__main__":
    main()
