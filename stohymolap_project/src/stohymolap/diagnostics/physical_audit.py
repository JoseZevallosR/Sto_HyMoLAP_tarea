"""Auditoria fisica de experimentos RAMIS/baseflow.

Fase 2.6: esta capa no cambia el modelo; revisa las salidas producidas por
E1/E2 o por cualquier experimento fisico/estocastico para confirmar que las
series, parametros y componentes sean trazables y fisicamente consistentes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..metrics.deterministic import all_deterministic, pbias

_REQUIRED_PARAMETER_COLUMNS = {
    "mu", "lambda", "sigma", "alpha", "beta", "c_r", "k_b", "S0_b",
}
_REQUIRED_TOPK_COLUMNS = _REQUIRED_PARAMETER_COLUMNS | {
    "nse", "J", "KGE", "PBIAS", "rank", "selected",
}


@dataclass
class AuditIssue:
    """Una alerta o error de auditoria."""

    experiment: str
    level: str
    check: str
    message: str

    def as_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "level": self.level,
            "check": self.check,
            "message": self.message,
        }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _finite_series(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    for c in cols:
        if c not in df.columns:
            return False
        if not np.all(np.isfinite(df[c].to_numpy(dtype=float))):
            return False
    return True


def _param_stability_rows(df: pd.DataFrame) -> pd.Series:
    """True para filas RAMIS con estabilidad basica mu/lambda < 1."""
    mu = df["mu"].astype(float)
    lam = df["lambda"].astype(float)
    sigma = df["sigma"].astype(float)
    alpha_area_ok = True
    return (
        np.isfinite(mu) & np.isfinite(lam) & np.isfinite(sigma)
        & (mu > 0.5) & (lam > 0.0) & (sigma >= 0.0)
        & ((mu / lam) > 0.0) & ((mu / lam) < 1.0)
        & alpha_area_ok
    )


def _check_parameters(eid: str, exp_dir: Path) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    best_path = exp_dir / "best_parameters.csv"
    top_path = exp_dir / "top_k_parameters.csv"

    if not best_path.exists():
        issues.append(AuditIssue(eid, "ERROR", "best_parameters", "No existe best_parameters.csv."))
        return issues
    best = _read_csv(best_path)
    missing_best = sorted(_REQUIRED_PARAMETER_COLUMNS - set(best.columns))
    if missing_best:
        issues.append(AuditIssue(eid, "ERROR", "best_parameters", f"Faltan columnas: {missing_best}"))
    elif len(best) != 1:
        issues.append(AuditIssue(eid, "ERROR", "best_parameters", f"Debe tener 1 fila; tiene {len(best)}."))
    else:
        if not bool(_param_stability_rows(best).iloc[0]):
            issues.append(AuditIssue(eid, "ERROR", "ramis_stability", "best_parameters no cumple estabilidad mu/lambda/sigma."))
        if "selection_strategy" in best.columns and str(best.loc[0, "selection_strategy"]) != "best_j":
            issues.append(AuditIssue(eid, "WARN", "selection_strategy", "La seleccion no es best_j; revisar si es un modo legado."))
        if "selected_rank" in best.columns and int(float(best.loc[0, "selected_rank"])) != 1:
            issues.append(AuditIssue(eid, "WARN", "selected_rank", "El parametro final no proviene de la fila rank=1 del top-K."))

    if not top_path.exists():
        issues.append(AuditIssue(eid, "ERROR", "top_k_parameters", "No existe top_k_parameters.csv."))
        return issues
    top = _read_csv(top_path)
    missing_top = sorted(_REQUIRED_TOPK_COLUMNS - set(top.columns))
    if missing_top:
        issues.append(AuditIssue(eid, "ERROR", "top_k_parameters", f"Faltan columnas: {missing_top}"))
        return issues
    if top.empty:
        issues.append(AuditIssue(eid, "ERROR", "top_k_parameters", "top_k_parameters.csv esta vacio."))
        return issues
    if not _param_stability_rows(top).all():
        n_bad = int((~_param_stability_rows(top)).sum())
        issues.append(AuditIssue(eid, "ERROR", "ramis_stability_topk", f"{n_bad} filas top-K no cumplen estabilidad."))
    j = top["J"].astype(float).to_numpy()
    if not np.all(np.isfinite(j)):
        issues.append(AuditIssue(eid, "ERROR", "top_k_J", "J contiene valores no finitos."))
    elif len(j) > 1 and np.any(np.diff(j) < -1e-12):
        issues.append(AuditIssue(eid, "ERROR", "top_k_order", "top_k_parameters.csv no esta ordenado por J ascendente."))
    selected_count = int(np.isclose(top["selected"].astype(float), 1.0).sum())
    if selected_count != 1:
        issues.append(AuditIssue(eid, "ERROR", "top_k_selected", f"Debe haber una sola fila selected=1; hay {selected_count}."))
    return issues


def _check_predictions(eid: str, exp_dir: Path) -> tuple[List[AuditIssue], dict]:
    issues: List[AuditIssue] = []
    summary: dict = {"experiment": eid}

    for split in ("train", "validation"):
        p = exp_dir / f"predictions_{split}.csv"
        if not p.exists():
            issues.append(AuditIssue(eid, "ERROR", f"predictions_{split}", f"No existe {p.name}."))
            continue
        df = _read_csv(p)
        for c in ("date", "Qobs", "Qsim"):
            if c not in df.columns:
                issues.append(AuditIssue(eid, "ERROR", f"predictions_{split}", f"Falta columna {c}."))
        if not {"Qobs", "Qsim"}.issubset(df.columns):
            continue
        qobs = df["Qobs"].to_numpy(dtype=float)
        qsim = df["Qsim"].to_numpy(dtype=float)
        mask = np.isfinite(qobs) & np.isfinite(qsim)
        summary[f"n_{split}"] = int(mask.sum())
        if mask.sum() == 0:
            issues.append(AuditIssue(eid, "ERROR", f"finite_{split}", "No hay pares Qobs/Qsim finitos."))
            continue
        if np.any(qsim[mask] < -1e-9):
            issues.append(AuditIssue(eid, "ERROR", f"nonnegative_{split}", "Qsim contiene valores negativos."))
        metrics = all_deterministic(qobs, qsim)
        for key, val in metrics.items():
            summary[f"{split}_{key}"] = float(val) if np.isfinite(val) else np.nan
        summary[f"{split}_Qobs_mean"] = float(np.nanmean(qobs))
        summary[f"{split}_Qsim_mean"] = float(np.nanmean(qsim))
        summary[f"{split}_Qsim_Qobs_ratio"] = float(np.nanmean(qsim) / np.nanmean(qobs)) if np.nanmean(qobs) != 0 else np.nan
    return issues, summary


def _check_components(eid: str, exp_dir: Path) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    for split in ("train", "validation"):
        p = exp_dir / f"physical_components_{split}.csv"
        pred_path = exp_dir / f"predictions_{split}.csv"
        if not p.exists():
            issues.append(AuditIssue(eid, "WARN", f"components_{split}", f"No existe {p.name}; se recomienda regenerar con Fase 2.6."))
            continue
        comp = _read_csv(p)
        required = {"date", "Qtotal", "Qbase", "Qfast"}
        missing = sorted(required - set(comp.columns))
        if missing:
            issues.append(AuditIssue(eid, "ERROR", f"components_{split}", f"Faltan columnas: {missing}"))
            continue
        if not _finite_series(comp, ["Qtotal", "Qbase", "Qfast"]):
            issues.append(AuditIssue(eid, "ERROR", f"components_finite_{split}", "Qtotal/Qbase/Qfast contienen valores no finitos."))
            continue
        qtotal = comp["Qtotal"].to_numpy(dtype=float)
        qbase = comp["Qbase"].to_numpy(dtype=float)
        qfast = comp["Qfast"].to_numpy(dtype=float)
        if np.any(qbase < -1e-9):
            issues.append(AuditIssue(eid, "ERROR", f"qbase_nonnegative_{split}", "Qbase contiene valores negativos."))
        if np.any(qfast < -1e-9):
            issues.append(AuditIssue(eid, "WARN", f"qfast_nonnegative_{split}", "Qfast contiene valores negativos; revisar clamp/ruido."))
        err = np.nanmax(np.abs((qfast + qbase) - qtotal))
        if not np.isfinite(err) or err > 1e-6:
            issues.append(AuditIssue(eid, "ERROR", f"component_balance_{split}", f"max|Qfast+Qbase-Qtotal|={err:.3e}."))
        if pred_path.exists():
            pred = _read_csv(pred_path)
            if "Qsim" in pred.columns and len(pred) == len(comp):
                diff = np.nanmax(np.abs(pred["Qsim"].to_numpy(dtype=float) - qtotal))
                if np.isfinite(diff) and diff > 1e-6:
                    issues.append(AuditIssue(eid, "ERROR", f"prediction_component_match_{split}", f"Qsim no coincide con Qtotal; max diff={diff:.3e}."))
        if np.nanmean(qtotal) > 0:
            frac = float(np.nanmean(qbase) / np.nanmean(qtotal))
            if frac < -1e-9 or frac > 1.05:
                issues.append(AuditIssue(eid, "WARN", f"baseflow_fraction_{split}", f"Fraccion media Qbase/Qtotal={frac:.3f}; revisar parametros."))
    return issues


def _check_regimes(eid: str, exp_dir: Path) -> List[AuditIssue]:
    issues: List[AuditIssue] = []
    p = exp_dir / "metrics_by_regime.csv"
    if not p.exists():
        issues.append(AuditIssue(eid, "WARN", "metrics_by_regime", "No existe metrics_by_regime.csv."))
        return issues
    df = _read_csv(p)
    expected = {"low", "mid", "high"}
    got = set(df.get("regime", pd.Series(dtype=str)).astype(str))
    if got != expected:
        issues.append(AuditIssue(eid, "ERROR", "regime_set", f"Regimenes esperados {expected}; encontrados {got}."))
    if "n" in df.columns and (df["n"].astype(float) <= 0).any():
        issues.append(AuditIssue(eid, "ERROR", "regime_n", "Algun regimen tiene n<=0."))
    if "PBIAS" in df.columns:
        high_abs_bias = df.loc[df["PBIAS"].astype(float).abs() > 100.0, "regime"].astype(str).tolist()
        if high_abs_bias:
            issues.append(AuditIssue(eid, "WARN", "regime_pbias", f"PBIAS absoluto >100% en regimenes {high_abs_bias}."))
    return issues


def audit_experiment_outputs(exp_root: str | Path, experiment_id: str) -> tuple[pd.DataFrame, dict]:
    """Audita un experimento ya ejecutado y devuelve (issues, summary)."""
    exp_root = Path(exp_root)
    exp_dir = exp_root / experiment_id
    issues: List[AuditIssue] = []
    if not exp_dir.exists():
        issues.append(AuditIssue(experiment_id, "ERROR", "experiment_dir", f"No existe {exp_dir}."))
        return pd.DataFrame([i.as_dict() for i in issues]), {"experiment": experiment_id}

    issues.extend(_check_parameters(experiment_id, exp_dir))
    pred_issues, summary = _check_predictions(experiment_id, exp_dir)
    issues.extend(pred_issues)
    issues.extend(_check_components(experiment_id, exp_dir))
    issues.extend(_check_regimes(experiment_id, exp_dir))
    return pd.DataFrame([i.as_dict() for i in issues]), summary


def _status_from_issues(issues: pd.DataFrame) -> str:
    if issues.empty:
        return "PASS"
    levels = set(issues["level"].astype(str))
    if "ERROR" in levels:
        return "FAIL"
    return "WARN"




def _format_markdown_table(df: pd.DataFrame) -> str:
    """Formatea una tabla Markdown sin depender de pandas.to_markdown/tabulate."""
    if df.empty:
        return ""

    work = df.copy()
    for col in work.columns:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))

    headers = [str(c) for c in work.columns]
    rows = work.astype(str).values.tolist()
    widths = []
    for i, header in enumerate(headers):
        max_cell = max([len(row[i]) for row in rows], default=0)
        widths.append(max(len(header), max_cell))

    def fmt_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(row) for row in rows])


def audit_physical_experiments(
    exp_root: str | Path,
    comparison_root: str | Path,
    experiment_ids: Iterable[str] = ("E1_RAMIS_DET_BF", "E2_RAMIS_LEVY_BF"),
) -> dict:
    """Audita una lista de experimentos y escribe salidas de Fase 2.6.

    Returns
    -------
    dict con status, audit_summary, audit_issues y rutas de salida.
    """
    exp_root = Path(exp_root)
    comparison_root = Path(comparison_root)
    comparison_root.mkdir(parents=True, exist_ok=True)

    all_issues = []
    summaries = []
    for eid in experiment_ids:
        issues, summary = audit_experiment_outputs(exp_root, eid)
        if not issues.empty:
            all_issues.append(issues)
        summaries.append(summary)

    issues_df = pd.concat(all_issues, ignore_index=True) if all_issues else pd.DataFrame(columns=["experiment", "level", "check", "message"])
    summary_df = pd.DataFrame(summaries)
    status = _status_from_issues(issues_df)

    issues_path = comparison_root / "physical_audit_issues.csv"
    summary_path = comparison_root / "physical_audit_summary.csv"
    report_path = comparison_root / "physical_audit_report.md"
    status_path = comparison_root / "physical_audit_status.txt"

    issues_df.to_csv(issues_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    status_path.write_text(status + "\n", encoding="utf-8")
    _write_report(report_path, status, summary_df, issues_df)

    return {
        "status": status,
        "audit_summary": summary_df,
        "audit_issues": issues_df,
        "issues_path": issues_path,
        "summary_path": summary_path,
        "report_path": report_path,
        "status_path": status_path,
    }


def _write_report(path: Path, status: str, summary: pd.DataFrame, issues: pd.DataFrame) -> None:
    lines = ["# Auditoria fisica RAMIS/baseflow - Fase 2.6\n"]
    lines.append(f"**Estado:** `{status}`\n")
    lines.append("## Resumen E1/E2\n")
    if summary.empty:
        lines.append("No hay resumen disponible.\n")
    else:
        cols = [c for c in [
            "experiment", "n_train", "train_NSE", "train_KGE", "train_PBIAS",
            "n_validation", "validation_NSE", "validation_KGE", "validation_PBIAS",
            "validation_Qsim_Qobs_ratio",
        ] if c in summary.columns]
        lines.append(_format_markdown_table(summary[cols]))
        lines.append("")

    lines.append("## Incidencias\n")
    if issues.empty:
        lines.append("No se encontraron errores ni advertencias.\n")
    else:
        lines.append(_format_markdown_table(issues))
        lines.append("")

    lines.append("## Criterio de cierre\n")
    lines.append(
        "Fase 2 se considera cerrada si el estado es `PASS` o, como maximo, `WARN` "
        "con advertencias metodologicamente explicables. Un estado `FAIL` requiere "
        "corregir salidas, parametros o componentes antes de pasar a ablaciones/ML.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")
