"""Auditoria anti-leakage y QA para Fase 3.4A.

La auditoria no recalcula modelos. Inspecciona configuracion y outputs ya
producidos para dejar evidencia reproducible de que:

* train y validation estan separados temporalmente;
* lags, secuencias y horizonte no usan informacion futura;
* las variables de entrada no incluyen Qobs/targets observados;
* los umbrales de regimen provienen solo de calibracion;
* la ventana comun se aplico de forma consistente;
* los backends ML/GRU quedan documentados;
* el postproceso Qsim>=0 esta auditado para ML/hibridos.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from ..experiments.registry import list_experiments
from ..utils.logging import get_logger

_log = get_logger("diagnostics.leakage_audit")

_SEVERITY_ORDER = {"PASS": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
_PROHIBITED_FEATURE_NAMES = {
    "qobs", "flow_obs", "observed_flow", "obs", "target", "y", "qsim_raw",
}
_ALLOWED_FEATURE_SOURCES = {
    "observed_forcing",
    "ramis_deterministic",
    "stochastic_ensemble",
}
_REAL_GRU_BACKENDS = {"tensorflow", "torch"}
_FALLBACK_SEQUENCE_BACKENDS = {"sklearn_mlp"}


@dataclass
class AuditIssue:
    """Fila individual de auditoria."""

    severity: str
    experiment: str
    check: str
    message: str
    evidence: str = ""


@dataclass
class ExperimentAuditSummary:
    """Resumen por experimento."""

    experiment: str
    status: str
    train_start: Optional[str]
    train_end: Optional[str]
    validation_start: Optional[str]
    validation_end: Optional[str]
    n_train_predictions: Optional[int]
    n_validation_predictions: Optional[int]
    model_type: str
    ml_model: str
    backend: str
    backend_label: str
    feature_source: str
    horizon: Optional[int]
    lags: str
    sequence_length: Optional[int]
    common_window_start: Optional[str]
    common_window_end: Optional[str]
    postprocess_method: str
    postprocess_validation_negatives: Optional[int]
    regime_threshold_source: str


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _issue(rows: List[AuditIssue], severity: str, experiment: str, check: str, message: str, evidence: Any = "") -> None:
    rows.append(
        AuditIssue(
            severity=severity,
            experiment=experiment,
            check=check,
            message=message,
            evidence=_json_text(evidence) if not isinstance(evidence, str) else evidence,
        )
    )


def _status_from_issues(issues: Sequence[AuditIssue]) -> str:
    worst = 0
    for row in issues:
        worst = max(worst, _SEVERITY_ORDER.get(row.severity, 0))
    if worst >= _SEVERITY_ORDER["ERROR"]:
        return "FAIL"
    if worst >= _SEVERITY_ORDER["WARN"]:
        return "WARN"
    return "PASS"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log.warning("No se pudo leer JSON %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _read_predictions(path: Path, rows: List[AuditIssue], eid: str, split: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        _issue(rows, "ERROR", eid, f"predictions_{split}_exists", f"Falta {path.name}.")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _issue(rows, "ERROR", eid, f"predictions_{split}_readable", f"No se pudo leer {path.name}: {exc}")
        return None
    required = {"date", "Qobs", "Qsim"}
    missing = sorted(required.difference(df.columns))
    if missing:
        _issue(rows, "ERROR", eid, f"predictions_{split}_columns", "Faltan columnas requeridas.", missing)
        return None
    if df.empty:
        _issue(rows, "ERROR", eid, f"predictions_{split}_nonempty", f"{path.name} esta vacio.")
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        _issue(rows, "ERROR", eid, f"predictions_{split}_dates", "Hay fechas no parseables.")
    if df["date"].duplicated().any():
        dup = df.loc[df["date"].duplicated(), "date"].iloc[0]
        _issue(rows, "ERROR", eid, f"predictions_{split}_duplicates", "Hay fechas duplicadas.", str(dup))
    if not df["date"].is_monotonic_increasing:
        _issue(rows, "ERROR", eid, f"predictions_{split}_order", "Las fechas no estan ordenadas crecientemente.")
    for col in ["Qobs", "Qsim"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            _issue(rows, "ERROR", eid, f"predictions_{split}_{col}_finite", f"{col} contiene NaN/no numericos.")
    return df


def _date_bounds(df: Optional[pd.DataFrame]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    if df is None or df.empty or "date" not in df.columns:
        return None, None, None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None, int(len(df))
    return dates.min().date().isoformat(), dates.max().date().isoformat(), int(len(df))


def _audit_temporal_split(
    rows: List[AuditIssue],
    eid: str,
    train_pred: Optional[pd.DataFrame],
    val_pred: Optional[pd.DataFrame],
) -> None:
    if train_pred is None or val_pred is None:
        return
    train_dates = pd.to_datetime(train_pred["date"], errors="coerce").dropna().dt.normalize()
    val_dates = pd.to_datetime(val_pred["date"], errors="coerce").dropna().dt.normalize()
    if train_dates.empty or val_dates.empty:
        _issue(rows, "ERROR", eid, "temporal_split_nonempty", "Train o validation no tiene fechas validas.")
        return
    overlap = sorted(set(train_dates).intersection(set(val_dates)))
    if overlap:
        _issue(rows, "ERROR", eid, "temporal_split_no_overlap", "Hay fechas compartidas entre train y validation.", str(overlap[0].date()))
    if train_dates.max() >= val_dates.min():
        _issue(
            rows,
            "ERROR",
            eid,
            "temporal_split_order",
            "max(train) debe ser menor que min(validation).",
            {"max_train": train_dates.max().date().isoformat(), "min_validation": val_dates.min().date().isoformat()},
        )
    else:
        _issue(rows, "PASS", eid, "temporal_split_order", "Train termina antes de validation.")


def _experiment_cfg(cfg: Mapping[str, Any], eid: str) -> Dict[str, Any]:
    return dict((cfg.get("experiments", {}) or {}).get(eid, {}) or {})


def _global_horizon(cfg: Mapping[str, Any]) -> Optional[int]:
    try:
        return int((cfg.get("global", {}) or {}).get("forecast_horizon", 1))
    except Exception:  # noqa: BLE001
        return None


def _feature_lags(cfg: Mapping[str, Any], exp: Mapping[str, Any]) -> List[int]:
    feats = exp.get("features", {}) or {}
    raw_lags = feats.get("lags", (cfg.get("global", {}) or {}).get("lags", []))
    try:
        return [int(v) for v in raw_lags]
    except Exception:  # noqa: BLE001
        return []


def _audit_feature_config(rows: List[AuditIssue], cfg: Mapping[str, Any], eid: str) -> None:
    exp = _experiment_cfg(cfg, eid)
    model_type = str(exp.get("model_type", "physical"))
    feats = exp.get("features", {}) or {}
    source = str(feats.get("source", "physical_only"))
    variables = [str(v) for v in feats.get("variables", [])]
    horizon = _global_horizon(cfg)
    lags = _feature_lags(cfg, exp)

    if horizon is None or horizon < 0:
        _issue(rows, "ERROR", eid, "forecast_horizon_nonnegative", "forecast_horizon debe ser entero >= 0.", horizon)
    else:
        _issue(rows, "PASS", eid, "forecast_horizon_nonnegative", "forecast_horizon es no negativo.", horizon)

    if model_type in {"machine_learning", "hybrid"}:
        bad_lags = [lag for lag in lags if lag < 0]
        if bad_lags:
            _issue(rows, "ERROR", eid, "lags_no_future", "Hay lags negativos que miran al futuro.", bad_lags)
        else:
            _issue(rows, "PASS", eid, "lags_no_future", "Lags no negativos; no se usan predictores futuros.", lags)
    elif model_type == "hybrid_sequence":
        try:
            seq = int(exp.get("sequence_length", 0))
        except Exception:  # noqa: BLE001
            seq = 0
        if seq < 1:
            _issue(rows, "ERROR", eid, "sequence_length_valid", "sequence_length debe ser >= 1.", seq)
        else:
            _issue(rows, "PASS", eid, "sequence_no_future", "La secuencia usa ventanas pasadas hasta t y predice t+horizon.", {"sequence_length": seq, "horizon": horizon})

    if model_type in {"machine_learning", "hybrid", "hybrid_sequence"}:
        if source not in _ALLOWED_FEATURE_SOURCES:
            _issue(rows, "ERROR", eid, "feature_source_allowed", "Fuente de features no reconocida.", source)
        else:
            _issue(rows, "PASS", eid, "feature_source_allowed", "Fuente de features permitida.", source)
        prohibited = [v for v in variables if v.lower() in _PROHIBITED_FEATURE_NAMES]
        if prohibited:
            _issue(rows, "ERROR", eid, "features_no_observed_target", "Variables de entrada incluyen Qobs/target observado.", prohibited)
        else:
            _issue(rows, "PASS", eid, "features_no_observed_target", "Las variables de entrada no incluyen Qobs/target observado.", variables)


def _read_backend_from_joblib(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        import joblib

        obj = joblib.load(path)
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(obj, dict):
        return str(obj.get("backend", ""))
    return str(getattr(obj, "backend", ""))


def classify_backend(ml_model: str, backend: str) -> str:
    """Clasifica el backend para reporte paper-ready."""
    ml = str(ml_model or "").lower()
    be = str(backend or "").lower()
    if ml == "gru":
        if be in _REAL_GRU_BACKENDS:
            return "real_gru"
        if be in _FALLBACK_SEQUENCE_BACKENDS:
            return "fallback_mlp_on_flattened_sequences"
        if be:
            return f"unknown_sequence_backend:{be}"
        return "unknown_sequence_backend"
    if be:
        return be
    if ml:
        return "backend_not_recorded"
    return "physical"


def _audit_backend(rows: List[AuditIssue], eid: str, exp_dir: Path, exp: Mapping[str, Any]) -> Tuple[str, str]:
    ml_model = str(exp.get("ml_model", "") or "")
    model_type = str(exp.get("model_type", "physical") or "physical")
    if not ml_model:
        return "physical", "physical"

    meta = _read_json(exp_dir / "ml_backend.json")
    backend = str(meta.get("backend", "") or "")
    if not backend:
        backend = _read_backend_from_joblib(exp_dir / "model_artifact" / "model.joblib")
    label = classify_backend(ml_model, backend)

    if not backend:
        _issue(rows, "WARN", eid, "ml_backend_recorded", "No se encontro ml_backend.json ni backend legible en model_artifact.", {"ml_model": ml_model, "model_type": model_type})
    else:
        _issue(rows, "PASS", eid, "ml_backend_recorded", "Backend ML registrado.", {"backend": backend, "label": label})

    if ml_model == "gru" and label != "real_gru":
        _issue(
            rows,
            "WARN",
            eid,
            "gru_backend_real",
            "El experimento declarado GRU no debe redactarse como GRU real salvo que backend sea tensorflow/torch.",
            {"backend": backend or "missing", "label": label},
        )
    elif ml_model == "gru":
        _issue(rows, "PASS", eid, "gru_backend_real", "Backend secuencial corresponde a GRU real.", backend)
    return backend or "missing", label


def _audit_evaluation_window(
    rows: List[AuditIssue],
    eid: str,
    exp_dir: Path,
    train_pred: Optional[pd.DataFrame],
    val_pred: Optional[pd.DataFrame],
) -> Tuple[Optional[str], Optional[str]]:
    meta = _read_json(exp_dir / "evaluation_window.json")
    if not meta:
        _issue(rows, "WARN", eid, "evaluation_window_exists", "No existe evaluation_window.json; no se puede probar ventana comun desde metadata.")
        return None, None
    _issue(rows, "PASS", eid, "evaluation_window_exists", "evaluation_window.json presente.")
    val = meta.get("validation", {}) or {}
    tr = meta.get("train", {}) or {}

    if val_pred is not None:
        start, end, n = _date_bounds(val_pred)
        expected = {"start_date": start, "end_date": end, "n_after": n}
        observed = {k: val.get(k) for k in expected}
        if observed != expected:
            _issue(rows, "ERROR", eid, "evaluation_window_validation_matches_predictions", "Metadata de ventana validation no coincide con predictions_validation.csv.", {"expected_from_predictions": expected, "metadata": observed})
        else:
            _issue(rows, "PASS", eid, "evaluation_window_validation_matches_predictions", "Ventana validation coincide con predicciones.", expected)
    if train_pred is not None:
        start, end, n = _date_bounds(train_pred)
        expected = {"start_date": start, "end_date": end, "n_after": n}
        observed = {k: tr.get(k) for k in expected}
        if observed != expected:
            _issue(rows, "ERROR", eid, "evaluation_window_train_matches_predictions", "Metadata de ventana train no coincide con predictions_train.csv.", {"expected_from_predictions": expected, "metadata": observed})
        else:
            _issue(rows, "PASS", eid, "evaluation_window_train_matches_predictions", "Ventana train coincide con predicciones.", expected)
    return val.get("start_date"), val.get("end_date")


def _audit_regime_thresholds(rows: List[AuditIssue], eid: str, exp_dir: Path) -> str:
    meta = _read_json(exp_dir / "regime_thresholds.json")
    if not meta:
        _issue(rows, "WARN", eid, "regime_threshold_source", "Falta regime_thresholds.json; se espera source=train_Qobs_only desde Fase 3.4A.")
        return "missing"
    source = str(meta.get("source", ""))
    if source != "train_Qobs_only":
        _issue(rows, "ERROR", eid, "regime_threshold_source", "Los umbrales de regimen no declaran origen train_Qobs_only.", meta)
    else:
        _issue(rows, "PASS", eid, "regime_threshold_source", "Umbrales de regimen calculados solo con Qobs de calibracion.", meta)
    return source or "missing"


def _audit_postprocessing(rows: List[AuditIssue], eid: str, exp_dir: Path, exp: Mapping[str, Any]) -> Tuple[str, Optional[int]]:
    model_type = str(exp.get("model_type", "physical"))
    expected_applied = model_type in {"machine_learning", "hybrid", "hybrid_sequence"}
    meta = _read_json(exp_dir / "postprocessing_report.json")
    if not meta:
        _issue(rows, "WARN", eid, "postprocessing_report_exists", "Falta postprocessing_report.json.")
        return "missing", None
    applied = bool(meta.get("applied", False))
    method = str(meta.get("method", "missing"))
    nneg = ((meta.get("validation", {}) or {}).get("n_negative_raw"))
    if applied != expected_applied:
        _issue(rows, "ERROR", eid, "postprocessing_expected_scope", "El alcance de postproceso no coincide con el tipo de modelo.", {"model_type": model_type, "applied": applied, "expected_applied": expected_applied})
    else:
        _issue(rows, "PASS", eid, "postprocessing_expected_scope", "Postproceso aplicado solo donde corresponde.", {"model_type": model_type, "method": method, "validation_negatives": nneg})
    return method, int(nneg) if nneg is not None else None


def _audit_common_window_consistency(rows: List[AuditIssue], summaries: Sequence[ExperimentAuditSummary]) -> None:
    vals = [s for s in summaries if s.validation_start and s.validation_end]
    if not vals:
        _issue(rows, "WARN", "GLOBAL", "common_window_consistency", "No hay suficientes ventanas validation para comparar.")
        return
    starts = sorted({s.validation_start for s in vals})
    ends = sorted({s.validation_end for s in vals})
    if len(starts) == 1 and len(ends) == 1:
        _issue(rows, "PASS", "GLOBAL", "common_window_consistency", "Todos los experimentos comparten inicio/fin de validation.", {"start": starts[0], "end": ends[0], "n_experiments": len(vals)})
    else:
        _issue(rows, "ERROR", "GLOBAL", "common_window_consistency", "Los experimentos no comparten la misma ventana validation.", {"starts": starts, "ends": ends})


def _audit_common_intersection(rows: List[AuditIssue], comparison_root: Path, expected_experiments: Sequence[str]) -> None:
    path = comparison_root / "leaderboard_common_intersection.csv"
    if not path.exists():
        _issue(rows, "WARN", "GLOBAL", "common_intersection_exists", "Falta leaderboard_common_intersection.csv.")
        return
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _issue(rows, "ERROR", "GLOBAL", "common_intersection_readable", f"No se pudo leer leaderboard_common_intersection.csv: {exc}")
        return
    missing = sorted(set(expected_experiments).difference(set(df.get("experiment", []))))
    if missing:
        _issue(rows, "WARN", "GLOBAL", "common_intersection_experiment_coverage", "Faltan experimentos en la interseccion comun.", missing)
    else:
        _issue(rows, "PASS", "GLOBAL", "common_intersection_experiment_coverage", "La interseccion comun cubre todos los experimentos esperados.")
    if "n_eval" in df.columns and df["n_eval"].nunique(dropna=True) == 1:
        _issue(rows, "PASS", "GLOBAL", "common_intersection_n_eval_equal", "n_eval es identico para todos los experimentos en la interseccion exacta.", int(df["n_eval"].iloc[0]))
    elif "n_eval" in df.columns:
        _issue(rows, "ERROR", "GLOBAL", "common_intersection_n_eval_equal", "n_eval difiere entre experimentos en la interseccion exacta.", df[["experiment", "n_eval"]].to_dict(orient="records"))


def audit_leakage(
    cfg: Mapping[str, Any],
    exp_root: str | Path,
    comparison_root: str | Path,
    experiment_ids: Optional[Sequence[str]] = None,
    *,
    write: bool = True,
) -> Dict[str, Any]:
    """Ejecuta auditoria anti-leakage y escribe artefactos de Fase 3.4A.

    Parameters
    ----------
    cfg:
        Configuracion YAML ya cargada/mezclada.
    exp_root:
        Directorio ``outputs/experiments``.
    comparison_root:
        Directorio ``outputs/comparison`` donde se escriben reportes.
    experiment_ids:
        Subconjunto a auditar. Si es ``None``, usa ``list_experiments(cfg)``.
    write:
        Si es True escribe CSV/JSON/MD en ``comparison_root``.
    """
    exp_root = Path(exp_root)
    comparison_root = Path(comparison_root)
    experiment_ids = list(experiment_ids or list_experiments(dict(cfg)))
    rows: List[AuditIssue] = []
    summaries: List[ExperimentAuditSummary] = []

    global_cfg = cfg.get("global", {}) or {}
    state_mode = str(global_cfg.get("ramis_state_mode", ""))
    if state_mode == "continuous_train_validation":
        _issue(rows, "PASS", "GLOBAL", "ramis_state_mode", "RAMIS mantiene estado continuo train+validation.", state_mode)
    else:
        _issue(rows, "WARN", "GLOBAL", "ramis_state_mode", "RAMIS no declara estado continuo; revisar posible reinicio con Qobs de validation.", state_mode or "missing")

    for eid in experiment_ids:
        exp = _experiment_cfg(cfg, eid)
        exp_dir = exp_root / eid
        exp_issues_start = len(rows)
        if not exp_dir.exists():
            _issue(rows, "ERROR", eid, "experiment_output_dir_exists", f"Falta directorio de outputs: {exp_dir}")
        else:
            _issue(rows, "PASS", eid, "experiment_output_dir_exists", "Directorio de outputs presente.")

        _audit_feature_config(rows, cfg, eid)
        train_pred = _read_predictions(exp_dir / "predictions_train.csv", rows, eid, "train")
        val_pred = _read_predictions(exp_dir / "predictions_validation.csv", rows, eid, "validation")
        _audit_temporal_split(rows, eid, train_pred, val_pred)
        win_start, win_end = _audit_evaluation_window(rows, eid, exp_dir, train_pred, val_pred)
        backend, backend_label = _audit_backend(rows, eid, exp_dir, exp)
        method, nneg = _audit_postprocessing(rows, eid, exp_dir, exp)
        threshold_source = _audit_regime_thresholds(rows, eid, exp_dir)

        tr_start, tr_end, ntr = _date_bounds(train_pred)
        va_start, va_end, nva = _date_bounds(val_pred)
        lags = _feature_lags(cfg, exp)
        horizon = _global_horizon(cfg)
        exp_rows = rows[exp_issues_start:]
        summaries.append(
            ExperimentAuditSummary(
                experiment=eid,
                status=_status_from_issues(exp_rows),
                train_start=tr_start,
                train_end=tr_end,
                validation_start=va_start,
                validation_end=va_end,
                n_train_predictions=ntr,
                n_validation_predictions=nva,
                model_type=str(exp.get("model_type", "physical")),
                ml_model=str(exp.get("ml_model", "") or ""),
                backend=backend,
                backend_label=backend_label,
                feature_source=str((exp.get("features", {}) or {}).get("source", "physical_only")),
                horizon=horizon,
                lags=",".join(str(v) for v in lags),
                sequence_length=int(exp.get("sequence_length")) if exp.get("sequence_length") is not None else None,
                common_window_start=win_start,
                common_window_end=win_end,
                postprocess_method=method,
                postprocess_validation_negatives=nneg,
                regime_threshold_source=threshold_source,
            )
        )

    _audit_common_window_consistency(rows, summaries)
    _audit_common_intersection(rows, comparison_root, experiment_ids)

    status = _status_from_issues(rows)
    issues_df = pd.DataFrame([asdict(r) for r in rows])
    summary_df = pd.DataFrame([asdict(s) for s in summaries])
    result = {
        "status": status,
        "audit_issues": issues_df,
        "summary": summary_df,
        "issues_path": comparison_root / "leakage_audit_issues.csv",
        "summary_path": comparison_root / "leakage_audit_summary.csv",
        "report_path": comparison_root / "leakage_audit_report.md",
        "json_path": comparison_root / "leakage_audit.json",
    }
    if write:
        write_leakage_audit_outputs(result)
    return result


def write_leakage_audit_outputs(result: Mapping[str, Any]) -> None:
    """Escribe CSV, JSON y Markdown de auditoria."""
    issues = result["audit_issues"]
    summary = result["summary"]
    issues_path = Path(result["issues_path"])
    summary_path = Path(result["summary_path"])
    report_path = Path(result["report_path"])
    json_path = Path(result["json_path"])
    issues_path.parent.mkdir(parents=True, exist_ok=True)

    issues.to_csv(issues_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "status": result["status"],
        "summary": summary.to_dict(orient="records"),
        "issues": issues.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_markdown_report(result), encoding="utf-8")
    _log.info("Auditoria leakage Fase 3.4A: %s (%s)", result["status"], report_path)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Renderiza una tabla Markdown sin depender de pandas[tabulate].

    pandas.DataFrame.to_markdown requiere la dependencia opcional ``tabulate``.
    Para que la auditoria funcione en entornos limpios del proyecto, usamos un
    renderizador minimo y suficiente para los reportes de QA.
    """
    if df.empty:
        return ""

    table = df.copy()
    table = table.where(pd.notna(table), "")

    headers = [str(col) for col in table.columns]
    rows = [
        [str(value).replace("\n", " ") for value in row]
        for row in table.to_numpy(dtype=object)
    ]

    def esc(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [
        "| " + " | ".join(esc(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(value) for value in row) + " |")
    return "\n".join(lines)


def _render_markdown_report(result: Mapping[str, Any]) -> str:
    issues: pd.DataFrame = result["audit_issues"]
    summary: pd.DataFrame = result["summary"]
    lines = [
        "# Auditoria anti-leakage y QA — Fase 3.4A",
        "",
        f"**Estado:** `{result['status']}`",
        "",
        "## Resumen por experimento",
        "",
    ]
    if summary.empty:
        lines.append("No hay experimentos auditados.")
    else:
        cols = [
            "experiment", "status", "train_start", "train_end", "validation_start",
            "validation_end", "n_validation_predictions", "backend_label",
            "postprocess_method", "regime_threshold_source",
        ]
        lines.append(_dataframe_to_markdown(summary[cols]))
    lines.extend(["", "## Hallazgos WARN/ERROR", ""])
    severe = issues[issues["severity"].isin(["WARN", "ERROR"])] if not issues.empty else pd.DataFrame()
    if severe.empty:
        lines.append("No se detectaron WARN/ERROR.")
    else:
        lines.append(_dataframe_to_markdown(severe[["severity", "experiment", "check", "message", "evidence"]]))
    lines.extend([
        "",
        "## Criterios auditados",
        "",
        "- Separacion temporal estricta entre calibracion y validacion.",
        "- Lags no negativos y secuencias formadas con informacion hasta t para predecir t+horizon.",
        "- Variables predictoras sin Qobs/target observado.",
        "- Ventana comun declarada consistente con los CSV de prediccion.",
        "- Umbrales de regimen documentados como `train_Qobs_only`.",
        "- Backend ML registrado para distinguir GRU real de fallback MLP.",
        "- Postproceso no-negativo aplicado solo a ML/hibridos.",
        "",
    ])
    return "\n".join(lines)
