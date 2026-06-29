"""Ventana comun de evaluacion para la matriz experimental.

Los modelos fisicos pueden producir predicciones desde el primer dia de
validacion, mientras que los modelos ML/hibridos pierden dias iniciales por
lags o longitud de secuencia. Para que el leaderboard de Fase 3 sea comparable,
esta utilidad define una ventana comun de fechas a partir del mayor warm-up
requerido por los experimentos declarados en la configuracion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationWindow:
    """Descripcion serializable de la ventana comun."""

    enabled: bool
    mode: str
    start_offset: int
    end_trim: int
    inferred_from: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "start_offset": int(self.start_offset),
            "end_trim": int(self.end_trim),
            "inferred_from": self.inferred_from,
        }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return int(default)


def _max_lag(lags: Iterable[Any]) -> int:
    vals: List[int] = []
    for lag in lags:
        vals.append(_safe_int(lag, 0))
    return max(vals) if vals else 0


def experiment_warmup_offset(global_cfg: Dict[str, Any], exp_cfg: Dict[str, Any]) -> int:
    """Dias iniciales que un experimento no puede evaluar de forma justa.

    El offset se expresa como indice de fecha objetivo dentro del subconjunto
    evaluado. Para features en t y objetivo en t+horizon:

    * tabular: primer objetivo = max(lags) + horizon;
    * secuencial: primer objetivo = sequence_length - 1 + horizon;
    * fisico puro: 0.
    """
    horizon = _safe_int(global_cfg.get("forecast_horizon", 1), 1)
    model_type = str(exp_cfg.get("model_type", "physical"))

    if model_type == "hybrid_sequence":
        seq = max(1, _safe_int(exp_cfg.get("sequence_length", 7), 7))
        return max(0, seq - 1 + horizon)

    if model_type in {"machine_learning", "hybrid"}:
        features = exp_cfg.get("features", {}) or {}
        lags = features.get("lags", global_cfg.get("lags", [0, 1, 2]))
        return max(0, _max_lag(lags) + horizon)

    return 0


def infer_common_start_offset(cfg: Dict[str, Any]) -> int:
    """Mayor warm-up de la matriz declarada en ``experiments``."""
    global_cfg = cfg.get("global", {}) or {}
    experiments = cfg.get("experiments", {}) or {}
    if not experiments:
        return 0
    return max(experiment_warmup_offset(global_cfg, exp) for exp in experiments.values())


def resolve_evaluation_window(cfg: Dict[str, Any]) -> EvaluationWindow:
    """Resuelve la politica de ventana comun desde la configuracion.

    Bloque esperado opcional::

      evaluation:
        common_window:
          enabled: true
          mode: declared_matrix_max_warmup
          start_offset: auto
          end_trim: 0
    """
    eval_cfg = (cfg.get("evaluation", {}) or {}).get("common_window", {}) or {}
    enabled = bool(eval_cfg.get("enabled", True))
    mode = str(eval_cfg.get("mode", "declared_matrix_max_warmup"))
    end_trim = max(0, _safe_int(eval_cfg.get("end_trim", 0), 0))

    raw_start = eval_cfg.get("start_offset", "auto")
    if raw_start in {None, "auto"}:
        start_offset = infer_common_start_offset(cfg)
        inferred_from = "experiments"
    else:
        start_offset = max(0, _safe_int(raw_start, 0))
        inferred_from = "config"

    if not enabled:
        start_offset = 0
        end_trim = 0

    return EvaluationWindow(
        enabled=enabled,
        mode=mode,
        start_offset=start_offset,
        end_trim=end_trim,
        inferred_from=inferred_from,
    )


def apply_common_window(
    reference_dates: Iterable[Any],
    dates: Iterable[Any],
    arrays: Tuple[np.ndarray, ...],
    window: EvaluationWindow,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Dict[str, Any]]:
    """Filtra arrays a la ventana comun definida por fechas.

    ``reference_dates`` corresponde al subconjunto original (train o validation)
    y ``dates`` a las fechas ya alineadas con las predicciones. Esto permite
    recortar fisicos y ML usando la misma fecha objetivo.
    """
    dates_ref = pd.to_datetime(pd.Series(list(reference_dates)))
    dates_eval = pd.to_datetime(pd.Series(list(dates)))

    if len(dates_ref) == 0 or len(dates_eval) == 0:
        return dates_eval.to_numpy(), arrays, {
            "enabled": bool(window.enabled),
            "start_date": None,
            "end_date": None,
            "n_before": int(len(dates_eval)),
            "n_after": int(len(dates_eval)),
        }

    if not window.enabled:
        start_date = dates_ref.iloc[0]
        end_date = dates_ref.iloc[-1]
        mask = np.ones(len(dates_eval), dtype=bool)
    else:
        start_idx = min(max(0, int(window.start_offset)), len(dates_ref) - 1)
        end_idx = len(dates_ref) - 1 - min(max(0, int(window.end_trim)), len(dates_ref) - 1)
        if end_idx < start_idx:
            raise ValueError(
                "Ventana comun invalida: end_idx < start_idx. "
                f"start_offset={window.start_offset}, end_trim={window.end_trim}, n={len(dates_ref)}"
            )
        start_date = dates_ref.iloc[start_idx]
        end_date = dates_ref.iloc[end_idx]
        mask = (dates_eval >= start_date) & (dates_eval <= end_date)
        mask = mask.to_numpy(dtype=bool)

    filtered_dates = dates_eval.loc[mask].to_numpy()
    filtered_arrays = tuple(np.asarray(arr)[mask] for arr in arrays)
    meta = {
        "enabled": bool(window.enabled),
        "start_date": None if pd.isna(start_date) else str(pd.Timestamp(start_date).date()),
        "end_date": None if pd.isna(end_date) else str(pd.Timestamp(end_date).date()),
        "n_before": int(len(dates_eval)),
        "n_after": int(mask.sum()),
    }
    return filtered_dates, filtered_arrays, meta
