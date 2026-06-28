"""Validaciones de integridad, física básica y anti-fuga temporal.

Este módulo centraliza chequeos ligeros que se usan desde el runner y los tests.
No modifica los datos; solo falla temprano cuando detecta condiciones que harían
que una comparación experimental sea inválida o poco reproducible.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Error explícito para fallas de validación del pipeline."""


def _as_series(values: Iterable, name: str = "values") -> pd.Series:
    """Convierte una entrada tipo array/Series en ``pd.Series`` sin mutarla."""
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values, name=name)


def _finite_numeric(values: Iterable, name: str) -> np.ndarray:
    arr = pd.to_numeric(_as_series(values, name=name), errors="coerce").to_numpy(dtype=float)
    return arr


def check_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Verifica que un DataFrame contenga todas las columnas requeridas."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValidationError(
            "Faltan columnas requeridas: "
            + ", ".join(missing)
            + f". Columnas disponibles: {list(df.columns)}"
        )


def qobs_source_summary(df: pd.DataFrame) -> Mapping[str, int]:
    """Resume el origen de Qobs: observed/missing/filled_from_qsim."""
    if "Qobs_source" not in df.columns:
        return {"unknown": int(len(df))}
    counts = df["Qobs_source"].fillna("unknown").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def validate_hydro_dataframe(
    df: pd.DataFrame,
    *,
    require_daily: bool = True,
    allow_missing_qobs: bool = False,
    allow_filled_qobs: bool = False,
) -> None:
    """Valida una tabla hidrometeorológica ya normalizada.

    Reglas principales:
    - columnas canónicas presentes;
    - fechas válidas y sin duplicados;
    - precipitación, PET si existe, temperaturas y caudales finitos/no negativos
      donde corresponda;
    - no aceptar relleno Qobs<-Qsim si no fue declarado explícitamente;
    - continuidad diaria opcional.
    """
    check_required_columns(df, ["date", "Qobs", "P", "Tmin", "Tmax"])
    if df.empty:
        raise ValidationError("El DataFrame hidrológico está vacío.")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise ValidationError("Hay fechas inválidas o no parseables en la columna date.")
    if dates.duplicated().any():
        dup = dates[dates.duplicated()].iloc[0]
        raise ValidationError(f"Hay fechas duplicadas; primera duplicada: {dup}.")
    if not dates.is_monotonic_increasing:
        raise ValidationError("Las fechas deben estar ordenadas crecientemente.")
    if require_daily:
        check_daily_continuity(dates)

    p = _finite_numeric(df["P"], "P")
    if np.isnan(p).any():
        raise ValidationError("Hay valores faltantes/no numéricos en P.")
    if np.any(p < 0):
        first = int(np.where(p < 0)[0][0])
        raise ValidationError(f"P negativa detectada en fila {first}: {p[first]}.")

    qobs = _finite_numeric(df["Qobs"], "Qobs")
    if not allow_missing_qobs and np.isnan(qobs).any():
        raise ValidationError("Hay Qobs faltante y allow_missing_qobs=False.")
    if np.any(qobs[np.isfinite(qobs)] < 0):
        first = int(np.where(qobs < 0)[0][0])
        raise ValidationError(f"Qobs negativo detectado en fila {first}: {qobs[first]}.")

    tmin = _finite_numeric(df["Tmin"], "Tmin")
    tmax = _finite_numeric(df["Tmax"], "Tmax")
    if np.isnan(tmin).any() or np.isnan(tmax).any():
        raise ValidationError("Hay Tmin/Tmax faltante o no numérico.")
    bad_temp = tmin > tmax
    if np.any(bad_temp):
        first = int(np.where(bad_temp)[0][0])
        raise ValidationError(
            f"Tmin mayor que Tmax en fila {first}: Tmin={tmin[first]}, Tmax={tmax[first]}."
        )

    if "PET" in df.columns:
        check_pet_non_negative(df["PET"])
    if "Peff" in df.columns:
        peff = _finite_numeric(df["Peff"], "Peff")
        if np.isnan(peff).any():
            raise ValidationError("Hay Peff faltante/no numérico.")
        if np.any(peff < -1e-12):
            first = int(np.where(peff < -1e-12)[0][0])
            raise ValidationError(f"Peff negativa detectada en fila {first}: {peff[first]}.")

    if not allow_filled_qobs and "Qobs_is_filled_from_qsim" in df.columns:
        filled = df["Qobs_is_filled_from_qsim"].fillna(False).astype(bool)
        if filled.any():
            raise ValidationError(
                "Se detectó Qobs rellenado desde Qsim, pero allow_filled_qobs=False."
            )


def check_pet_non_negative(values: Iterable) -> None:
    arr = _finite_numeric(values, "PET")
    if np.isnan(arr).any():
        raise ValidationError("Hay PET faltante/no numérico.")
    if np.any(arr < -1e-12):
        first = int(np.where(arr < -1e-12)[0][0])
        raise ValidationError(f"PET negativa detectada en fila {first}: {arr[first]}.")


def check_temporal_split(train_dates: Iterable, val_dates: Iterable) -> None:
    """Verifica que validación ocurra estrictamente después de calibración."""
    tr = pd.to_datetime(_as_series(train_dates, "train_dates"), errors="coerce").dropna()
    va = pd.to_datetime(_as_series(val_dates, "val_dates"), errors="coerce").dropna()
    if tr.empty or va.empty:
        raise ValidationError("Split temporal inválido: train o validation sin fechas válidas.")
    if tr.duplicated().any() or va.duplicated().any():
        raise ValidationError("Split temporal inválido: fechas duplicadas dentro de train o validation.")
    overlap = set(tr.dt.normalize()).intersection(set(va.dt.normalize()))
    if overlap:
        first = min(overlap)
        raise ValidationError(f"Fuga temporal: fecha presente en train y validation: {first.date()}.")
    if tr.max() >= va.min():
        raise ValidationError(
            f"Fuga temporal: max(train)={tr.max()} debe ser menor que min(validation)={va.min()}."
        )


def check_daily_continuity(dates: Iterable) -> None:
    """Exige una serie diaria ordenada, sin duplicados ni huecos."""
    d = pd.to_datetime(_as_series(dates, "date"), errors="coerce").dropna().reset_index(drop=True)
    if len(d) < 2:
        return
    if not d.is_monotonic_increasing:
        raise ValidationError("Las fechas no están ordenadas crecientemente.")
    if d.duplicated().any():
        raise ValidationError("La serie diaria contiene fechas duplicadas.")
    delta = d.diff().dropna()
    bad = delta != pd.Timedelta(days=1)
    if bad.any():
        idx = int(np.where(bad.to_numpy())[0][0]) + 1
        raise ValidationError(
            "La serie diaria no es continua: "
            f"salto entre {d.iloc[idx - 1].date()} y {d.iloc[idx].date()} = {delta.iloc[idx - 1]}."
        )


def check_lags_no_future(horizon: int, lags: Sequence[int] | None = None) -> None:
    """Chequeo defensivo para no usar información futura como predictor.

    En este proyecto, los lags permitidos son ``0, 1, 2, ...`` y el horizonte
    debe ser no negativo. Un lag negativo implicaría mirar hacia el futuro.
    """
    if int(horizon) < 0:
        raise ValidationError(f"forecast_horizon inválido: {horizon}. Debe ser >= 0.")
    if lags is not None:
        bad = [lag for lag in lags if int(lag) < 0]
        if bad:
            raise ValidationError(f"Lags futuros/no permitidos detectados: {bad}.")


def check_lengths_match(obs: Iterable, sim: Iterable) -> None:
    if len(_as_series(obs, "obs")) != len(_as_series(sim, "sim")):
        raise ValidationError(
            f"Longitudes incompatibles: obs={len(_as_series(obs, 'obs'))}, "
            f"sim={len(_as_series(sim, 'sim'))}."
        )


def check_non_negative_q(values: Iterable) -> None:
    arr = _finite_numeric(values, "Qsim")
    if np.isnan(arr).any():
        raise ValidationError("Qsim contiene NaN/no numéricos.")
    if np.any(arr < -1e-9):
        first = int(np.where(arr < -1e-9)[0][0])
        raise ValidationError(f"Qsim negativo detectado en fila {first}: {arr[first]}.")


def check_quantiles_ordered(lower: Iterable, upper: Iterable) -> None:
    lo = _finite_numeric(lower, "lower")
    up = _finite_numeric(upper, "upper")
    if len(lo) != len(up):
        raise ValidationError(f"Bandas con longitudes distintas: lower={len(lo)}, upper={len(up)}.")
    if np.isnan(lo).any() or np.isnan(up).any():
        raise ValidationError("Las bandas de incertidumbre contienen NaN/no numéricos.")
    bad = lo > up
    if np.any(bad):
        first = int(np.where(bad)[0][0])
        raise ValidationError(
            f"Cuantiles desordenados en fila {first}: lower={lo[first]}, upper={up[first]}."
        )

