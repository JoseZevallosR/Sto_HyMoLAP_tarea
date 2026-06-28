"""Comprobaciones de integridad y anti-fuga (anti-leakage).

Estas funciones se invocan desde el ``ExperimentRunner`` en puntos clave del
pipeline para garantizar que los datos y las predicciones cumplen invariantes
científicas básicas:

* no hay fechas duplicadas,
* la serie diaria no tiene saltos cuando se exige continuidad,
* P, Qobs y PET no son negativas,
* ``Tmax >= Tmin``,
* las columnas requeridas existen,
* el corte temporal train/val no mezcla fechas,
* los rezagos no usan información futura,
* observaciones y simulaciones tienen la misma longitud,
* el caudal final no es negativo,
* los cuantiles están ordenados.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Se lanza cuando una comprobación de integridad del pipeline falla."""


def _finite_series(values: Iterable, name: str) -> pd.Series:
    s = pd.Series(values, name=name)
    return pd.to_numeric(s, errors="coerce")


def check_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Verifica que el DataFrame contenga todas las columnas requeridas."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Faltan columnas requeridas: {missing}. Disponibles: {list(df.columns)}"
        )


def check_no_duplicate_dates(dates: Iterable) -> None:
    """Verifica que no existan fechas duplicadas en la serie temporal."""
    s = pd.Series(pd.to_datetime(list(dates), errors="coerce"))
    dup = s[s.duplicated() & s.notna()].unique()
    if len(dup) > 0:
        raise ValidationError(
            f"Se encontraron {len(dup)} fechas duplicadas, p. ej. {dup[:3]}"
        )


def check_daily_continuity(dates: Iterable, *, allow_empty: bool = False) -> None:
    """Verifica frecuencia diaria continua y ordenada.

    Se usa sobre el calendario hidrometeorológico completo. No debe aplicarse a
    subconjuntos ya filtrados solo por ``Qobs`` porque allí pueden existir huecos
    observacionales legítimos que se evalúan con máscara.
    """
    s = pd.Series(pd.to_datetime(list(dates), errors="coerce")).dropna().sort_values()
    if s.empty:
        if allow_empty:
            return
        raise ValidationError("La serie de fechas está vacía o solo contiene NaT.")
    gaps = s.diff().dropna().dt.days
    bad = gaps[gaps != 1]
    if not bad.empty:
        pos = int(bad.index[0])
        raise ValidationError(
            "La serie diaria no es continua: "
            f"salto de {int(bad.iloc[0])} días cerca de {s.loc[pos].date()}."
        )


def check_non_negative(values: Iterable, name: str, tol: float = 1e-9) -> None:
    """Verifica que una variable física no tenga valores negativos."""
    s = _finite_series(values, name)
    finite = s[np.isfinite(s)]
    if (finite < -tol).any():
        n = int((finite < -tol).sum())
        raise ValidationError(f"{name} negativa en {n} registros (min={finite.min():.4f}).")


def check_pet_non_negative(pet: np.ndarray, tol: float = 1e-9) -> None:
    """Compatibilidad con llamadas antiguas: PET no negativa."""
    check_non_negative(pet, "PET", tol=tol)


def check_tmax_ge_tmin(tmin: Iterable, tmax: Iterable, tol: float = 1e-9) -> None:
    """Verifica que ``Tmax`` sea mayor o igual que ``Tmin``."""
    a = _finite_series(tmin, "Tmin")
    b = _finite_series(tmax, "Tmax")
    mask = np.isfinite(a) & np.isfinite(b)
    bad = (b[mask] + tol) < a[mask]
    if bad.any():
        n = int(bad.sum())
        raise ValidationError(f"Tmax < Tmin en {n} registros.")


def check_missing_values(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    allow: Mapping[str, bool] | None = None,
) -> None:
    """Verifica NaN en columnas críticas.

    ``allow={"Qobs": True}`` permite huecos observacionales, pero mantiene
    estrictas variables de forzante como ``P``, ``Tmin``, ``Tmax`` y ``PET``.
    """
    allow = dict(allow or {})
    failures = []
    for col in columns:
        if col not in df.columns:
            failures.append(f"{col}: columna ausente")
            continue
        n = int(df[col].isna().sum())
        if n > 0 and not allow.get(col, False):
            failures.append(f"{col}: {n} NaN")
    if failures:
        raise ValidationError("Valores faltantes no permitidos: " + "; ".join(failures))


def qobs_source_summary(df: pd.DataFrame) -> dict[str, int]:
    """Resume el origen de Qobs: observed/missing/filled_from_qsim."""
    if "Qobs_source" not in df.columns:
        return {}
    vc = df["Qobs_source"].astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def check_no_filled_qobs(df: pd.DataFrame) -> None:
    """Falla si existen registros de Qobs rellenados desde Qsim."""
    if "Qobs_is_filled_from_qsim" not in df.columns:
        return
    n = int(pd.Series(df["Qobs_is_filled_from_qsim"]).fillna(False).sum())
    if n > 0:
        raise ValidationError(
            f"Qobs contiene {n} registros rellenados con Qsim. "
            "Para métricas principales usa observaciones reales o reporta métricas separadas."
        )


def validate_hydro_dataframe(
    df: pd.DataFrame,
    *,
    require_daily: bool = True,
    allow_missing_qobs: bool = True,
    allow_filled_qobs: bool = False,
) -> None:
    """Valida la tabla hidrometeorológica canónica antes del modelamiento."""
    check_required_columns(df, ["date", "Qobs", "P", "Tmin", "Tmax"])
    check_no_duplicate_dates(df["date"])
    if require_daily:
        check_daily_continuity(df["date"])
    check_missing_values(
        df,
        ["date", "Qobs", "P", "Tmin", "Tmax"],
        allow={"Qobs": allow_missing_qobs},
    )
    check_non_negative(df["P"], "P")
    check_non_negative(df["Qobs"], "Qobs")
    check_tmax_ge_tmin(df["Tmin"], df["Tmax"])
    if not allow_filled_qobs:
        check_no_filled_qobs(df)


def check_temporal_split(train_dates: Iterable, val_dates: Iterable) -> None:
    """Anti-leakage: el corte temporal no debe mezclar fechas entre train y val."""
    tr = pd.to_datetime(list(train_dates), errors="coerce")
    va = pd.to_datetime(list(val_dates), errors="coerce")
    if len(tr) == 0 or len(va) == 0:
        return
    overlap = set(pd.Series(tr)).intersection(set(pd.Series(va)))
    if overlap:
        raise ValidationError(
            f"Fuga temporal: {len(overlap)} fechas aparecen en train y val."
        )
    if tr.max() >= va.min():
        raise ValidationError(
            "Fuga temporal: la fecha máxima de train "
            f"({tr.max().date()}) no es anterior a la mínima de val "
            f"({va.min().date()})."
        )


def check_lags_no_future(horizon: int) -> None:
    """Verifica que el horizonte de rezago no use información del futuro."""
    if horizon < 0:
        raise ValidationError(
            f"Los rezagos no pueden usar el futuro: horizonte={horizon} < 0."
        )


def check_lengths_match(obs: np.ndarray, sim: np.ndarray) -> None:
    """Verifica que observaciones y simulaciones tengan la misma longitud."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    if obs.shape[0] != sim.shape[0]:
        raise ValidationError(
            f"Longitudes distintas: obs={obs.shape[0]} vs sim={sim.shape[0]}."
        )


def check_non_negative_q(q: np.ndarray, tol: float = 1e-6) -> None:
    """Verifica que el caudal simulado final no sea negativo."""
    q = np.asarray(q, dtype=float)
    finite = q[np.isfinite(q)]
    if finite.size and np.any(finite < -tol):
        n = int(np.sum(finite < -tol))
        raise ValidationError(
            f"Caudal negativo en {n} registros (min={np.nanmin(finite):.4f})."
        )


def check_quantiles_ordered(q_low: np.ndarray, q_high: np.ndarray, tol: float = 1e-6) -> None:
    """Verifica el orden de los cuantiles: q_low <= q_high."""
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    mask = np.isfinite(q_low) & np.isfinite(q_high)
    if np.any(q_low[mask] - q_high[mask] > tol):
        n = int(np.sum(q_low[mask] - q_high[mask] > tol))
        raise ValidationError(
            f"Cuantiles desordenados en {n} registros (q_low > q_high)."
        )
