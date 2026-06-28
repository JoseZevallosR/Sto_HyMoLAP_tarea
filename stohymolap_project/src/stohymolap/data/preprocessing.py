"""Preprocesamiento: PET, precipitación efectiva, split temporal y chequeos.

Reúne la lógica que en el código original estaba dispersa entre ``data_io.py``
y ``main.py``: cálculo de PET, ``Peff = max(P - PET, 0)``, división
calibración/validación estrictamente temporal y validaciones de integridad.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from ..hydro.pet import hargreaves_samani_pet
from ..utils.logging import get_logger

_log = get_logger("data.preprocessing")


def add_pet_and_peff(df: pd.DataFrame, latitude_deg: float) -> pd.DataFrame:
    """Añade columnas ``PET`` y ``Peff`` al DataFrame.

    ``Peff_t = max(P_t - PET_t, 0)``.
    """
    df = df.copy()
    df["PET"] = hargreaves_samani_pet(
        df["Tmin"].to_numpy(dtype=float),
        df["Tmax"].to_numpy(dtype=float),
        df["date"].to_numpy(),
        latitude_deg,
    )
    df["Peff"] = np.clip(df["P"].to_numpy(dtype=float) - df["PET"].to_numpy(dtype=float), 0.0, None)
    _log.info(
        "PET (mm/dia): min=%.3f media=%.3f max=%.3f",
        df["PET"].min(), df["PET"].mean(), df["PET"].max(),
    )
    return df


def drop_invalid_rows(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] | None = None,
    require_qobs: bool = True,
    label: str = "data",
) -> pd.DataFrame:
    """Elimina filas con NaN en columnas críticas.

    Por compatibilidad, ``require_qobs=True`` mantiene el comportamiento antiguo.
    Para Fase 1 se usa ``require_qobs=False`` antes de simular RAMIS, de modo
    que el calendario y los forzantes diarios se conserven aunque ``Qobs`` tenga
    huecos. Las métricas ya enmascaran observaciones faltantes.
    """
    required = list(required_columns or ["date", "Qobs", "P", "PET"])
    if not require_qobs:
        required = [c for c in required if c != "Qobs"]

    before = len(df)
    mask = df["date"].notna()
    for col in required:
        if col == "date":
            continue
        mask &= np.isfinite(df[col])
    df = df.loc[mask].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        _log.info("[%s] Se descartaron %d filas con NaN/fecha invalida.", label, dropped)
    if len(df) < 20:
        raise ValueError(f"Muy pocos datos validos para ejecutar el modelo en {label}.")
    return df


def trim_to_first_valid_qobs(df: pd.DataFrame, *, label: str = "data") -> pd.DataFrame:
    """Recorta la serie hasta la primera observación real disponible de Qobs.

    RAMIS necesita un caudal inicial ``q0``. Si la serie comienza con ``Qobs``
    faltante, se eliminan solo esos días iniciales sin observación, pero se
    mantienen los días posteriores aunque tengan huecos observacionales.
    """
    valid = np.isfinite(pd.to_numeric(df["Qobs"], errors="coerce"))
    if not valid.any():
        raise ValueError(f"No existe ningún Qobs finito para inicializar {label}.")
    first = int(np.argmax(valid.to_numpy()))
    if first > 0:
        _log.warning(
            "[%s] Se recortan %d días iniciales sin Qobs. Primera fecha con Qobs=%s.",
            label,
            first,
            pd.to_datetime(df.loc[first, "date"]).date(),
        )
    return df.iloc[first:].reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame, train_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame en train/validation de forma estrictamente temporal.

    No se baraja: las primeras ``train_fraction`` filas (ya ordenadas por fecha)
    van a calibración y el resto a validación. Esto evita fuga de información
    temporal.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction debe estar en (0, 1).")
    n_total = len(df)
    n_train = int(round(n_total * train_fraction))
    n_train = max(2, min(n_train, n_total - 1))
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:].reset_index(drop=True)

    if train["date"].max() >= val["date"].min():
        raise AssertionError(
            "El split temporal mezcla fechas: max(train) >= min(val)."
        )
    _log.info(
        "Split temporal: %d calibracion (%s a %s), %d validacion (%s a %s).",
        len(train), train["date"].min().date(), train["date"].max().date(),
        len(val), val["date"].min().date(), val["date"].max().date(),
    )
    return train, val
