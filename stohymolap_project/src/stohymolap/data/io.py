"""Carga de datos hidrológicos (``ramis_hydro.csv``).

Refactorización del ``data_io.py`` original. Los nombres de columna se leen de
la configuración (bloque ``global``) y se devuelve un ``pandas.DataFrame``
ordenado por fecha.

Fase 0/1: ``Qobs`` ya no se rellena silenciosamente con ``Qsim`` por defecto.
Cuando se decide usar ``Qsim`` para completar vacíos, el origen queda marcado
en columnas explícitas para que las métricas y el paper puedan distinguir datos
observados reales de datos completados.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

_log = get_logger("data.io")


def _resolve_col(cols_lower: Dict[str, str], *candidates: str) -> str | None:
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def load_raw_table(data_path: str | Path, csv_sep: str = ",") -> pd.DataFrame:
    """Lee el archivo de datos crudo (csv/txt/xlsx) como DataFrame."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {path}")

    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif ext in {".csv", ".txt"}:
        df = pd.read_csv(path, sep=csv_sep, encoding="utf-8")
        if df.shape[1] < 3:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    else:
        raise ValueError("Formato no soportado. Usa .csv, .txt, .xlsx o .xls")
    return df


def load_hydro_dataframe(
    data_path: str | Path,
    *,
    date_col: str = "date",
    qobs_col: str = "flow_obs",
    p_col: str = "precipitation_mean",
    tmin_col: str = "tmin",
    tmax_col: str = "tmax",
    qsim_col: str = "flow_sim",
    fill_obs_with_sim: bool = False,
    csv_sep: str = ",",
) -> pd.DataFrame:
    """Carga y normaliza la tabla hidrológica a columnas canónicas.

    Devuelve un DataFrame con columnas estandarizadas:
    ``date, Qobs, P, Tmin, Tmax`` y, si existe, ``Qsim``.

    Columnas de trazabilidad agregadas:

    * ``Qobs_raw``: observación original antes de cualquier relleno.
    * ``Qobs_source``: ``observed``, ``missing`` o ``filled_from_qsim``.
    * ``Qobs_is_filled_from_qsim``: bandera booleana explícita.

    Por defecto ``fill_obs_with_sim=False`` para evitar que el caudal observado
    quede mezclado silenciosamente con una serie simulada.
    """
    df = load_raw_table(data_path, csv_sep=csv_sep)
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    real_date = _resolve_col(cols_lower, date_col.lower(), "date", "fecha")
    real_q = _resolve_col(cols_lower, qobs_col.lower(), "flow_obs", "qobs", "q")
    real_p = _resolve_col(
        cols_lower, p_col.lower(), "precipitation_mean", "precipitation", "precip", "p"
    )
    real_tmin = _resolve_col(cols_lower, tmin_col.lower(), "tmin_mean", "tmin", "t_min")
    real_tmax = _resolve_col(cols_lower, tmax_col.lower(), "tmax_mean", "tmax", "t_max")
    real_qsim = _resolve_col(cols_lower, qsim_col.lower(), "flow_sim", "qsim")

    missing = [
        name
        for name, col in [
            ("date", real_date),
            ("Qobs", real_q),
            ("P", real_p),
            ("Tmin", real_tmin),
            ("Tmax", real_tmax),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            "Faltan columnas requeridas en los datos: "
            + ", ".join(missing)
            + f". Columnas encontradas: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[real_date], errors="coerce")
    out["Qobs_raw"] = pd.to_numeric(df[real_q], errors="coerce")
    out["Qobs"] = out["Qobs_raw"]
    out["P"] = pd.to_numeric(df[real_p], errors="coerce")
    out["Tmin"] = pd.to_numeric(df[real_tmin], errors="coerce")
    out["Tmax"] = pd.to_numeric(df[real_tmax], errors="coerce")
    if real_qsim is not None:
        out["Qsim"] = pd.to_numeric(df[real_qsim], errors="coerce")

    out["Qobs_source"] = np.where(np.isfinite(out["Qobs_raw"]), "observed", "missing")
    out["Qobs_is_filled_from_qsim"] = False

    if fill_obs_with_sim and "Qsim" in out.columns:
        fill_mask = (~np.isfinite(out["Qobs"])) & np.isfinite(out["Qsim"])
        n_filled = int(fill_mask.sum())
        if n_filled > 0:
            out.loc[fill_mask, "Qobs"] = out.loc[fill_mask, "Qsim"]
            out.loc[fill_mask, "Qobs_source"] = "filled_from_qsim"
            out.loc[fill_mask, "Qobs_is_filled_from_qsim"] = True
            _log.warning(
                "Se completaron %d valores de Qobs con Qsim. "
                "Estos registros quedan marcados en Qobs_source.",
                n_filled,
            )

    out = out.sort_values("date").reset_index(drop=True)
    if len(out) > 0:
        _log.info(
            "Datos cargados: %d filas, %s a %s | Qobs faltantes=%d | Qobs rellenados=%d",
            len(out),
            out["date"].min().date(),
            out["date"].max().date(),
            int(out["Qobs"].isna().sum()),
            int(out["Qobs_is_filled_from_qsim"].sum()),
        )
    return out
