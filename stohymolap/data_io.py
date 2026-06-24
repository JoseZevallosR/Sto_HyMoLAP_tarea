import os
import numpy as np
import pandas as pd

from config import PaperHyMoLAPConfig
from evapotranspiration import hargreaves_samani_pet


def _find_col(cols_lower, *candidates):
    """Devuelve el nombre real de la primera columna que coincida."""
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def load_ramis_hydro(config: PaperHyMoLAPConfig):
    """
    Lee el CSV de Ramis (guardado con df_final.to_csv(..., index=False)) y
    construye los vectores que necesita el modelo.

    Estructura esperada de columnas (no importa el orden):
        date, flow_obs, flow_sim, precipitation_mean, tmin_mean, tmax_mean

    - PET se estima con Hargreaves-Samani (tmin, tmax, fecha, latitud).
    - Si config.fill_obs_with_sim es True, los huecos de flow_obs se completan
      con flow_sim para conservar la continuidad temporal de la serie.

    Returns
    -------
    discharge, precip, pet, peff : np.ndarray
        Series alineadas y limpias. peff = max(precip - pet, 0).
    dates : np.ndarray (datetime64)
        Fechas alineadas con las series anteriores.
    """
    path = config.data_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext in [".csv", ".txt"]:
        # to_csv por defecto usa coma. Se intenta con el separador de config
        # y, si solo aparece una columna, se autodetecta.
        df = pd.read_csv(path, sep=config.csv_sep, encoding="utf-8")
        if df.shape[1] < 3:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    else:
        raise ValueError("Formato no soportado. Usa .csv, .txt, .xlsx o .xls")

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    date_col = _find_col(cols_lower, "date", "fecha")
    q_col = _find_col(cols_lower, "flow_obs", "qobs", "flow_observed", "q")
    sim_col = _find_col(cols_lower, "flow_sim", "qsim", "flow_simulated")
    p_col = _find_col(cols_lower, "precipitation_mean", "precipitation", "precip", "p")
    tmin_col = _find_col(cols_lower, "tmin_mean", "tmin", "t_min")
    tmax_col = _find_col(cols_lower, "tmax_mean", "tmax", "t_max")

    faltan = [
        nombre for nombre, col in [
            ("date", date_col), ("flow_obs", q_col),
            ("precipitation_mean", p_col),
            ("tmin_mean", tmin_col), ("tmax_mean", tmax_col),
        ] if col is None
    ]
    if faltan:
        raise ValueError(
            "Faltan columnas requeridas en el CSV: " + ", ".join(faltan) +
            f". Columnas encontradas: {list(df.columns)}"
        )

    dates = pd.to_datetime(df[date_col], errors="coerce")

    discharge = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)
    precip = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
    tmin = pd.to_numeric(df[tmin_col], errors="coerce").to_numpy(dtype=float)
    tmax = pd.to_numeric(df[tmax_col], errors="coerce").to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Completar flow_obs con flow_sim donde falte
    # ------------------------------------------------------------------
    if config.fill_obs_with_sim and sim_col is not None:
        flow_sim = pd.to_numeric(df[sim_col], errors="coerce").to_numpy(dtype=float)
        fill_mask = (~np.isfinite(discharge)) & np.isfinite(flow_sim)
        n_filled = int(fill_mask.sum())
        if n_filled > 0:
            discharge = discharge.copy()
            discharge[fill_mask] = flow_sim[fill_mask]
            print(
                f"[load_ramis_hydro] Se completaron {n_filled} valores de "
                f"flow_obs con flow_sim."
            )
    elif config.fill_obs_with_sim and sim_col is None:
        print(
            "[load_ramis_hydro] AVISO: fill_obs_with_sim=True pero no se "
            "encontró la columna flow_sim. No se rellenó nada."
        )

    # PET con Hargreaves-Samani (serie completa, antes de enmascarar).
    pet = hargreaves_samani_pet(tmin, tmax, dates, config.latitude_deg)

    # Se conservan solo los registros con todo válido.
    mask = (
        np.isfinite(discharge) &
        np.isfinite(precip) &
        np.isfinite(pet) &
        dates.notna().to_numpy()
    )

    n_drop = int((~mask).sum())
    if n_drop > 0:
        print(
            f"[load_ramis_hydro] Se descartaron {n_drop} registros con "
            f"NaN remanente (de {len(mask)} totales)."
        )

    discharge = discharge[mask]
    precip = precip[mask]
    pet = pet[mask]
    dates = dates.to_numpy()[mask]

    if len(discharge) < 20:
        raise ValueError("Muy pocos datos válidos para ejecutar el modelo.")

    peff = precip - pet
    peff[peff < 0.0] = 0.0

    print(
        f"[load_ramis_hydro] PET (mm/día): "
        f"min={pet.min():.3f}  media={pet.mean():.3f}  max={pet.max():.3f}"
    )

    return discharge, precip, pet, peff, dates


def load_hydrological_data(config: PaperHyMoLAPConfig):
    """
    Loader original (formato Qobs/Precipitation/PET o tres columnas numéricas).
    Se conserva para compatibilidad con datos que ya traen PET.
    """
    path = config.data_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path, sep=config.csv_sep, encoding="utf-8")

        if df.shape[1] < 3:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    else:
        raise ValueError("Formato no soportado. Usa .csv, .txt, .xlsx o .xls")

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    if (
        "qobs" in cols_lower and
        "precipitation" in cols_lower and
        "pet" in cols_lower
    ):
        q_col = cols_lower["qobs"]
        p_col = cols_lower["precipitation"]
        pet_col = cols_lower["pet"]

        discharge = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)
        precip = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
        pet = pd.to_numeric(df[pet_col], errors="coerce").to_numpy(dtype=float)

    else:
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = numeric_df.columns[numeric_df.notna().sum() > 0].tolist()

        if len(numeric_cols) < 3:
            raise ValueError(
                "No se encontraron al menos tres columnas numéricas. "
                "Se requieren Q, precipitación y PET."
            )

        discharge = numeric_df[numeric_cols[0]].to_numpy(dtype=float)
        precip = numeric_df[numeric_cols[1]].to_numpy(dtype=float)
        pet = numeric_df[numeric_cols[2]].to_numpy(dtype=float)

    mask = np.isfinite(discharge) & np.isfinite(precip) & np.isfinite(pet)

    discharge = discharge[mask]
    precip = precip[mask]
    pet = pet[mask]

    if len(discharge) < 20:
        raise ValueError("Muy pocos datos válidos para ejecutar el modelo.")

    peff = precip - pet
    peff[peff < 0.0] = 0.0

    return discharge, precip, pet, peff
