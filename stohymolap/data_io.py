import os
import numpy as np
import pandas as pd
from config import PaperHyMoLAPConfig


def load_hydrological_data(config: PaperHyMoLAPConfig):
    """
    Lee datos desde CSV o Excel.

    Soporta dos formatos:

    1. Formato con nombres:
       Qobs, Precipitation, PET

    2. Formato tipo paper:
       primera columna = discharge
       segunda columna = precipitation
       tercera columna = PET
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