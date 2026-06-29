"""Evapotranspiracion de referencia (ET0) por Hargreaves-Samani (1985).

Migrado sin cambios funcionales desde ``evapotranspiration.py``.

ET0 = 0.0023 * Ra * (Tmedia + 17.8) * sqrt(Tmax - Tmin)

donde Ra es la radiacion extraterrestre (mm/dia) calculada con el
procedimiento FAO-56 (Allen et al., 1998), dependiente de latitud y dia juliano.
"""
import numpy as np
import pandas as pd


def extraterrestrial_radiation(day_of_year, latitude_deg):
    """Radiacion extraterrestre Ra (FAO-56, Allen et al. 1998, Ec. 21), mm/dia."""
    J = np.asarray(day_of_year, dtype=float)

    phi = np.radians(latitude_deg)
    gsc = 0.0820  # constante solar [MJ m-2 min-1]

    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * J / 365.0)
    decl = 0.409 * np.sin(2.0 * np.pi * J / 365.0 - 1.39)

    ws_arg = np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(ws_arg)

    ra_mj = (24.0 * 60.0 / np.pi) * gsc * dr * (
        ws * np.sin(phi) * np.sin(decl)
        + np.cos(phi) * np.cos(decl) * np.sin(ws)
    )
    return 0.408 * ra_mj


def hargreaves_samani_pet(tmin, tmax, dates, latitude_deg):
    """ET0 por Hargreaves-Samani (1985), mm/dia, no negativa."""
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)

    doy = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear.to_numpy()
    ra = extraterrestrial_radiation(doy, latitude_deg)

    tmean = (tmax + tmin) / 2.0
    trange = np.clip(tmax - tmin, 0.0, None)

    et0 = 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange)
    return np.clip(et0, 0.0, None)
