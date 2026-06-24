"""
Estimación de evapotranspiración de referencia (ET0) a partir de
temperatura mínima y máxima mediante el método de Hargreaves-Samani (1985).

ET0 = 0.0023 * Ra * (Tmedia + 17.8) * (Tmax - Tmin)^0.5

donde Ra es la radiación extraterrestre expresada en mm/día (equivalente de
evaporación). Ra se calcula con el procedimiento estándar de la FAO-56
(Allen et al., 1998), que depende únicamente de la latitud y del día del año.
"""

import numpy as np
import pandas as pd


def extraterrestrial_radiation(day_of_year, latitude_deg):
    """
    Radiación extraterrestre Ra (FAO-56, Allen et al. 1998, Ec. 21).

    Parameters
    ----------
    day_of_year : array-like
        Día juliano del año (1-366).
    latitude_deg : float
        Latitud en grados decimales. NEGATIVA en el hemisferio sur.

    Returns
    -------
    np.ndarray
        Ra en mm/día (equivalente de evaporación).
    """
    J = np.asarray(day_of_year, dtype=float)

    phi = np.radians(latitude_deg)          # latitud en radianes
    gsc = 0.0820                            # constante solar [MJ m-2 min-1]

    # Distancia relativa inversa Tierra-Sol (Ec. 23)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * J / 365.0)

    # Declinación solar (Ec. 24)
    decl = 0.409 * np.sin(2.0 * np.pi * J / 365.0 - 1.39)

    # Ángulo horario de la puesta de sol (Ec. 25).
    # El clip evita NaN en combinaciones extremas de latitud/declinación.
    ws_arg = np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(ws_arg)

    # Ra en MJ m-2 día-1 (Ec. 21)
    ra_mj = (24.0 * 60.0 / np.pi) * gsc * dr * (
        ws * np.sin(phi) * np.sin(decl) +
        np.cos(phi) * np.cos(decl) * np.sin(ws)
    )

    # Conversión MJ m-2 día-1 -> mm/día (factor 0.408)
    return 0.408 * ra_mj


def hargreaves_samani_pet(tmin, tmax, dates, latitude_deg):
    """
    Evapotranspiración de referencia ET0 por Hargreaves-Samani (1985).

    Parameters
    ----------
    tmin, tmax : array-like
        Temperatura mínima y máxima diaria en °C.
    dates : array-like de datetime (o convertible con pd.to_datetime)
        Fechas correspondientes a cada registro.
    latitude_deg : float
        Latitud en grados decimales. NEGATIVA en el hemisferio sur.

    Returns
    -------
    np.ndarray
        ET0 en mm/día (valores no negativos).
    """
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)

    doy = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear.to_numpy()

    ra = extraterrestrial_radiation(doy, latitude_deg)

    tmean = (tmax + tmin) / 2.0
    trange = np.clip(tmax - tmin, 0.0, None)   # evita raíces negativas por ruido

    et0 = 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange)

    return np.clip(et0, 0.0, None)
