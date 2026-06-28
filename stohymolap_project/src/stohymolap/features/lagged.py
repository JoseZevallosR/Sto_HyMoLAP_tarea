"""Generacion de variables rezagadas (lags) sin fuga de informacion.

Para un horizonte de pronostico ``horizon`` (p.ej. t+1), el objetivo en la
fila ``i`` es ``y[i + horizon]`` y los predictores son valores en ``i``,
``i-1``, ``i-2`` ... Asi nunca se usa informacion futura como predictor.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

_log = get_logger("features.lagged")


def make_lagged_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    horizon: int = 1,
) -> pd.DataFrame:
    """Construye un DataFrame de features rezagadas.

    Parameters
    ----------
    df:
        DataFrame fuente (debe contener ``columns``).
    columns:
        Columnas a rezagar.
    lags:
        Lista de rezagos, p.ej. ``[0, 1, 2]`` (0 = valor actual en t).
    horizon:
        Horizonte de pronostico. El indice se conserva para alinear el target
        externamente con ``align_target``.

    Returns
    -------
    pandas.DataFrame
        Features rezagadas, con el mismo indice que ``df``. Las primeras filas
        contendran NaN por los rezagos y deben recortarse antes de entrenar.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas ausentes para lags: {missing}")

    feats = {}
    for col in columns:
        s = df[col].astype(float)
        for lag in lags:
            feats[f"{col}_lag{lag}"] = s.shift(lag)
    out = pd.DataFrame(feats, index=df.index)
    _log.debug("Features rezagadas: %d columnas, horizon=%d", out.shape[1], horizon)
    return out


def align_target(
    target: pd.Series, horizon: int = 1
) -> pd.Series:
    """Desplaza el objetivo ``horizon`` pasos hacia atras (y[i+horizon])."""
    return target.shift(-horizon)


def build_supervised(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    lags: Sequence[int],
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Arma (X, y) supervisado tabular con lags y horizonte.

    Returns
    -------
    X, y, valid_index
        ``X`` features, ``y`` objetivo alineado y ``valid_index`` el indice
        booleano (sobre ``df``) de las filas sin NaN utilizables.
    """
    X = make_lagged_features(df, feature_columns, lags, horizon=horizon)
    y = align_target(df[target_column].astype(float), horizon=horizon)

    valid = X.notna().all(axis=1) & y.notna()
    X_valid = X.loc[valid]
    y_valid = y.loc[valid]
    return X_valid, y_valid, valid.to_numpy()
