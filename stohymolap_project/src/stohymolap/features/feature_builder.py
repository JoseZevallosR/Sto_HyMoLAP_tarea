"""Feature builder: arma las matrices de entrada de cada experimento.

Centraliza la logica de construccion de features segun la fuente declarada en
``experiments[<id>].features.source``:

* ``observed_forcing``   : variables meteorologicas (P, PET, Tmin, Tmax) + lags  -> E3.
* ``ramis_deterministic``: Qdet_BF como predictor + lags                          -> E1 (eval).
* ``stochastic_ensemble``: features de incertidumbre (media/cuantiles/rangos)     -> E2, E4-E7.

Para los modelos secuenciales (GRU) tambien construye tensores 3D
(samples, sequence_length, n_features) respetando el orden temporal.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .lagged import make_lagged_features, align_target
from ..utils.logging import get_logger

_log = get_logger("features.feature_builder")


def select_feature_columns(variables: Sequence[str], available: Sequence[str]) -> List[str]:
    """Filtra las variables solicitadas a las realmente disponibles."""
    cols = [v for v in variables if v in available]
    missing = [v for v in variables if v not in available]
    if missing:
        _log.warning("Variables solicitadas no disponibles (se omiten): %s", missing)
    if not cols:
        raise ValueError(
            f"Ninguna de las variables {list(variables)} esta disponible "
            f"en {list(available)}."
        )
    return cols


def build_tabular(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    feature_columns: Sequence[str],
    lags: Sequence[int],
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Construye (X, y) tabular con lags y horizonte, sin filas con NaN."""
    X = make_lagged_features(feature_frame, feature_columns, lags, horizon=horizon)
    y = align_target(target, horizon=horizon)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid], valid.to_numpy()


def build_sequences(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    feature_columns: Sequence[str],
    sequence_length: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye tensores 3D para modelos secuenciales (GRU).

    Para cada posicion ``i`` valida, la muestra usa las filas
    ``[i-sequence_length+1, ..., i]`` como secuencia de entrada y
    ``target[i+horizon]`` como objetivo. No se mezclan periodos: las secuencias
    se forman dentro del propio subconjunto entregado.

    Returns
    -------
    X3d : (n_samples, sequence_length, n_features)
    y   : (n_samples,)
    end_index : indices ``i`` (sobre ``feature_frame``) del final de cada secuencia
    """
    feats = feature_frame[list(feature_columns)].to_numpy(dtype=float)
    y_full = target.to_numpy(dtype=float)
    n = len(feats)

    X_list, y_list, idx_list = [], [], []
    for i in range(sequence_length - 1, n - horizon):
        window = feats[i - sequence_length + 1: i + 1]
        y_val = y_full[i + horizon]
        if not np.all(np.isfinite(window)) or not np.isfinite(y_val):
            continue
        X_list.append(window)
        y_list.append(y_val)
        idx_list.append(i)

    if not X_list:
        raise ValueError("No se pudieron construir secuencias validas (revisa NaN/longitud).")

    X3d = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=float)
    end_index = np.asarray(idx_list, dtype=int)
    _log.debug("Secuencias: X=%s, y=%s", X3d.shape, y.shape)
    return X3d, y, end_index
