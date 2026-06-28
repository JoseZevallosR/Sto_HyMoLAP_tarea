"""Orquestacion de entrenamiento/prediccion ML segun el tipo de experimento.

Decide el modelo (XGBoost/PureML/GRU) y la forma de los datos (tabular vs
secuencial) a partir de la configuracion del experimento.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .gru_model import GRUModel
from .pure_ml import PureMLModel
from .xgboost_model import XGBoostModel
from ..utils.logging import get_logger

_log = get_logger("ml.train_predict")


def build_model(ml_model: str, model_type: str, seed: int, scale: bool = False):
    """Instancia el modelo ML adecuado."""
    if ml_model == "gru":
        return GRUModel(random_state=seed)
    if ml_model == "xgboost":
        if model_type == "machine_learning":
            return PureMLModel(random_state=seed, scale_features=scale)
        return XGBoostModel(random_state=seed, scale_features=scale)
    raise ValueError(f"ml_model no soportado: {ml_model}")


def fit_predict_tabular(model, X_train, y_train, X_val) -> Tuple[np.ndarray, np.ndarray, object]:
    """Entrena con train y predice train+val (modelos tabulares)."""
    model.fit(np.asarray(X_train), np.asarray(y_train))
    pred_train = model.predict(np.asarray(X_train))
    pred_val = model.predict(np.asarray(X_val)) if len(X_val) else np.array([])
    return pred_train, pred_val, model


def fit_predict_sequence(model, X3d_train, y_train, X3d_val) -> Tuple[np.ndarray, np.ndarray, object]:
    """Entrena y predice para modelos secuenciales (GRU)."""
    model.fit(X3d_train, y_train)
    pred_train = model.predict(X3d_train)
    pred_val = model.predict(X3d_val) if len(X3d_val) else np.array([])
    return pred_train, pred_val, model
