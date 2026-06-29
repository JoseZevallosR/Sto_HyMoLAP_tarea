"""Modelo XGBoost (con fallback a sklearn) para experimentos E3-E5.

Si ``xgboost`` esta instalado se usa ``XGBRegressor``; en caso contrario se
recurre a ``HistGradientBoostingRegressor`` de sklearn, que es un gradient
boosting equivalente en espiritu. La busqueda de hiperparametros usa
``RandomizedSearchCV`` con ``TimeSeriesSplit`` para respetar el orden temporal.

El backend efectivamente usado queda registrado en ``model.backend`` y se
guarda junto al artefacto del modelo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger

_log = get_logger("ml.xgboost_model")

try:  # pragma: no cover - depende del entorno
    from xgboost import XGBRegressor  # type: ignore

    _HAS_XGB = True
except Exception:  # noqa: BLE001
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HAS_XGB = False


class XGBoostModel:
    """Envoltorio uniforme de un regresor tipo gradient boosting."""

    def __init__(
        self,
        scale_features: bool = False,
        search: bool = True,
        n_iter_search: int = 15,
        n_splits: int = 4,
        random_state: int = 42,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.scale_features = scale_features
        self.search = search
        self.n_iter_search = n_iter_search
        self.n_splits = n_splits
        self.random_state = random_state
        self.user_params = params or {}
        self.backend = "xgboost" if _HAS_XGB else "sklearn_hgb"
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.best_params_: Dict[str, Any] = {}

    # --- construccion del estimador base --------------------------------
    def _base_estimator(self):
        if _HAS_XGB:
            base = dict(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, objective="reg:squarederror",
            )
            base.update(self.user_params)
            return XGBRegressor(**base)
        base = dict(
            max_depth=None, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=self.random_state,
        )
        base.update(self.user_params)
        return HistGradientBoostingRegressor(**base)

    def _search_space(self) -> Dict[str, list]:
        if _HAS_XGB:
            return {
                "n_estimators": [200, 400, 600, 800],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
        return {
            "max_iter": [200, 400, 600],
            "learning_rate": [0.02, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1.0, 5.0],
        }

    # --- API ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.scale_features:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)

        estimator = self._base_estimator()
        if self.search and len(X) > (self.n_splits + 1):
            cv = TimeSeriesSplit(n_splits=self.n_splits)
            rs = RandomizedSearchCV(
                estimator,
                self._search_space(),
                n_iter=self.n_iter_search,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=self.random_state,
                n_jobs=1,
            )
            rs.fit(X, y)
            self.model = rs.best_estimator_
            self.best_params_ = rs.best_params_
            _log.info("XGB(%s) mejores params: %s", self.backend, self.best_params_)
        else:
            estimator.fit(X, y)
            self.model = estimator
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(self.model.predict(X), dtype=float)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"backend": self.backend, "scaler": self.scaler,
             "model": self.model, "best_params": self.best_params_},
            path,
        )
        _log.info("Modelo XGB guardado en %s (backend=%s)", path, self.backend)
