"""Modelo GRU secuencial para experimentos E6-E7.

Backends, en orden de preferencia:

1. TensorFlow/Keras  -> GRU real con early stopping y dropout.
2. PyTorch           -> GRU real con early stopping y dropout.
3. sklearn MLP       -> *fallback*. Aplana la secuencia (samples,
   sequence_length * n_features) y entrena un MLPRegressor. No es una GRU,
   pero permite que el pipeline E6/E7 corra de extremo a extremo en entornos
   sin frameworks de deep learning. Queda claramente etiquetado en
   ``model.backend``.

El escalado (MinMax) se ajusta SOLO con los datos de entrenamiento.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..utils.logging import get_logger

_log = get_logger("ml.gru_model")


def _detect_backend() -> str:
    try:  # pragma: no cover
        import tensorflow  # noqa: F401

        return "tensorflow"
    except Exception:
        pass
    try:  # pragma: no cover
        import torch  # noqa: F401

        return "torch"
    except Exception:
        pass
    return "sklearn_mlp"


class GRUModel:
    """Envoltorio uniforme de un modelo secuencial GRU (o fallback)."""

    def __init__(
        self,
        units: int = 32,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.backend = _detect_backend()
        self.x_scaler: Optional[MinMaxScaler] = None
        self.y_scaler: Optional[MinMaxScaler] = None
        self.model = None
        self._n_features: Optional[int] = None
        self._seq_len: Optional[int] = None

    # --- escalado (ajustado solo en train) ------------------------------
    def _fit_scalers(self, X3d: np.ndarray, y: np.ndarray):
        n, s, f = X3d.shape
        self._seq_len, self._n_features = s, f
        self.x_scaler = MinMaxScaler().fit(X3d.reshape(-1, f))
        self.y_scaler = MinMaxScaler().fit(y.reshape(-1, 1))

    def _scale_X(self, X3d: np.ndarray) -> np.ndarray:
        n, s, f = X3d.shape
        flat = self.x_scaler.transform(X3d.reshape(-1, f))
        return flat.reshape(n, s, f)

    # --- API ------------------------------------------------------------
    def fit(self, X3d: np.ndarray, y: np.ndarray) -> "GRUModel":
        X3d = np.asarray(X3d, dtype=float)
        y = np.asarray(y, dtype=float)
        self._fit_scalers(X3d, y)
        Xs = self._scale_X(X3d)
        ys = self.y_scaler.transform(y.reshape(-1, 1)).ravel()

        if self.backend == "tensorflow":
            self._fit_tensorflow(Xs, ys)
        elif self.backend == "torch":
            self._fit_torch(Xs, ys)
        else:
            self._fit_sklearn(Xs, ys)
        return self

    def predict(self, X3d: np.ndarray) -> np.ndarray:
        X3d = np.asarray(X3d, dtype=float)
        Xs = self._scale_X(X3d)
        if self.backend == "tensorflow":
            pred_s = self.model.predict(Xs, verbose=0).ravel()
        elif self.backend == "torch":
            import torch

            self.model.eval()
            with torch.no_grad():
                pred_s = self.model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()
        else:
            pred_s = self.model.predict(Xs.reshape(Xs.shape[0], -1))
        return self.y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    # --- backends -------------------------------------------------------
    def _fit_tensorflow(self, Xs, ys):  # pragma: no cover
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks

        tf.random.set_seed(self.random_state)
        n, s, f = Xs.shape
        model = models.Sequential([
            layers.Input(shape=(s, f)),
            layers.GRU(self.units, dropout=self.dropout),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss="mse")
        es = callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        model.fit(Xs, ys, validation_split=0.2, epochs=self.epochs,
                  batch_size=self.batch_size, callbacks=[es], verbose=0)
        self.model = model

    def _fit_torch(self, Xs, ys):  # pragma: no cover
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        n, s, f = Xs.shape

        class _Net(nn.Module):
            def __init__(self, f, units, dropout):
                super().__init__()
                self.gru = nn.GRU(f, units, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Sequential(nn.Linear(units, 16), nn.ReLU(), nn.Linear(16, 1))

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(self.drop(out[:, -1, :])).squeeze(-1)

        net = _Net(f, self.units, self.dropout)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(ys, dtype=torch.float32)

        n_val = max(1, int(0.2 * n))
        Xtr, ytr = Xt[:-n_val], yt[:-n_val]
        Xv, yv = Xt[-n_val:], yt[-n_val:]
        best, wait, best_state = np.inf, 0, None
        for _ in range(self.epochs):
            net.train()
            opt.zero_grad()
            loss = loss_fn(net(Xtr), ytr)
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                vloss = loss_fn(net(Xv), yv).item()
            if vloss < best:
                best, wait, best_state = vloss, 0, net.state_dict()
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net

    def _fit_sklearn(self, Xs, ys):
        from sklearn.neural_network import MLPRegressor

        _log.warning(
            "GRU: sin TensorFlow/PyTorch. Fallback MLP sobre secuencias aplanadas."
        )
        n, s, f = Xs.shape
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.units, 16),
            early_stopping=True, n_iter_no_change=self.patience,
            max_iter=max(200, self.epochs * 5), random_state=self.random_state,
        )
        self.model.fit(Xs.reshape(n, -1), ys)

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "tensorflow":  # pragma: no cover
            self.model.save(path.with_suffix(".keras"))
            joblib.dump({"backend": self.backend, "x_scaler": self.x_scaler,
                         "y_scaler": self.y_scaler}, path)
        elif self.backend == "torch":  # pragma: no cover
            import torch

            torch.save(self.model.state_dict(), path.with_suffix(".pt"))
            joblib.dump({"backend": self.backend, "x_scaler": self.x_scaler,
                         "y_scaler": self.y_scaler}, path)
        else:
            joblib.dump({"backend": self.backend, "model": self.model,
                         "x_scaler": self.x_scaler, "y_scaler": self.y_scaler}, path)
        _log.info("Modelo GRU guardado en %s (backend=%s)", path, self.backend)
