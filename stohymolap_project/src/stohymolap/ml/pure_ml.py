"""Modelo ML puro (E3): alias semantico sobre XGBoostModel.

E3 usa exclusivamente forzantes meteorologicas observadas (P, PET, Tmin, Tmax)
y sus rezagos, sin ninguna senal del modelo fisico. La maquinaria es la misma
que la del backend de gradient boosting, por lo que aqui solo se re-exporta con
un nombre que deja explicito el rol en la matriz experimental.
"""
from .xgboost_model import XGBoostModel


class PureMLModel(XGBoostModel):
    """Modelo data-driven puro (sin entradas del modelo fisico)."""
    pass
