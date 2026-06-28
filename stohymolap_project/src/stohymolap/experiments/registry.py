"""Registro de experimentos.

Un unico ``ExperimentRunner`` cubre los 9 experimentos seleccionando ramas
internas segun la configuracion. El registro expone:

* ``list_experiments``: ids declarados en la configuracion.
* ``run_experiment``: ejecuta uno por id.

Mantener este punto unico de entrada facilita orquestar corridas individuales
o por lotes sin duplicar logica.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..utils.logging import get_logger
from .runner import ExperimentRunner

_log = get_logger("experiments.registry")

# Orden canonico de la matriz (E1-E7 + ablaciones).
CANONICAL_ORDER: List[str] = [
    "E1_RAMIS_DET_BF",
    "E2_RAMIS_LEVY_BF",
    "E3_ML_PURE",
    "E4_RAMIS_XGB_MEAN",
    "E5_RAMIS_XGB_QUANTILE",
    "E6_RAMIS_GRU_MEAN",
    "E7_RAMIS_GRU_QUANTILE",
    "A1_RAMIS_LEVY_NOBF",
    "A2_RAMIS_LEVY_BF",
]


def list_experiments(cfg: Dict[str, Any]) -> List[str]:
    """Devuelve los ids de experimento presentes en la configuracion.

    Respeta el orden canonico cuando es posible y agrega al final cualquier
    experimento adicional declarado por el usuario.
    """
    declared = list(cfg.get("experiments", {}).keys())
    ordered = [e for e in CANONICAL_ORDER if e in declared]
    extra = [e for e in declared if e not in CANONICAL_ORDER]
    return ordered + extra


def run_experiment(cfg: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
    """Instancia el runner y ejecuta un experimento por id."""
    _log.info("Lanzando experimento %s", experiment_id)
    runner = ExperimentRunner(cfg, experiment_id)
    return runner.run()
