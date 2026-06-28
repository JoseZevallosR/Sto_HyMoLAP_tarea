"""Registro de experimentos StoHyMoLAP.

Un unico ``ExperimentRunner`` cubre la matriz Fase 3.1 seleccionando ramas
internas segun la configuracion. La matriz canonica evita duplicados y ordena
las ablaciones para aislar tres efectos:

* memoria hidrologica/baseflow: E0->E1 y E2->E3;
* ruido Levy: E1->E3;
* hibridacion ML: E3/E4->E5..E8.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..utils.logging import get_logger
from .runner import ExperimentRunner

_log = get_logger("experiments.registry")

# Orden canonico Fase 3.1: 7 experimentos minimos + 2 extensiones secuenciales.
CANONICAL_ORDER: List[str] = [
    "E0_RAMIS_DET_NOBF",
    "E1_RAMIS_DET_BF",
    "E2_RAMIS_LEVY_NOBF",
    "E3_RAMIS_LEVY_BF",
    "E4_ML_PURE_XGB",
    "E5_HYB_XGB_MEAN",
    "E6_HYB_XGB_QUANTILES",
    "E7_HYB_GRU_MEAN",
    "E8_HYB_GRU_QUANTILES",
]

# IDs heredados de la matriz preliminar. No deben ejecutarse en Fase 3.1.
DEPRECATED_EXPERIMENT_IDS: List[str] = [
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
    """Devuelve ids de experimento respetando el orden canonico.

    Por defecto solo lista experimentos no deprecados. Si el usuario declara ids
    adicionales propios en la config, se agregan al final.
    """
    declared = list(cfg.get("experiments", {}).keys())
    declared_set = set(declared)
    deprecated_present = [eid for eid in DEPRECATED_EXPERIMENT_IDS if eid in declared_set]
    if deprecated_present:
        _log.warning(
            "La configuracion contiene ids heredados/deprecados que no forman parte "
            "de la matriz Fase 3.1: %s",
            deprecated_present,
        )
    ordered = [eid for eid in CANONICAL_ORDER if eid in declared_set]
    extra = [
        eid for eid in declared
        if eid not in CANONICAL_ORDER and eid not in DEPRECATED_EXPERIMENT_IDS
    ]
    return ordered + extra


def run_experiment(cfg: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
    """Instancia el runner y ejecuta un experimento por id."""
    _log.info("Lanzando experimento %s", experiment_id)
    runner = ExperimentRunner(cfg, experiment_id)
    return runner.run()
