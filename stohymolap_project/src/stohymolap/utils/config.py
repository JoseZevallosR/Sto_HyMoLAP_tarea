"""Carga y manejo de configuración YAML/JSON.

La plataforma se controla íntegramente desde archivos de configuración. Este
módulo provee:

* ``load_config``: lee YAML o JSON a un diccionario.
* ``deep_merge``: fusiona configuraciones (base + overrides).
* ``ExperimentConfig``: vista cómoda de la configuración de un experimento que
  resuelve la herencia de bloques globales (``global``, ``pet``, ``calibration``,
  ``stochastic``, ``baseflow``) hacia cada experimento.
* ``to_legacy_config``: puente hacia el ``PaperHyMoLAPConfig`` original para
  reutilizar el código de calibración/validación heredado sin reescribirlo.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .logging import get_logger

_log = get_logger("utils.config")


def load_config(path: str | Path) -> Dict[str, Any]:
    """Lee un archivo YAML o JSON y devuelve un diccionario."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        cfg = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        raise ValueError(f"Formato de configuración no soportado: {path.suffix}")

    if not isinstance(cfg, dict):
        raise ValueError(f"La configuración en {path} no es un mapeo válido.")
    _log.debug("Configuración cargada desde %s", path)
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Fusión recursiva de diccionarios (``override`` tiene prioridad)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def dump_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Serializa una configuración a YAML (para ``config_used.yaml``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    _log.debug("Configuración guardada en %s", path)


@dataclass
class ExperimentConfig:
    """Vista resuelta de un experimento concreto.

    Combina el bloque ``experiments[<id>]`` con los bloques globales para
    exponer una interfaz uniforme al ``ExperimentRunner``.
    """

    experiment_id: str
    raw: Dict[str, Any]
    exp: Dict[str, Any] = field(default_factory=dict)

    # Bloques globales accesibles directamente.
    glob: Dict[str, Any] = field(default_factory=dict)
    pet: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    stochastic: Dict[str, Any] = field(default_factory=dict)
    baseflow: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], experiment_id: str) -> "ExperimentConfig":
        if "experiments" not in cfg or experiment_id not in cfg["experiments"]:
            available = list(cfg.get("experiments", {}).keys())
            raise KeyError(
                f"Experimento '{experiment_id}' no encontrado. "
                f"Disponibles: {available}"
            )
        return cls(
            experiment_id=experiment_id,
            raw=cfg,
            exp=cfg["experiments"][experiment_id],
            glob=cfg.get("global", {}),
            pet=cfg.get("pet", {}),
            calibration=cfg.get("calibration", {}),
            stochastic=cfg.get("stochastic", {}),
            baseflow=cfg.get("baseflow", {}),
            evaluation=cfg.get("evaluation", {}),
        )

    # --- accesos cómodos -------------------------------------------------
    @property
    def model_type(self) -> str:
        return self.exp.get("model_type", "physical")

    @property
    def description(self) -> str:
        return self.exp.get("description", self.experiment_id)

    @property
    def use_stochastic(self) -> bool:
        return bool(self.exp.get("stochastic", False))

    @property
    def use_baseflow(self) -> bool:
        # El experimento manda; si no se especifica, hereda del bloque global.
        if "baseflow" in self.exp:
            return bool(self.exp["baseflow"])
        return bool(self.baseflow.get("enabled", False))

    @property
    def ml_model(self) -> Optional[str]:
        return self.exp.get("ml_model")

    @property
    def sequence_length(self) -> int:
        return int(self.exp.get("sequence_length", 7))

    @property
    def feature_spec(self) -> Dict[str, Any]:
        return self.exp.get("features", {})

    @property
    def lags(self) -> List[int]:
        feats = self.feature_spec
        if "lags" in feats:
            return list(feats["lags"])
        return list(self.glob.get("lags", [0, 1, 2]))

    @property
    def seed(self) -> int:
        return int(self.glob.get("seed", 42))

    @property
    def output_root(self) -> Path:
        return Path(self.glob.get("output_root", "outputs/experiments"))

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.experiment_id
