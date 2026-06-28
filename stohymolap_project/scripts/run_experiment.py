#!/usr/bin/env python
"""Ejecuta un experimento individual de la matriz StoHyMoLAP / RAMIS.

Uso:
    python scripts/run_experiment.py --config configs/experiments.yaml \
           --experiment E1_RAMIS_DET_BF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script sin instalar el paquete.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stohymolap.experiments.registry import run_experiment  # noqa: E402
from stohymolap.utils.config import deep_merge, load_config  # noqa: E402
from stohymolap.utils.logging import get_logger, setup_logging  # noqa: E402

_log = get_logger("scripts.run_experiment")


def _load_merged_config(config_path: Path) -> dict:
    """Carga el config; si no trae bloque ``global`` intenta heredar de base.yaml."""
    cfg = load_config(config_path)
    if "global" not in cfg:
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            base = load_config(base_path)
            cfg = deep_merge(base, cfg)
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ejecuta un experimento RAMIS.")
    parser.add_argument("--config", required=True, type=Path, help="Ruta al YAML/JSON.")
    parser.add_argument("--experiment", required=True, help="ID del experimento.")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = _load_merged_config(args.config)
    result = run_experiment(cfg, args.experiment)

    val = result.get("metrics_validation", {})
    _log.info(
        "Resultado %s -> NSE=%.3f KGE=%.3f RMSE=%.3f (backend=%s)",
        args.experiment,
        val.get("NSE", float("nan")),
        val.get("KGE", float("nan")),
        val.get("RMSE", float("nan")),
        result.get("backend", "physical"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
