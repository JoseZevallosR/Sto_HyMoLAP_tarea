#!/usr/bin/env python
"""Ejecuta un único experimento StoHyMoLAP.

Ejemplos:
    python scripts/run_experiment.py --config configs/experiments.yaml --experiment E1_RAMIS_DET_BF
    python scripts/run_experiment.py --config configs/experiments.yaml --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stohymolap.experiments.registry import list_experiments, run_experiment  # noqa: E402
from stohymolap.utils.config import deep_merge, load_config  # noqa: E402


def _load_merged_config(config_path: Path) -> dict:
    cfg = load_config(config_path)
    if "global" not in cfg:
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ejecuta un experimento StoHyMoLAP.")
    parser.add_argument("--config", required=True, type=Path, help="Archivo YAML de configuración.")
    parser.add_argument("--experiment", help="ID del experimento a ejecutar.")
    parser.add_argument("--list", action="store_true", help="Lista experimentos disponibles y termina.")
    args = parser.parse_args(argv)

    cfg = _load_merged_config(args.config)
    available = list_experiments(cfg)

    if args.list:
        for eid in available:
            print(eid)
        return 0

    if not args.experiment:
        parser.error("Debes indicar --experiment o usar --list.")
    if args.experiment not in available:
        parser.error(f"Experimento no encontrado: {args.experiment}. Disponibles: {available}")

    result = run_experiment(cfg, args.experiment)
    metrics = result.get("metrics_validation", {}) if isinstance(result, dict) else {}
    print(f"Experimento completado: {args.experiment}")
    if metrics:
        print(
            "Validación: "
            f"NSE={metrics.get('NSE', float('nan')):.4f}, "
            f"KGE={metrics.get('KGE', float('nan')):.4f}, "
            f"PBIAS={metrics.get('PBIAS', float('nan')):.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

