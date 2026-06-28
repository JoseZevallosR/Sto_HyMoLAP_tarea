#!/usr/bin/env python
"""Ejecuta TODA la matriz de experimentos y construye la comparacion final.

Uso:
    python scripts/run_all_experiments.py --config configs/experiments.yaml
    python scripts/run_all_experiments.py --config configs/experiments.yaml \
           --only E1_RAMIS_DET_BF E2_RAMIS_LEVY_BF
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stohymolap.diagnostics.leakage_audit import audit_leakage  # noqa: E402
from stohymolap.experiments.comparison import run_comparison  # noqa: E402
from stohymolap.experiments.registry import list_experiments, run_experiment  # noqa: E402
from stohymolap.utils.config import deep_merge, load_config  # noqa: E402
from stohymolap.utils.logging import get_logger, setup_logging  # noqa: E402

_log = get_logger("scripts.run_all")


def _load_merged_config(config_path: Path) -> dict:
    cfg = load_config(config_path)
    if "global" not in cfg:
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def _setup_run_all_logging(cfg: dict) -> Path:
    """Activa un log global separado de los logs por experimento."""
    exp_root = Path(cfg.get("global", {}).get("output_root", "outputs/experiments"))
    comparison_root = exp_root.parent / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=comparison_root / "run_all.log")
    return comparison_root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ejecuta la matriz completa de experimentos.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Subconjunto de experimentos a correr (por id).")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="No abortar si un experimento falla.")
    args = parser.parse_args(argv)

    cfg = _load_merged_config(args.config)
    comparison_root = _setup_run_all_logging(cfg)

    experiment_ids = list_experiments(cfg)
    if args.only:
        experiment_ids = [e for e in experiment_ids if e in set(args.only)]

    _log.info("Experimentos a ejecutar: %s", experiment_ids)
    completed = []
    for eid in experiment_ids:
        try:
            _log.info("Preparando experimento %s", eid)
            run_experiment(cfg, eid)
            completed.append(eid)
            # El runner cambia el logger global al run.log del experimento.
            # Se restaura inmediatamente para que el siguiente lanzamiento no
            # contamine el log del experimento anterior.
            comparison_root = _setup_run_all_logging(cfg)
            _log.info("Experimento completado: %s", eid)
        except Exception:  # noqa: BLE001
            comparison_root = _setup_run_all_logging(cfg)
            _log.error("Fallo el experimento %s:\n%s", eid, traceback.format_exc())
            if not args.continue_on_error:
                raise

    exp_root = Path(cfg.get("global", {}).get("output_root", "outputs/experiments"))
    leaderboard = run_comparison(exp_root, comparison_root, completed)
    leakage = audit_leakage(cfg, exp_root, comparison_root, completed, write=True)

    if not leaderboard.empty:
        _log.info("Leaderboard:\n%s", leaderboard.to_string(index=False))
    _log.info("Auditoria leakage Fase 3.4A: %s", leakage["status"])
    if leakage["status"] == "FAIL":
        raise RuntimeError(
            "La auditoria anti-leakage Fase 3.4A fallo. "
            f"Revisa {leakage['report_path']}"
        )
    _log.info("Listo. Comparacion en %s", comparison_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
