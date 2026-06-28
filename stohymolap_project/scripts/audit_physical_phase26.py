#!/usr/bin/env python
"""Cierre Fase 2.6: ejecuta/audita E1-E2 RAMIS/baseflow.

Uso recomendado completo:
    python scripts/audit_physical_phase26.py --config configs/experiments.yaml --run

Uso rapido para smoke test:
    python scripts/audit_physical_phase26.py --config configs/experiments.yaml --run --quick

Uso solo auditoria sobre salidas ya existentes:
    python scripts/audit_physical_phase26.py --config configs/experiments.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stohymolap.diagnostics.physical_audit import audit_physical_experiments  # noqa: E402
from stohymolap.experiments.comparison import run_comparison  # noqa: E402
from stohymolap.experiments.registry import run_experiment  # noqa: E402
from stohymolap.utils.config import deep_merge, load_config  # noqa: E402
from stohymolap.utils.logging import get_logger, setup_logging  # noqa: E402

_log = get_logger("scripts.audit_phase26")


def _load_merged_config(config_path: Path) -> dict:
    cfg = load_config(config_path)
    if "global" not in cfg:
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def _apply_quick_overrides(cfg: dict) -> dict:
    """Reduce costos para validar el pipeline; no usar para resultados finales."""
    cfg = dict(cfg)
    cfg["calibration"] = dict(cfg.get("calibration", {}))
    cfg["stochastic"] = dict(cfg.get("stochastic", {}))
    cfg["calibration"]["n_param_samples"] = 40
    cfg["calibration"]["n_iter"] = 12
    cfg["calibration"]["top_frac"] = 0.25
    cfg["stochastic"]["n_trajectories"] = 40
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Auditoria fisica Fase 2.6 RAMIS/baseflow.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiments", nargs="*", default=["E1_RAMIS_DET_BF", "E2_RAMIS_LEVY_BF"],
                        help="Experimentos fisicos a auditar.")
    parser.add_argument("--run", action="store_true", help="Ejecuta los experimentos antes de auditar.")
    parser.add_argument("--quick", action="store_true", help="Reduce iteraciones para prueba rapida; no sirve para resultados finales.")
    args = parser.parse_args(argv)

    cfg = _load_merged_config(args.config)
    if args.quick:
        cfg = _apply_quick_overrides(cfg)

    exp_root = Path(cfg.get("global", {}).get("output_root", "outputs/experiments"))
    comparison_root = exp_root.parent / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=comparison_root / "phase26_audit.log")

    if args.run:
        for eid in args.experiments:
            _log.info("Ejecutando %s%s", eid, " [quick]" if args.quick else "")
            run_experiment(cfg, eid)
            setup_logging(level=logging.INFO, log_file=comparison_root / "phase26_audit.log")

        run_comparison(exp_root, comparison_root, list(args.experiments))
        setup_logging(level=logging.INFO, log_file=comparison_root / "phase26_audit.log")

    result = audit_physical_experiments(exp_root, comparison_root, args.experiments)
    status = result["status"]
    _log.info("Estado auditoria Fase 2.6: %s", status)
    _log.info("Reporte: %s", result["report_path"])
    _log.info("Incidencias: %s", result["issues_path"])
    print(f"FASE 2.6 STATUS: {status}")
    print(f"Reporte: {result['report_path']}")
    print(f"Incidencias: {result['issues_path']}")
    return 0 if status in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
