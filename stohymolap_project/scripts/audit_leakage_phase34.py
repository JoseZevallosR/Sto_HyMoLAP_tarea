#!/usr/bin/env python
"""Ejecuta la auditoria anti-leakage/QA de Fase 3.4A."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stohymolap.diagnostics.leakage_audit import audit_leakage  # noqa: E402
from stohymolap.experiments.registry import list_experiments  # noqa: E402
from stohymolap.utils.config import deep_merge, load_config  # noqa: E402


def _load_merged_config(config_path: Path) -> dict:
    cfg = load_config(config_path)
    if "global" not in cfg:
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audita leakage y QA de outputs StoHyMoLAP Fase 3.4A.")
    parser.add_argument("--config", required=True, type=Path, help="Archivo YAML de configuracion.")
    parser.add_argument("--only", nargs="*", default=None, help="Subconjunto de experimentos a auditar.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Salir con codigo 1 tambien en estado WARN.")
    args = parser.parse_args(argv)

    cfg = _load_merged_config(args.config)
    experiment_ids = list_experiments(cfg)
    if args.only:
        requested = set(args.only)
        experiment_ids = [eid for eid in experiment_ids if eid in requested]

    exp_root = Path(cfg.get("global", {}).get("output_root", "outputs/experiments"))
    comparison_root = exp_root.parent / "comparison"
    result = audit_leakage(cfg, exp_root, comparison_root, experiment_ids, write=True)

    print(f"Auditoria Fase 3.4A: {result['status']}")
    print(f"Reporte: {result['report_path']}")
    print(f"Issues: {result['issues_path']}")
    if result["status"] == "FAIL" or (args.fail_on_warn and result["status"] == "WARN"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
