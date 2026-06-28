#!/usr/bin/env python
"""Reconstruye la comparacion a partir de salidas ya existentes.

Util cuando los experimentos ya se ejecutaron y solo se quiere regenerar el
leaderboard, las tablas y figuras de comparacion sin recalcular nada.

Uso:
    python scripts/summarize_results.py --config configs/experiments.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stohymolap.experiments.comparison import run_comparison  # noqa: E402
from stohymolap.experiments.registry import list_experiments  # noqa: E402
from stohymolap.utils.config import deep_merge, load_config  # noqa: E402
from stohymolap.utils.logging import get_logger, setup_logging  # noqa: E402

_log = get_logger("scripts.summarize")


def _load_merged_config(config_path: Path) -> dict:
    cfg = load_config(config_path)
    if "global" not in cfg:
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resume resultados existentes.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = _load_merged_config(args.config)
    experiment_ids = list_experiments(cfg)

    exp_root = Path(cfg.get("global", {}).get("output_root", "outputs/experiments"))
    # Solo considerar los que tienen salidas en disco.
    present = [e for e in experiment_ids if (exp_root / e / "metrics_validation.csv").exists()]
    comparison_root = exp_root.parent / "comparison"
    leaderboard = run_comparison(exp_root, comparison_root, present)

    if not leaderboard.empty:
        _log.info("Leaderboard:\n%s", leaderboard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
