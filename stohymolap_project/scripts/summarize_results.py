#!/usr/bin/env python
"""Regenera la comparación final desde outputs ya existentes, sin recalcular modelos."""
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


def _load_merged_config(config_path: Path) -> dict:
    cfg = load_config(config_path)
    if "global" not in cfg:
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resume resultados StoHyMoLAP existentes.")
    parser.add_argument("--config", required=True, type=Path, help="Archivo YAML de configuración.")
    parser.add_argument("--only", nargs="*", default=None, help="Subconjunto de experimentos a comparar.")
    parser.add_argument("--n-best-for-plots", type=int, default=4)
    args = parser.parse_args(argv)

    cfg = _load_merged_config(args.config)
    experiment_ids = list_experiments(cfg)
    if args.only:
        requested = set(args.only)
        experiment_ids = [eid for eid in experiment_ids if eid in requested]

    exp_root = Path(cfg.get("global", {}).get("output_root", "outputs/experiments"))
    comparison_root = exp_root.parent / "comparison"
    leaderboard = run_comparison(
        exp_root,
        comparison_root,
        experiment_ids,
        n_best_for_plots=args.n_best_for_plots,
    )
    print(f"Comparación regenerada en: {comparison_root}")
    if not leaderboard.empty:
        print(leaderboard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

