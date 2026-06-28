from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_phase34_report.py"


def _load_phase34_report_module():
    spec = importlib.util.spec_from_file_location("generate_phase34_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_minimal_phase34_outputs(out: Path) -> None:
    comp = out / "comparison"
    figs = out / "figures" / "phase34"
    comp.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "rank": [1, 2],
        "experiment": ["E8_HYB_GRU_QUANTILES", "E6_HYB_XGB_QUANTILES"],
        "n_eval": [8, 8],
        "common_start_date": ["2020-01-01", "2020-01-01"],
        "common_end_date": ["2020-01-08", "2020-01-08"],
        "NSE": [0.85, 0.74],
        "KGE": [0.90, 0.81],
        "RMSE": [0.21, 0.28],
        "MAE": [0.12, 0.16],
        "PBIAS": [1.5, 0.9],
        "R2": [0.85, 0.74],
    }).to_csv(comp / "leaderboard_common_intersection.csv", index=False)

    pd.DataFrame({
        "experiment": ["E8_HYB_GRU_QUANTILES", "E6_HYB_XGB_QUANTILES"],
        "model_type": ["hybrid", "hybrid"],
        "ml_model": ["gru", "xgb"],
        "backend": ["sklearn_mlp", "xgboost"],
        "backend_label": ["fallback_mlp_on_flattened_sequences", "xgboost"],
        "feature_source": ["ramis_features", "ramis_features"],
        "postprocess_method": ["clip_nonnegative", "clip_nonnegative"],
    }).to_csv(comp / "ml_backend_summary.csv", index=False)

    pd.DataFrame({
        "ablation": ["hybrid_gain"],
        "baseline": ["E3_RAMIS_LEVY_BF"],
        "candidate": ["E8_HYB_GRU_QUANTILES"],
        "delta_NSE": [0.4],
        "delta_KGE": [0.2],
        "delta_RMSE": [-0.1],
        "delta_MAE": [-0.08],
        "delta_abs_PBIAS_improvement": [2.0],
    }).to_csv(comp / "ablation_effects_common_intersection.csv", index=False)

    pd.DataFrame({
        "experiment": ["E8_HYB_GRU_QUANTILES", "E8_HYB_GRU_QUANTILES"],
        "regime": ["low", "high"],
        "n": [3, 3],
        "NSE": [0.4, 0.7],
        "KGE": [0.5, 0.8],
        "RMSE": [0.1, 0.2],
        "MAE": [0.08, 0.14],
        "PBIAS": [4.0, -2.0],
    }).to_csv(comp / "regime_metrics_comparison.csv", index=False)

    pd.DataFrame({
        "experiment": ["E2_RAMIS_LEVY_NOBF", "E3_RAMIS_LEVY_BF"],
        "PICP": [0.16, 0.004],
        "PINAW": [0.08, 0.003],
        "MPIW": [0.2, 0.01],
        "Winkler": [7.2, 10.0],
    }).to_csv(comp / "uncertainty_comparison.csv", index=False)

    (comp / "leakage_audit_status.txt").write_text("WARN\n", encoding="utf-8")
    (comp / "physical_audit_status.txt").write_text("WARN\n", encoding="utf-8")

    pd.DataFrame({
        "figure_id": ["hydrograph_validation_common", "physical_components_E8_HYB_GRU_QUANTILES"],
        "status": ["generated", "skipped"],
        "path": ["outputs/figures/phase34/hydrograph_validation_common.png", ""],
        "experiments": ["E8_HYB_GRU_QUANTILES", "E8_HYB_GRU_QUANTILES"],
        "warnings": ["declared GRU uses backend=fallback_mlp_on_flattened_sequences; not a real GRU", ""],
    }).to_csv(figs / "phase34_figure_manifest.csv", index=False)


def test_phase34_report_generates_markdown_and_tables(tmp_path):
    module = _load_phase34_report_module()
    out = tmp_path / "outputs"
    _write_minimal_phase34_outputs(out)

    result = module.generate_phase34_report(output_root=out)

    report_path = result["report_path"]
    manifest_path = result["manifest_path"]
    assert report_path.exists()
    assert manifest_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Paper-ready result package" in text
    assert "E8_HYB_GRU_QUANTILES" in text
    assert "fallback MLP" in text
    assert "Anti-leakage audit status: `WARN`" in text

    report_dir = result["report_dir"]
    expected_tables = [
        "table_final_ranking_common_window.csv",
        "table_selected_models_common_window.csv",
        "table_ablation_effects_common_window.csv",
        "table_backend_manuscript_notes.csv",
        "table_reproducibility_checklist.csv",
    ]
    for name in expected_tables:
        assert (report_dir / name).exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) >= len(expected_tables)
    assert any("GRU over-claim risk" in warning for warning in manifest["warnings"])


def test_phase34_report_degrades_when_optional_inputs_missing(tmp_path):
    module = _load_phase34_report_module()
    out = tmp_path / "outputs"
    comp = out / "comparison"
    comp.mkdir(parents=True)
    pd.DataFrame({
        "experiment": ["E0_RAMIS_DET_NOBF"],
        "n_eval": [4],
        "common_start_date": ["2020-01-01"],
        "common_end_date": ["2020-01-04"],
        "NSE": [0.4],
        "KGE": [0.7],
        "RMSE": [0.3],
        "MAE": [0.2],
        "PBIAS": [1.0],
        "R2": [0.5],
    }).to_csv(comp / "validation_metrics_common_intersection.csv", index=False)

    result = module.generate_phase34_report(output_root=out, selected_experiments=["E0_RAMIS_DET_NOBF"])

    assert result["report_path"].exists()
    text = result["report_path"].read_text(encoding="utf-8")
    assert "E0_RAMIS_DET_NOBF" in text
    checklist = pd.read_csv(result["report_dir"] / "table_reproducibility_checklist.csv")
    assert "Figure manifest" in set(checklist["item"])
    assert result["warnings"]
