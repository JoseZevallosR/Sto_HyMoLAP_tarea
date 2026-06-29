from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_phase34_figures.py"


def _load_phase34_module():
    spec = importlib.util.spec_from_file_location("generate_phase34_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prediction(root: Path, experiment: str, *, qsim_shift: float = 0.0, raw: bool = False, components: bool = False) -> None:
    exp_dir = root / "experiments" / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    qobs = np.linspace(1.0, 3.0, len(dates))
    df = pd.DataFrame({
        "date": dates,
        "Qobs": qobs,
        "Qsim": qobs + qsim_shift,
    })
    if raw:
        df["Qsim_raw"] = df["Qsim"]
    if components:
        df["Qfast"] = df["Qsim"] * 0.6
        df["Qbase"] = df["Qsim"] * 0.4
        df["Qtotal"] = df["Qsim"]
    df.to_csv(exp_dir / "predictions_validation.csv", index=False)
    (exp_dir / "regime_thresholds.json").write_text(
        json.dumps({"source": "train_Qobs_only", "p25": 1.5, "p75": 2.5}),
        encoding="utf-8",
    )


def test_phase34_figures_generate_minimal_manifest(tmp_path):
    module = _load_phase34_module()
    out = tmp_path / "outputs"
    comp = out / "comparison"
    comp.mkdir(parents=True)
    selected = ["E0_RAMIS_DET_NOBF", "E8_HYB_GRU_QUANTILES"]
    _write_prediction(out, selected[0], qsim_shift=0.1)
    _write_prediction(out, selected[1], qsim_shift=-0.1, raw=True)
    (out / "experiments" / selected[1] / "ml_backend.json").write_text(
        json.dumps({
            "experiment_id": selected[1],
            "ml_model": "gru",
            "backend": "sklearn_mlp",
            "backend_label": "fallback_mlp_on_flattened_sequences",
            "is_real_gru": False,
        }),
        encoding="utf-8",
    )
    pd.DataFrame({
        "experiment": selected,
        "n_eval": [8, 8],
        "common_start_date": ["2020-01-01", "2020-01-01"],
        "common_end_date": ["2020-01-08", "2020-01-08"],
    }).to_csv(comp / "leaderboard_common_intersection.csv", index=False)

    result = module.generate_phase34_figures(
        output_root=out,
        selected_experiments=selected,
        stochastic_references=[],
    )

    manifest_csv = result["manifest_csv"]
    manifest_json = result["manifest_json"]
    assert manifest_csv.exists()
    assert manifest_json.exists()
    manifest = pd.read_csv(manifest_csv)
    assert "hydrograph_validation_common" in set(manifest["figure_id"])
    assert "fdc_common" in set(manifest["figure_id"])
    assert f"scatter_{selected[1]}" in set(manifest["figure_id"])
    assert f"residuals_by_regime_{selected[0]}" in set(manifest["figure_id"])
    generated_paths = manifest.loc[manifest["status"].isin(["generated", "fallback"]), "path"].dropna()
    assert all(Path(path).exists() for path in generated_paths if path)
    hydro = manifest[manifest["figure_id"] == "hydrograph_validation_common"].iloc[0]
    assert "not a real GRU" in hydro["warnings"]


def test_phase34_figures_degrades_when_optional_columns_missing(tmp_path):
    module = _load_phase34_module()
    out = tmp_path / "outputs"
    comp = out / "comparison"
    comp.mkdir(parents=True)
    selected = ["E1_RAMIS_DET_BF"]
    refs = ["E2_RAMIS_LEVY_NOBF", "E3_RAMIS_LEVY_BF"]
    _write_prediction(out, selected[0], qsim_shift=0.0)
    _write_prediction(out, refs[0], qsim_shift=0.2)
    _write_prediction(out, refs[1], qsim_shift=-0.2)
    pd.DataFrame({
        "experiment": selected,
        "n_eval": [8],
        "common_start_date": ["2020-01-01"],
        "common_end_date": ["2020-01-08"],
    }).to_csv(comp / "leaderboard_common_intersection.csv", index=False)
    pd.DataFrame({
        "experiment": refs,
        "PICP": [0.16, 0.004],
        "PINAW": [0.08, 0.003],
    }).to_csv(comp / "uncertainty_comparison.csv", index=False)

    result = module.generate_phase34_figures(
        output_root=out,
        selected_experiments=selected,
        stochastic_references=refs,
    )
    manifest = pd.read_csv(result["manifest_csv"])

    component_row = manifest[manifest["figure_id"] == "physical_components_E1_RAMIS_DET_BF"].iloc[0]
    assert component_row["status"] == "skipped"
    assert "Qfast" in component_row["missing_columns"]

    stochastic_row = manifest[manifest["figure_id"] == "stochastic_reference_E2_E3"].iloc[0]
    assert stochastic_row["status"] == "fallback"
    assert "underdispersive" in stochastic_row["warnings"]
    assert Path(stochastic_row["path"]).exists()
