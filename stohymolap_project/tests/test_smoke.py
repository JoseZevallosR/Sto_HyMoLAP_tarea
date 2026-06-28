"""Pruebas de humo del pipeline.

Verifican imports, validaciones de Fase 0/1 y que un experimento determinista
corre de extremo a extremo sobre un dataset pequeño sintético.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _tiny_dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    p = np.clip(rng.gamma(1.4, 5.0, n) * (rng.random(n) < 0.4), 0, None)
    tmin = 5 + rng.normal(0, 1, n)
    tmax = tmin + 10 + rng.normal(0, 1, n)
    q = np.clip(np.convolve(p, [0.5, 0.3, 0.2], mode="same") + rng.normal(0, 0.5, n), 0, None)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "precipitation_mean": np.round(p, 2),
        "tmin": np.round(tmin, 2),
        "tmax": np.round(tmax, 2),
        "flow_obs": np.round(q, 3),
        "flow_sim": np.round(q + 0.1, 3),
    })
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


def test_imports():
    import stohymolap  # noqa: F401
    from stohymolap.experiments.runner import ExperimentRunner  # noqa: F401
    from stohymolap import validation_checks  # noqa: F401


def test_validation_checks_catch_leakage():
    from stohymolap.validation_checks import ValidationError, check_temporal_split
    tr = pd.to_datetime(["2020-01-01", "2020-01-05"])
    va = pd.to_datetime(["2020-01-03", "2020-01-10"])
    try:
        check_temporal_split(tr, va)
    except ValidationError:
        return
    raise AssertionError("No detecto la fuga temporal.")


def test_qobs_not_filled_by_default(tmp_path):
    from stohymolap.data.io import load_hydro_dataframe

    path = _tiny_dataset(tmp_path)
    raw = pd.read_csv(path)
    raw.loc[0, "flow_obs"] = np.nan
    raw.to_csv(path, index=False)

    df = load_hydro_dataframe(path)
    assert pd.isna(df.loc[0, "Qobs"])
    assert df.loc[0, "Qobs_source"] == "missing"
    assert not bool(df.loc[0, "Qobs_is_filled_from_qsim"])

    df_fill = load_hydro_dataframe(path, fill_obs_with_sim=True)
    assert np.isfinite(df_fill.loc[0, "Qobs"])
    assert df_fill.loc[0, "Qobs_source"] == "filled_from_qsim"
    assert bool(df_fill.loc[0, "Qobs_is_filled_from_qsim"])


def test_validation_rejects_bad_physical_values(tmp_path):
    from stohymolap.validation_checks import ValidationError, validate_hydro_dataframe

    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3, freq="D"),
        "Qobs": [1.0, 1.1, 1.2],
        "P": [0.0, -1.0, 2.0],
        "Tmin": [5.0, 6.0, 7.0],
        "Tmax": [10.0, 11.0, 12.0],
    })
    try:
        validate_hydro_dataframe(df)
    except ValidationError as exc:
        assert "P negativa" in str(exc)
        return
    raise AssertionError("No detecto precipitacion negativa.")


def test_e1_runs_end_to_end(tmp_path):
    from stohymolap.experiments.registry import run_experiment

    data_path = _tiny_dataset(tmp_path)
    cfg = {
        "global": {
            "data_path": str(data_path),
            "train_fraction": 0.7,
            "forecast_horizon": 1,
            "lags": [0, 1, 2],
            "seed": 1,
            "output_root": str(tmp_path / "out"),
            "fill_obs_with_sim": False,
        },
        "pet": {"latitude": -15.0},
        "calibration": {"n_param_samples": 20, "n_iter": 15, "top_frac": 0.2},
        "stochastic": {"n_trajectories": 20, "levy": {}},
        "baseflow": {"enabled": True, "calibrate": True},
        "experiments": {
            "E1_RAMIS_DET_BF": {
                "description": "test",
                "model_type": "physical",
                "stochastic": False,
                "baseflow": True,
            }
        },
    }
    res = run_experiment(cfg, "E1_RAMIS_DET_BF")
    assert "metrics_validation" in res
    assert np.isfinite(res["metrics_validation"]["NSE"])
    out_dir = tmp_path / "out" / "E1_RAMIS_DET_BF"
    assert (out_dir / "metrics_validation.csv").exists()
    used = (out_dir / "config_used.yaml").read_text(encoding="utf-8")
    assert "runtime:" in used
    assert "git_commit:" in used


def test_baseflow_uses_same_alpha_area_scale_as_ramis():
    from stohymolap.hydro.baseflow import BaseflowParams, linear_reservoir

    peff = np.array([0.0, 10.0, 0.0, 0.0], dtype=float)
    params = BaseflowParams(c_r=1.0, k_b=0.5, S0_b=0.0)
    q_base_1, _ = linear_reservoir(peff, params, alpha_area=1.0)
    q_base_2, _ = linear_reservoir(peff, params, alpha_area=2.0)

    assert np.allclose(q_base_2, 2.0 * q_base_1)


def test_water_balance_without_baseflow_keeps_qfast():
    from stohymolap.hydro.water_balance import assemble_total_discharge

    q_fast = np.array([1.0, 2.0, 3.0])
    peff = np.array([10.0, 0.0, 0.0])
    q_total, q_base = assemble_total_discharge(
        q_fast, peff, use_baseflow=False, alpha_area=999.0
    )

    assert np.allclose(q_total, q_fast)
    assert np.allclose(q_base, 0.0)

def test_deterministic_calibration_forces_sigma_zero():
    from stohymolap.calibration.monte_carlo_search import calibrate

    rng = np.random.default_rng(123)
    n = 80
    peff = np.clip(rng.gamma(1.2, 2.0, n) * (rng.random(n) < 0.35), 0, None)
    discharge = np.clip(0.2 + np.convolve(peff, [0.18, 0.08, 0.03], mode="same"), 0, None)
    bounds = {
        "mu": (0.75, 0.95),
        "lambda": (2.0, 3.4),
        "sigma": (0.05, 0.10),  # debe ignorarse en modo deterministico
        "alpha": (1.1, 1.9),
        "beta": (-1.0, 0.0),
        "c_r": (0.0, 0.0),
        "k_b": (0.05, 0.05),
        "S0_b": (0.0, 0.0),
    }

    result = calibrate(
        discharge, peff, bounds=bounds,
        n_traj=4, n_param_samples=8, top_frac=0.5, seed=7,
        use_baseflow=False, use_stochastic=False,
    )

    assert result.best_params["sigma"] == 0.0
    assert np.allclose(result.top_k_table[:, 2], 0.0)
    assert result.diagnostics["use_stochastic"] is False

