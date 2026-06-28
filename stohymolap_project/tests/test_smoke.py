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



def test_ramis_continuous_validation_does_not_use_validation_q0(tmp_path):
    from stohymolap.calibration.monte_carlo_search import CalibrationResult
    from stohymolap.experiments.runner import ExperimentRunner

    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    train = pd.DataFrame({
        "date": dates[:5],
        "Qobs": [1.0, 1.1, 1.2, 1.3, 1.4],
        "Peff": [0.0, 1.0, 0.0, 0.0, 0.0],
        "P": [0.0, 1.0, 0.0, 0.0, 0.0],
        "PET": [0.0] * 5,
        "Tmin": [5.0] * 5,
        "Tmax": [15.0] * 5,
    })
    val = pd.DataFrame({
        "date": dates[5:],
        "Qobs": [999.0, 2.0, 2.1],  # valor extremo: no debe inicializar la simulacion
        "Peff": [0.0, 0.0, 0.0],
        "P": [0.0, 0.0, 0.0],
        "PET": [0.0] * 3,
        "Tmin": [5.0] * 3,
        "Tmax": [15.0] * 3,
    })
    cfg = {
        "global": {"seed": 1, "output_root": str(tmp_path / "out")},
        "pet": {},
        "calibration": {},
        "stochastic": {},
        "baseflow": {"enabled": False},
        "experiments": {
            "E1_RAMIS_DET_BF": {
                "description": "test",
                "model_type": "physical",
                "stochastic": False,
                "baseflow": False,
            }
        },
    }
    runner = ExperimentRunner(cfg, "E1_RAMIS_DET_BF")
    runner.cal_result = CalibrationResult(
        best_params={
            "mu": 0.8, "lambda": 2.5, "sigma": 0.0, "alpha": 2.0, "beta": 0.0,
            "c_r": 0.0, "k_b": 0.001, "S0_b": 0.0, "n_top": 1,
        },
        top_k_table=np.zeros((1, 10)),
        qq_best=np.zeros((8, 1)),
        mean_trajectory=np.zeros(8),
        inf_trajectory=np.zeros(8),
        sup_trajectory=np.zeros(8),
        alpha_area=1.0,
        metrics={},
        diagnostics={},
    )

    sim_train, sim_val = runner.run_ramis_continuous(train, val)

    assert len(sim_train.q_total) == len(train)
    assert len(sim_val.q_total) == len(val)
    assert not np.isclose(sim_val.q_total[0], 999.0)
    assert sim_val.q_total[0] < 10.0


def test_baseflow_initialization_does_not_double_count_q0():
    from stohymolap.hydro.baseflow import (
        BaseflowParams,
        add_baseflow,
        quickflow_initial_from_total,
    )
    from stohymolap.hydro.ramis import simulate_fast_deterministic

    q0_total = 10.0
    peff = np.zeros(5, dtype=float)
    params = BaseflowParams(c_r=0.0, k_b=0.2, S0_b=15.0)  # Qbase0=3

    q0_fast = quickflow_initial_from_total(q0_total, params, use_baseflow=True)
    q_fast = simulate_fast_deterministic(
        mu=0.8, lambda_=2.5, peff=peff, q0=q0_fast, alpha_area=1.0
    )
    q_total, q_base = add_baseflow(
        q_fast, peff, params, q0_obs=q0_total, alpha_area=1.0
    )

    assert np.isclose(q_base[0], 3.0)
    assert np.isclose(q_fast[0], 7.0)
    assert np.isclose(q_total[0], q0_total)


def test_baseflow_initial_storage_is_capped_by_total_q0():
    from stohymolap.hydro.baseflow import BaseflowParams, linear_reservoir

    params = BaseflowParams(c_r=0.0, k_b=0.5, S0_b=1000.0)
    q_base, storage = linear_reservoir(
        np.zeros(3), params, q0_obs=10.0, alpha_area=1.0
    )

    assert q_base[0] <= 9.5 + 1e-12
    assert storage[0] <= 19.0 + 1e-12


def test_ramis_rejects_unstable_parameter_bounds():
    from stohymolap.hydro.ramis import validate_ramis_bounds

    bad_bounds = {
        "mu": (0.75, 1.2),
        "lambda": (1.0, 3.0),
        "sigma": (0.0, 0.1),
    }
    try:
        validate_ramis_bounds(bad_bounds)
    except ValueError as exc:
        assert "inestables" in str(exc) or "mu/lambda" in str(exc)
        return
    raise AssertionError("No rechazo bounds RAMIS inestables.")


def test_vectorized_objective_can_select_by_pbias_not_nse():
    from stohymolap.calibration.objective import objective_value_vectorized

    obs = np.array([1.0, 2.0, 3.0, 4.0])
    # sim_a conserva mejor la forma, pero tiene sesgo positivo claro.
    # sim_b tiene peor forma, pero balance volumetrico perfecto.
    sims = np.array([
        [2.0, 3.0, 4.0, 5.0],
        [4.0, 1.0, 2.0, 3.0],
    ])
    obj = objective_value_vectorized(
        obs, sims, weights={"nse": 0.0, "kge": 0.0, "pbias": 1.0}
    )

    assert int(np.nanargmin(obj["J"])) == 1
    assert abs(obj["PBIAS"][1]) < abs(obj["PBIAS"][0])


def test_calibration_selects_best_j_row_instead_of_top_k_mean():
    from stohymolap.calibration.monte_carlo_search import calibrate

    rng = np.random.default_rng(321)
    n = 90
    peff = np.clip(rng.gamma(1.4, 2.0, n) * (rng.random(n) < 0.4), 0, None)
    discharge = np.clip(0.3 + np.convolve(peff, [0.20, 0.10, 0.04], mode="same"), 0, None)
    bounds = {
        "mu": (0.75, 0.95),
        "lambda": (2.0, 3.4),
        "sigma": (0.0, 0.0),
        "alpha": (1.1, 1.9),
        "beta": (-1.0, 0.0),
        "c_r": (0.0, 0.2),
        "k_b": (0.05, 0.3),
        "S0_b": (0.0, 2.0),
    }

    result = calibrate(
        discharge, peff, bounds=bounds,
        n_traj=6, n_param_samples=10, top_frac=1.0, seed=11,
        use_baseflow=True, calibrate_baseflow=True, use_stochastic=False,
        parameter_selection="best_j",
    )

    assert result.best_params["selection_strategy"] == "best_j"
    assert result.best_params["selected_rank"] == 1
    assert np.isclose(result.best_params["mu"], result.top_k_table[0, 0])
    assert np.isclose(result.best_params["selected_J"], result.top_k_table[0, 9])
    assert bool(result.top_k_table[0, -1])


def test_phase26_e1_saves_physical_components(tmp_path):
    from stohymolap.experiments.registry import run_experiment

    data_path = _tiny_dataset(tmp_path)
    cfg = {
        "global": {
            "data_path": str(data_path),
            "train_fraction": 0.7,
            "forecast_horizon": 1,
            "lags": [0, 1, 2],
            "seed": 5,
            "output_root": str(tmp_path / "out"),
            "fill_obs_with_sim": False,
        },
        "pet": {"latitude": -15.0},
        "calibration": {"n_param_samples": 15, "n_iter": 8, "top_frac": 0.25, "parameter_selection": "best_j"},
        "stochastic": {"n_trajectories": 10, "levy": {}},
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
    run_experiment(cfg, "E1_RAMIS_DET_BF")
    comp_path = tmp_path / "out" / "E1_RAMIS_DET_BF" / "physical_components_validation.csv"
    assert comp_path.exists()
    comp = pd.read_csv(comp_path)
    for col in ["Qtotal", "Qfast", "Qbase"]:
        assert col in comp.columns
    assert np.allclose(comp["Qtotal"], comp["Qfast"] + comp["Qbase"])


def test_phase26_physical_audit_passes_on_valid_outputs(tmp_path):
    from stohymolap.diagnostics.physical_audit import audit_physical_experiments

    exp_root = tmp_path / "experiments"
    eid = "E1_RAMIS_DET_BF"
    exp_dir = exp_root / eid
    exp_dir.mkdir(parents=True)

    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    qobs = np.linspace(1.0, 3.0, len(dates))
    qbase = np.full(len(dates), 0.3)
    qfast = qobs - qbase
    qtotal = qfast + qbase

    pred = pd.DataFrame({"date": dates, "Qobs": qobs, "Qsim": qtotal})
    pred.to_csv(exp_dir / "predictions_train.csv", index=False)
    pred.to_csv(exp_dir / "predictions_validation.csv", index=False)
    comp = pd.DataFrame({
        "date": dates, "Qobs": qobs, "Peff": np.zeros(len(dates)),
        "Qtotal": qtotal, "Qbase": qbase, "Qfast": qfast,
    })
    comp.to_csv(exp_dir / "physical_components_train.csv", index=False)
    comp.to_csv(exp_dir / "physical_components_validation.csv", index=False)

    best = pd.DataFrame([{
        "mu": 0.8, "lambda": 2.5, "sigma": 0.0, "alpha": 2.0, "beta": 0.0,
        "c_r": 0.2, "k_b": 0.1, "S0_b": 3.0, "n_top": 1,
        "selection_strategy": "best_j", "selected_rank": 1, "selected_J": 0.1,
    }])
    best.to_csv(exp_dir / "best_parameters.csv", index=False)
    top = pd.DataFrame([{
        "mu": 0.8, "lambda": 2.5, "sigma": 0.0, "alpha": 2.0, "beta": 0.0,
        "c_r": 0.2, "k_b": 0.1, "S0_b": 3.0, "nse": 0.9, "J": 0.1,
        "KGE": 0.9, "PBIAS": 0.0, "rank": 1, "selected": 1,
    }])
    top.to_csv(exp_dir / "top_k_parameters.csv", index=False)
    pd.DataFrame({
        "experiment": [eid, eid, eid],
        "regime": ["low", "mid", "high"],
        "n": [4, 4, 4],
        "NSE": [1.0, 1.0, 1.0],
        "KGE": [1.0, 1.0, 1.0],
        "RMSE": [0.0, 0.0, 0.0],
        "MAE": [0.0, 0.0, 0.0],
        "PBIAS": [0.0, 0.0, 0.0],
    }).to_csv(exp_dir / "metrics_by_regime.csv", index=False)

    result = audit_physical_experiments(exp_root, tmp_path / "comparison", [eid])
    assert result["status"] == "PASS"
    assert result["report_path"].exists()
    assert result["audit_issues"].empty



def test_phase31_experiment_matrix_is_canonical():
    from stohymolap.experiments.registry import CANONICAL_ORDER, DEPRECATED_EXPERIMENT_IDS, list_experiments
    from stohymolap.utils.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml")
    listed = list_experiments(cfg)

    assert listed == CANONICAL_ORDER
    assert "E0_RAMIS_DET_NOBF" in listed
    assert "E2_RAMIS_LEVY_NOBF" in listed
    assert "E3_RAMIS_LEVY_BF" in listed
    for old_id in DEPRECATED_EXPERIMENT_IDS:
        assert old_id not in listed


def test_phase31_comparison_matrix_and_ablation_effects(tmp_path):
    from stohymolap.experiments.comparison import (
        build_ablation_effects,
        build_experiment_matrix,
    )
    from stohymolap.experiments.registry import CANONICAL_ORDER

    matrix = build_experiment_matrix(CANONICAL_ORDER)
    assert matrix["experiment"].tolist() == CANONICAL_ORDER
    row_e0 = matrix.set_index("experiment").loc["E0_RAMIS_DET_NOBF"]
    row_e3 = matrix.set_index("experiment").loc["E3_RAMIS_LEVY_BF"]
    assert not bool(row_e0["baseflow"])
    assert bool(row_e3["estocastico"])
    assert bool(row_e3["baseflow"])

    results = pd.DataFrame([
        {"experiment": "E2_RAMIS_LEVY_NOBF", "NSE": 0.30, "KGE": 0.40, "RMSE": 2.0, "MAE": 1.5, "PBIAS": 20.0, "n_eval": 100},
        {"experiment": "E3_RAMIS_LEVY_BF", "NSE": 0.45, "KGE": 0.55, "RMSE": 1.7, "MAE": 1.2, "PBIAS": 10.0, "n_eval": 100},
    ])
    ab = build_ablation_effects(results).set_index("ablation")
    assert "stochastic_baseflow" in ab.index
    assert np.isclose(ab.loc["stochastic_baseflow", "delta_NSE"], 0.15)
    assert np.isclose(ab.loc["stochastic_baseflow", "delta_RMSE"], -0.3)
    assert np.isclose(ab.loc["stochastic_baseflow", "delta_abs_PBIAS_improvement"], 10.0)


def test_phase32_common_window_infers_canonical_warmup():
    from stohymolap.experiments.evaluation_window import resolve_evaluation_window
    from stohymolap.utils.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml")
    win = resolve_evaluation_window(cfg)

    assert win.enabled is True
    assert win.mode == "declared_matrix_max_warmup"
    assert win.start_offset == 7
    assert win.end_trim == 0


def test_phase32_common_validation_window_equalizes_physical_and_ml(tmp_path, monkeypatch):
    from stohymolap.experiments.registry import run_experiment
    import stohymolap.experiments.runner as runner_mod

    class _DummyModel:
        backend = "dummy"
        def fit(self, X, y):
            self.mean_ = float(np.nanmean(y))
            return self
        def predict(self, X):
            return np.full(len(X), self.mean_, dtype=float)
        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(runner_mod, "build_model", lambda *args, **kwargs: _DummyModel())

    data_path = _tiny_dataset(tmp_path)
    cfg = {
        "global": {
            "data_path": str(data_path),
            "train_fraction": 0.7,
            "forecast_horizon": 1,
            "lags": [0, 1, 2],
            "seed": 9,
            "output_root": str(tmp_path / "out"),
            "fill_obs_with_sim": False,
            "ramis_state_mode": "continuous_train_validation",
        },
        "pet": {"latitude": -15.0},
        "calibration": {"n_param_samples": 15, "n_iter": 8, "top_frac": 0.25, "parameter_selection": "best_j"},
        "stochastic": {"n_trajectories": 10, "levy": {}},
        "baseflow": {"enabled": True, "calibrate": True},
        "evaluation": {"common_window": {"enabled": True, "mode": "declared_matrix_max_warmup", "start_offset": "auto", "end_trim": 0}},
        "experiments": {
            "E1_RAMIS_DET_BF": {
                "description": "physical",
                "model_type": "physical",
                "stochastic": False,
                "baseflow": True,
            },
            "E4_ML_PURE_XGB": {
                "description": "ml",
                "model_type": "machine_learning",
                "ml_model": "xgboost",
                "stochastic": False,
                "baseflow": False,
                "features": {"source": "observed_forcing", "variables": ["P", "PET", "Tmin", "Tmax"], "lags": [0, 1, 2]},
            },
        },
    }

    run_experiment(cfg, "E1_RAMIS_DET_BF")
    run_experiment(cfg, "E4_ML_PURE_XGB")

    out = tmp_path / "out"
    p_phys = pd.read_csv(out / "E1_RAMIS_DET_BF" / "predictions_validation.csv")
    p_ml = pd.read_csv(out / "E4_ML_PURE_XGB" / "predictions_validation.csv")

    assert len(p_phys) == len(p_ml)
    assert p_phys["date"].iloc[0] == p_ml["date"].iloc[0]
    assert p_phys["date"].iloc[-1] == p_ml["date"].iloc[-1]

    import json
    meta = json.loads((out / "E1_RAMIS_DET_BF" / "evaluation_window.json").read_text())
    assert meta["start_offset"] == 3
    assert meta["validation"]["n_after"] == len(p_phys)
