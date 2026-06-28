"""ExperimentRunner: orquesta un experimento de principio a fin.

Implementa el flujo descrito en el pliego:

1. Lee configuracion del experimento.        7. Calcula baseflow si esta activo.
2. Carga datos.                              8. Genera features de incertidumbre.
3. Calcula PET.                              9. Construye features (tabular/secuencia).
4. Split train/validation temporal.         10. Entrena ML si corresponde.
5. Calibra RAMIS si corresponde.             11. Predice train y validation.
6. Ejecuta RAMIS det/estocastico.            12-14. Metricas, figuras y guardado.

Un mismo runner cubre la matriz Fase 3.2 seleccionando ramas
segun ``model_type``, ``stochastic``, ``baseflow`` y ``ml_model``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..calibration.monte_carlo_search import CalibrationResult, calibrate
from ..calibration.parameter_store import save_best_parameters, save_top_k
from ..data.io import load_hydro_dataframe
from ..data.preprocessing import (
    add_pet_and_peff,
    drop_invalid_rows,
    temporal_split,
    trim_to_first_valid_qobs,
)
from ..features.feature_builder import build_sequences, build_tabular, select_feature_columns
from .evaluation_window import apply_common_window, resolve_evaluation_window
from ..features.uncertainty_features import ensemble_summary_to_frame
from ..hydro.baseflow import BaseflowParams, quickflow_initial_from_total
from ..hydro.ramis import simulate_fast_deterministic
from ..hydro.water_balance import assemble_total_discharge
from ..metrics.deterministic import all_deterministic
from ..metrics.regime_metrics import metrics_by_regime, regime_thresholds
from ..metrics.uncertainty import all_uncertainty
from ..ml.train_predict import build_model, fit_predict_sequence, fit_predict_tabular
from ..stochastic.monte_carlo import run_monte_carlo, summarize_ensemble
from ..utils.config import ExperimentConfig, dump_config
from ..utils.logging import get_logger, setup_logging
from ..utils.reproducibility import child_rng, runtime_metadata, seed_everything
from .. import validation_checks as checks  # type: ignore

_log = get_logger("experiments.runner")


def _read_json_safe(path: Path) -> Dict[str, Any]:
    """Lee JSON si existe; devuelve {} si aun no fue creado."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class _SimBundle:
    """Salida de RAMIS para un subconjunto (train o val)."""

    q_total: np.ndarray                 # prediccion fisica puntual (Qmean o Qdet)
    q_base: np.ndarray
    summary: Dict[str, np.ndarray]
    lower: Optional[np.ndarray] = None  # q025
    upper: Optional[np.ndarray] = None  # q975


class ExperimentRunner:
    """Ejecuta un experimento individual y escribe sus salidas."""

    def __init__(self, cfg: Dict[str, Any], experiment_id: str):
        self.full_cfg = cfg
        self.ec = ExperimentConfig.from_dict(cfg, experiment_id)
        self.experiment_id = experiment_id
        self.out_dir = Path(self.ec.output_dir)
        self.fig_dir = self.out_dir / "figures"
        self.rng = seed_everything(self.ec.seed)
        self.cal_result: Optional[CalibrationResult] = None
        self.baseflow_params: Optional[BaseflowParams] = None

    # ---------------------------------------------------------------- setup
    def _setup_outputs(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(level=logging.INFO, log_file=self.out_dir / "run.log")
        # Guarda la config efectivamente usada.
        used = {
            "experiment_id": self.experiment_id,
            "seed": self.ec.seed,
            "runtime": runtime_metadata(),
            "global": self.ec.glob,
            "pet": self.ec.pet,
            "calibration": self.ec.calibration,
            "stochastic": self.ec.stochastic,
            "baseflow": self.ec.baseflow,
            "evaluation": self.ec.evaluation,
            "experiment": self.ec.exp,
        }
        dump_config(used, self.out_dir / "config_used.yaml")

    # ----------------------------------------------------------- data steps
    def load_and_prepare_data(self) -> pd.DataFrame:
        g = self.ec.glob
        fill_obs_with_sim = bool(g.get("fill_obs_with_sim", False))
        df = load_hydro_dataframe(
            g["data_path"],
            date_col=g.get("date_col", "date"),
            qobs_col=g.get("qobs_col", "flow_obs"),
            p_col=g.get("p_col", "precipitation_mean"),
            tmin_col=g.get("tmin_col", "tmin"),
            tmax_col=g.get("tmax_col", "tmax"),
            qsim_col=g.get("qsim_col", "flow_sim"),
            fill_obs_with_sim=fill_obs_with_sim,
            csv_sep=g.get("csv_sep", ","),
        )
        checks.validate_hydro_dataframe(
            df,
            require_daily=True,
            allow_missing_qobs=not fill_obs_with_sim,
            allow_filled_qobs=fill_obs_with_sim,
        )
        _log.info("Resumen origen Qobs: %s", checks.qobs_source_summary(df))
        df = add_pet_and_peff(df, latitude_deg=self.ec.pet.get("latitude", -15.0))
        checks.check_required_columns(df, ["date", "Qobs", "P", "PET", "Peff"])
        checks.check_pet_non_negative(df["PET"].to_numpy())
        return df

    # ----------------------------------------------------------- calibration
    def _bounds(self) -> Dict[str, Any]:
        st = self.ec.stochastic.get("levy", {})
        bf = self.ec.baseflow
        return {
            "mu": tuple(self.ec.calibration.get("mu_bounds", (0.75, 0.95))),
            "lambda": tuple(self.ec.calibration.get("lambda_bounds", (2.0, 3.4))),
            "sigma": tuple(self.ec.calibration.get("sigma_bounds", (0.0, 0.1))),
            "alpha": tuple(st.get("alpha_bounds", (1.1, 1.9))),
            "beta": tuple(st.get("beta_bounds", (-1.0, 0.0))),
            "c_r": tuple(bf.get("c_r_bounds", (0.0, 1.0))),
            "k_b": tuple(bf.get("k_b_bounds", (0.001, 0.5))),
            "S0_b": tuple(bf.get("S0_b_bounds", (0.0, 10.0))),
        }

    def calibrate_params(self, train: pd.DataFrame) -> CalibrationResult:
        cal = self.ec.calibration
        weights = cal.get("objective", {}).get("weights")
        result = calibrate(
            discharge=train["Qobs"].to_numpy(),
            peff=train["Peff"].to_numpy(),
            bounds=self._bounds(),
            n_traj=int(cal.get("n_iter", 1000)),
            n_param_samples=int(cal.get("n_param_samples", 3000)),
            top_frac=float(cal.get("top_frac", 0.1)),
            seed=self.ec.seed,
            use_baseflow=self.ec.use_baseflow,
            calibrate_baseflow=bool(self.ec.baseflow.get("calibrate", False)),
            use_stochastic=self.ec.use_stochastic,
            objective_weights=weights,
            parameter_selection=cal.get("parameter_selection", "best_j"),
        )
        self.cal_result = result
        bp = result.best_params
        self.baseflow_params = BaseflowParams(c_r=bp["c_r"], k_b=bp["k_b"], S0_b=bp["S0_b"])
        save_best_parameters(result, self.out_dir / "best_parameters.csv")
        save_top_k(result, self.out_dir / "top_k_parameters.csv")
        return result

    # ------------------------------------------------------------- run RAMIS
    def run_ramis(self, subset: pd.DataFrame, *, offset: int) -> _SimBundle:
        """Ejecuta RAMIS (det o estocastico) sobre un subconjunto."""
        bp = self.cal_result.best_params
        peff = subset["Peff"].to_numpy()
        qobs = subset["Qobs"].to_numpy(dtype=float)
        finite_qobs = np.isfinite(qobs)
        if not finite_qobs.any():
            raise ValueError(f"{self.experiment_id}: subconjunto sin Qobs finito para q0.")
        q0 = float(qobs[np.argmax(finite_qobs)])
        n = len(subset)

        if self.ec.use_stochastic:
            n_traj = int(self.ec.stochastic.get("n_trajectories", 2000))
            rng = child_rng(self.ec.seed, offset)
            ens = run_monte_carlo(
                mu=bp["mu"], lambda_=bp["lambda"], sigma=bp["sigma"],
                peff=peff, q0=q0,
                alpha_levy=bp["alpha"], beta_levy=bp["beta"],
                alpha_area=self.cal_result.alpha_area,
                n_traj=n_traj, rng=rng,
                use_baseflow=self.ec.use_baseflow,
                baseflow_params=self.baseflow_params, q0_obs=q0,
            )
            s = ens.summary
            return _SimBundle(
                q_total=s["Qmean"], q_base=ens.q_base, summary=s,
                lower=s["q025"], upper=s["q975"],
            )

        # Deterministico (E0/E1): Qfast sin ruido + baseflow opcional.
        q0_fast = quickflow_initial_from_total(
            q0, self.baseflow_params, use_baseflow=self.ec.use_baseflow
        )
        q_fast = simulate_fast_deterministic(
            mu=bp["mu"], lambda_=bp["lambda"], peff=peff, q0=q0_fast,
            alpha_area=self.cal_result.alpha_area,
        )
        q_total, q_base = assemble_total_discharge(
            q_fast, peff, use_baseflow=self.ec.use_baseflow,
            baseflow_params=self.baseflow_params, q0_obs=q0,
            alpha_area=self.cal_result.alpha_area,
        )
        summary = {"Qmean": q_total, "Qbase": q_base, "Qfast": q_fast}
        return _SimBundle(q_total=q_total, q_base=q_base, summary=summary)

    def _slice_sim_bundle(self, sim: _SimBundle, start: int, stop: int) -> _SimBundle:
        """Recorta una simulacion continua preservando resumen y bandas."""
        summary = {
            k: (np.asarray(v)[start:stop] if np.ndim(v) == 1 and len(v) == len(sim.q_total) else v)
            for k, v in sim.summary.items()
        }
        lower = None if sim.lower is None else np.asarray(sim.lower)[start:stop]
        upper = None if sim.upper is None else np.asarray(sim.upper)[start:stop]
        return _SimBundle(
            q_total=np.asarray(sim.q_total)[start:stop],
            q_base=np.asarray(sim.q_base)[start:stop],
            summary=summary,
            lower=lower,
            upper=upper,
        )

    def run_ramis_continuous(self, train: pd.DataFrame, val: pd.DataFrame) -> Tuple[_SimBundle, _SimBundle]:
        """Ejecuta RAMIS una sola vez sobre train+validation y luego separa.

        Esto evita reiniciar la validacion con el primer Qobs de validacion y
        mantiene la memoria hidrologica de RAMIS/baseflow desde calibracion
        hacia validacion. El unico Qobs usado como condicion inicial es el
        primer Qobs finito de calibracion.
        """
        full = pd.concat([train, val], axis=0, ignore_index=True)
        qobs_train = train["Qobs"].to_numpy(dtype=float)
        if not np.isfinite(qobs_train).any():
            raise ValueError(f"{self.experiment_id}: train sin Qobs finito para inicializar RAMIS.")
        q0_train = float(qobs_train[np.argmax(np.isfinite(qobs_train))])
        _log.info(
            "RAMIS estado continuo train+validation: q0 tomado de train=%.4f; "
            "no se usa Qobs de validacion para inicializar.",
            q0_train,
        )
        sim_full = self.run_ramis(full, offset=999)
        n_train = len(train)
        sim_train = self._slice_sim_bundle(sim_full, 0, n_train)
        sim_val = self._slice_sim_bundle(sim_full, n_train, len(full))
        return sim_train, sim_val

    # ----------------------------------------------------------- feature frames
    def _feature_frame(self, subset: pd.DataFrame, sim: Optional[_SimBundle]) -> pd.DataFrame:
        """Construye el DataFrame de features segun la fuente declarada."""
        source = self.ec.feature_spec.get("source", "observed_forcing")
        n = len(subset)
        if source == "observed_forcing":
            return pd.DataFrame({
                "P": subset["P"].to_numpy(),
                "PET": subset["PET"].to_numpy(),
                "Tmin": subset["Tmin"].to_numpy(),
                "Tmax": subset["Tmax"].to_numpy(),
            })
        if source == "ramis_deterministic":
            return pd.DataFrame({"Qdet_BF": sim.q_total})
        if source == "stochastic_ensemble":
            frame = ensemble_summary_to_frame(sim.summary, n=n)
            frame["Qmean"] = sim.summary["Qmean"]
            return frame
        raise ValueError(f"Fuente de features no soportada: {source}")

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        self._setup_outputs()
        _log.info("== Ejecutando %s : %s ==", self.experiment_id, self.ec.description)

        data = self.load_and_prepare_data()
        data = drop_invalid_rows(
            data,
            required_columns=["date", "P", "Tmin", "Tmax", "PET", "Peff"],
            require_qobs=False,
            label="full_forcing",
        )
        train, val = temporal_split(data, float(self.ec.glob.get("train_fraction", 0.7)))
        checks.check_temporal_split(train["date"], val["date"])
        train = trim_to_first_valid_qobs(train, label="train")
        val = trim_to_first_valid_qobs(val, label="validation")
        checks.check_daily_continuity(train["date"])
        checks.check_daily_continuity(val["date"])
        checks.check_temporal_split(train["date"], val["date"])
        _log.info(
            "Qobs finitos: train=%d/%d, validation=%d/%d",
            int(np.isfinite(train["Qobs"]).sum()), len(train),
            int(np.isfinite(val["Qobs"]).sum()), len(val),
        )

        mt = self.ec.model_type
        sim_train = sim_val = None
        if mt in {"physical", "stochastic_physical", "hybrid", "hybrid_sequence"}:
            self.calibrate_params(train)
            state_mode = self.ec.glob.get("ramis_state_mode", "continuous_train_validation")
            if state_mode == "continuous_train_validation":
                sim_train, sim_val = self.run_ramis_continuous(train, val)
            elif state_mode == "reset_each_subset":
                _log.warning(
                    "RAMIS state_mode=reset_each_subset: la validacion se inicializa "
                    "con su primer Qobs. Usar solo para comparaciones heredadas."
                )
                sim_train = self.run_ramis(train, offset=1)
                sim_val = self.run_ramis(val, offset=999)
            else:
                raise ValueError(f"ramis_state_mode no soportado: {state_mode}")

        if mt in {"machine_learning", "hybrid", "hybrid_sequence"}:
            pred_train, pred_val = self._run_ml(train, val, sim_train, sim_val)
        else:
            pred_train = sim_train.q_total
            pred_val = sim_val.q_total

        result = self._evaluate_and_save(train, val, pred_train, pred_val, sim_train, sim_val)
        _log.info("== %s completado ==", self.experiment_id)
        return result

    # ------------------------------------------------------------------- ML
    def _run_ml(self, train, val, sim_train, sim_val) -> Tuple[np.ndarray, np.ndarray]:
        horizon = int(self.ec.glob.get("forecast_horizon", 1))
        lags = self.ec.lags
        variables = self.ec.feature_spec.get("variables", [])

        ftrain = self._feature_frame(train, sim_train)
        fval = self._feature_frame(val, sim_val)
        avail = list(ftrain.columns)
        cols = select_feature_columns(variables, avail)

        y_train = train["Qobs"].reset_index(drop=True)
        y_val = val["Qobs"].reset_index(drop=True)

        model = build_model(self.ec.ml_model, self.ec.model_type, self.ec.seed)

        if self.ec.model_type == "hybrid_sequence":
            seq = self.ec.sequence_length
            Xtr, ytr, idx_tr = build_sequences(ftrain, y_train, cols, seq, horizon)
            Xv, yv, idx_v = build_sequences(fval, y_val, cols, seq, horizon)
            checks.check_lags_no_future(horizon)
            ptr, pv, model = fit_predict_sequence(model, Xtr, ytr, Xv)
            self._ml_index = {"train": idx_tr + horizon, "val": idx_v + horizon,
                              "ytr": ytr, "yv": yv}
        else:
            Xtr, ytr, valid_tr = build_tabular(ftrain, y_train, cols, lags, horizon)
            Xv, yv, valid_v = build_tabular(fval, y_val, cols, lags, horizon)
            checks.check_lags_no_future(horizon)
            ptr, pv, model = fit_predict_tabular(model, Xtr.to_numpy(), ytr.to_numpy(), Xv.to_numpy())
            self._ml_index = {
                "train_y": ytr.to_numpy(),
                "val_y": yv.to_numpy(),
                "train_target_idx": valid_tr.nonzero()[0] + horizon,
                "val_target_idx": valid_v.nonzero()[0] + horizon,
            }

        self._ml_backend = getattr(model, "backend", "unknown")
        self._save_ml_backend_report(model)
        model.save(self.out_dir / "model_artifact" / "model.joblib")
        return ptr, pv

    def _save_ml_backend_report(self, model) -> None:
        """Guarda evidencia del backend ML realmente usado.

        En particular, E7/E8 pueden declararse como GRU pero caer en
        ``sklearn_mlp`` si TensorFlow/PyTorch no estan instalados. Este archivo
        evita ambiguedad al redactar resultados publicables.
        """
        backend = str(getattr(model, "backend", "unknown"))
        ml_model = str(self.ec.ml_model or "")
        if ml_model == "gru" and backend in {"tensorflow", "torch"}:
            label = "real_gru"
        elif ml_model == "gru" and backend == "sklearn_mlp":
            label = "fallback_mlp_on_flattened_sequences"
        elif ml_model:
            label = backend
        else:
            label = "physical"
        report = {
            "experiment_id": self.experiment_id,
            "model_type": self.ec.model_type,
            "ml_model": ml_model,
            "backend": backend,
            "backend_label": label,
            "is_real_gru": bool(ml_model == "gru" and backend in {"tensorflow", "torch"}),
        }
        (self.out_dir / "ml_backend.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _save_physical_components(self, subset: pd.DataFrame, sim: Optional[_SimBundle], split: str) -> None:
        """Guarda componentes RAMIS/baseflow para auditoria fisica.

        Para experimentos fisicos e hibridos escribe una tabla con Qtotal,
        Qfast, Qbase y, cuando existan, estadisticos del ensemble. Esta salida
        permite verificar que Qtotal = Qfast + Qbase y que Qsim coincide con
        el componente fisico usado como prediccion/base de features.
        """
        if sim is None:
            return
        n = len(subset)
        rows = {
            "date": subset["date"].to_numpy(),
            "Qobs": subset["Qobs"].to_numpy(dtype=float),
            "Peff": subset["Peff"].to_numpy(dtype=float),
            "Qtotal": np.asarray(sim.q_total, dtype=float),
            "Qbase": np.asarray(sim.q_base, dtype=float),
        }
        if "Qfast" in sim.summary:
            rows["Qfast"] = np.asarray(sim.summary["Qfast"], dtype=float)
        else:
            rows["Qfast"] = rows["Qtotal"] - rows["Qbase"]

        # Guarda estadisticos adicionales del ensemble, si existen y tienen la
        # misma longitud del subconjunto.
        for key, value in sim.summary.items():
            arr = np.asarray(value)
            if key in rows:
                continue
            if arr.ndim == 1 and len(arr) == n:
                rows[key] = arr.astype(float)

        pd.DataFrame(rows).to_csv(self.out_dir / f"physical_components_{split}.csv", index=False)

    # ------------------------------------------------------------- evaluation
    def _evaluate_and_save(self, train, val, pred_train, pred_val, sim_train, sim_val):
        from ..plotting.hydrographs import plot_hydrograph
        from ..plotting.scatter import plot_scatter
        from ..plotting.flow_duration import plot_flow_duration
        from ..plotting.uncertainty import plot_uncertainty_band, plot_residuals_by_regime
        from ..plotting.diagrams import plot_taylor

        mt = self.ec.model_type
        # Alinear obs/pred segun si hubo recorte por lags/secuencias.
        obs_tr, sim_tr, dates_tr = self._align_obs(train, pred_train, "train")
        obs_v, sim_v, dates_v = self._align_obs(val, pred_val, "val")

        window = resolve_evaluation_window(self.full_cfg)
        dates_tr, (obs_tr, sim_tr), win_tr = apply_common_window(
            train["date"].to_numpy(), dates_tr, (obs_tr, sim_tr), window
        )
        dates_v, (obs_v, sim_v), win_v = apply_common_window(
            val["date"].to_numpy(), dates_v, (obs_v, sim_v), window
        )
        if len(obs_v) == 0:
            raise ValueError(
                f"{self.experiment_id}: la ventana comun de evaluacion dejo validation vacio. "
                f"Revisa evaluation.common_window o la longitud de validation."
            )
        (self.out_dir / "evaluation_window.json").write_text(
            json.dumps({
                **window.to_dict(),
                "train": win_tr,
                "validation": win_v,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _log.info(
            "Ventana comun evaluacion: validation %s a %s | n=%d/%d",
            win_v.get("start_date"), win_v.get("end_date"), win_v.get("n_after"), win_v.get("n_before")
        )

        checks.check_lengths_match(obs_v, sim_v)
        sim_tr_raw = np.asarray(sim_tr, dtype=float).copy()
        sim_v_raw = np.asarray(sim_v, dtype=float).copy()
        sim_tr, sim_v, postprocess = self._postprocess_qsim(sim_tr_raw, sim_v_raw)
        (self.out_dir / "postprocessing_report.json").write_text(
            json.dumps(postprocess, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if postprocess.get("applied") and postprocess["validation"]["n_negative_raw"] > 0:
            _log.warning(
                "%s: %d predicciones ML negativas en validation fueron truncadas a cero "
                "para respetar Q>=0. Min raw=%.6f",
                self.experiment_id,
                postprocess["validation"]["n_negative_raw"],
                postprocess["validation"]["min_raw"],
            )
        checks.check_non_negative_q(sim_v)

        det_tr = all_deterministic(obs_tr, sim_tr)
        det_v = all_deterministic(obs_v, sim_v)
        pd.DataFrame([det_tr]).to_csv(self.out_dir / "metrics_train.csv", index=False)
        pd.DataFrame([det_v]).to_csv(self.out_dir / "metrics_validation.csv", index=False)

        # Umbrales de regimen definidos en CALIBRACION (no en validacion).
        p25, p75 = regime_thresholds(train["Qobs"].to_numpy())
        regime_meta = {
            "source": "train_Qobs_only",
            "p25": float(p25),
            "p75": float(p75),
            "n_train_finite_qobs": int(np.isfinite(train["Qobs"].to_numpy(dtype=float)).sum()),
            "note": "Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion.",
        }
        (self.out_dir / "regime_thresholds.json").write_text(
            json.dumps(regime_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        regime_v = metrics_by_regime(obs_v, sim_v, p25, p75)
        regime_v.insert(0, "experiment", self.experiment_id)
        regime_v.to_csv(self.out_dir / "metrics_by_regime.csv", index=False)

        # Incertidumbre: solo se reporta como uncertainty_metrics.csv cuando
        # la banda corresponde al Qsim evaluado. En hibridos, Qsim es la salida
        # ML post-procesada, mientras que la banda proviene del ensemble fisico;
        # por eso se guarda aparte como referencia fisica y no entra al ranking
        # de incertidumbre del modelo final.
        unc = {}
        lo_v = up_v = None
        if sim_val is not None and sim_val.lower is not None:
            lo_v = self._values_for_dates(val, sim_val.lower, dates_v)
            up_v = self._values_for_dates(val, sim_val.upper, dates_v)
            checks.check_quantiles_ordered(lo_v, up_v)
            band_metrics = all_uncertainty(obs_v, lo_v, up_v, alpha=0.05)
            if mt == "stochastic_physical":
                unc = band_metrics
                pd.DataFrame([unc]).to_csv(self.out_dir / "uncertainty_metrics.csv", index=False)
            else:
                pd.DataFrame([band_metrics]).to_csv(
                    self.out_dir / "physical_uncertainty_reference.csv", index=False
                )

        # Predicciones. Para ML/hibridos se conserva Qsim_raw para auditar
        # cualquier truncamiento no-negativo aplicado a la salida final.
        pred_train_rows = {"date": dates_tr, "Qobs": obs_tr, "Qsim": sim_tr}
        pred_val_rows = {"date": dates_v, "Qobs": obs_v, "Qsim": sim_v}
        if postprocess.get("applied"):
            pred_train_rows["Qsim_raw"] = sim_tr_raw
            pred_val_rows["Qsim_raw"] = sim_v_raw
        pd.DataFrame(pred_train_rows).to_csv(
            self.out_dir / "predictions_train.csv", index=False)
        pred_val_df = pd.DataFrame(pred_val_rows)
        pred_val_df.to_csv(self.out_dir / "predictions_validation.csv", index=False)

        # Componentes fisicos para auditoria Fase 2.6.
        self._save_physical_components(train, sim_train, "train")
        self._save_physical_components(val, sim_val, "validation")

        # Ensemble summary (si aplica).
        if sim_val is not None and "q05" in sim_val.summary:
            ens_cols = {k: v for k, v in sim_val.summary.items()
                        if np.ndim(v) == 1 and len(v) == len(val)}
            ens_df = pd.DataFrame(ens_cols)
            ens_df.insert(0, "date", val["date"].to_numpy())
            ens_df.to_csv(self.out_dir / "ensemble_summary.csv", index=False)

        # --- figuras ---
        precip_v = self._values_for_dates(val, val["P"].to_numpy(), dates_v)
        lo_plot, up_plot = lo_v, up_v  # ya recortados a len(obs_v)
        plot_hydrograph(obs_v, sim_v, f"{self.experiment_id} validacion",
                        self.fig_dir / "hydrograph_validation.png",
                        precip=precip_v, qinf=lo_plot, qsup=up_plot, dates=dates_v)
        plot_hydrograph(obs_tr, sim_tr, f"{self.experiment_id} calibracion",
                        self.fig_dir / "hydrograph_train.png", dates=dates_tr)
        plot_scatter(obs_v, sim_v, f"{self.experiment_id} dispersion (val)",
                     self.fig_dir / "scatter_validation.png")
        plot_flow_duration(obs_v, sim_v, f"{self.experiment_id} FDC (val)",
                           self.fig_dir / "flow_duration_curve.png")
        plot_residuals_by_regime(obs_v, sim_v, p25, p75,
                                 f"{self.experiment_id} residuos por regimen",
                                 self.fig_dir / "residuals_by_regime.png")
        plot_taylor(obs_v, sim_v, f"{self.experiment_id} Taylor (val)",
                    self.fig_dir / "taylor_diagram.png")
        if lo_plot is not None and mt == "stochastic_physical":
            plot_uncertainty_band(obs_v, sim_v, lo_plot, up_plot,
                                  f"{self.experiment_id} banda incertidumbre",
                                  self.fig_dir / "uncertainty_band.png", dates=dates_v)

        self._save_leakage_manifest(
            train=train, val=val, dates_tr=dates_tr, dates_v=dates_v,
            win_tr=win_tr, win_v=win_v, regime_meta=regime_meta,
            postprocess=postprocess, n_eval_train=len(obs_tr), n_eval_validation=len(obs_v),
        )

        return {
            "experiment": self.experiment_id,
            "description": self.ec.description,
            "model_type": mt,
            "metrics_validation": det_v,
            "metrics_train": det_tr,
            "uncertainty": unc,
            "regime": regime_v,
            "n_eval_train": int(len(obs_tr)),
            "n_eval_validation": int(len(obs_v)),
            "evaluation_window": {**window.to_dict(), "train": win_tr, "validation": win_v},
            "backend": getattr(self, "_ml_backend", "physical"),
        }

    def _save_leakage_manifest(
        self, *, train, val, dates_tr, dates_v, win_tr, win_v,
        regime_meta: Dict[str, Any], postprocess: Dict[str, Any],
        n_eval_train: int, n_eval_validation: int,
    ) -> None:
        """Guarda un manifiesto compacto para auditoria anti-leakage.

        Este archivo es deliberadamente redundante con otros outputs: facilita
        revisar cada experimento sin abrir el codigo fuente.
        """

        def _bounds(dates) -> Dict[str, Any]:
            d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
            if d.empty:
                return {"start": None, "end": None, "n": 0}
            return {
                "start": d.min().date().isoformat(),
                "end": d.max().date().isoformat(),
                "n": int(len(d)),
            }

        feature_spec = self.ec.feature_spec or {}
        ml_backend = _read_json_safe(self.out_dir / "ml_backend.json")
        target_alignment = {
            "forecast_horizon": int(self.ec.glob.get("forecast_horizon", 1)),
            "rule": "features hasta t; target Qobs en t+horizon",
            "lags": [int(x) for x in self.ec.lags],
            "sequence_length": int(self.ec.sequence_length) if self.ec.model_type == "hybrid_sequence" else None,
            "uses_validation_qobs_as_feature": False,
        }
        manifest = {
            "experiment_id": self.experiment_id,
            "model_type": self.ec.model_type,
            "ml_model": self.ec.ml_model,
            "ramis_state_mode": self.ec.glob.get("ramis_state_mode", "continuous_train_validation"),
            "raw_split_periods": {
                "train": _bounds(train["date"].to_numpy()),
                "validation": _bounds(val["date"].to_numpy()),
            },
            "evaluated_periods": {
                "train": {**_bounds(dates_tr), "n_eval": int(n_eval_train)},
                "validation": {**_bounds(dates_v), "n_eval": int(n_eval_validation)},
            },
            "evaluation_window": {"train": win_tr, "validation": win_v},
            "feature_source": feature_spec.get("source", "physical_only"),
            "feature_variables": feature_spec.get("variables", []),
            "target_alignment": target_alignment,
            "regime_thresholds": regime_meta,
            "postprocessing": postprocess,
            "ml_backend": ml_backend or {"backend": getattr(self, "_ml_backend", "physical")},
        }
        (self.out_dir / "leakage_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _postprocess_qsim(self, sim_tr_raw: np.ndarray, sim_v_raw: np.ndarray):
        """Aplica restricciones fisicas finales a predicciones ML/hibridas.

        Los modelos de aprendizaje pueden producir valores ligeramente negativos
        por extrapolacion numerica, especialmente en el fallback MLP usado por
        GRU cuando no hay TensorFlow/PyTorch. Como el caudal no puede ser
        negativo, solo para salidas ML/hibridas se trunca Qsim<0 a 0 y se
        guarda el valor crudo en ``Qsim_raw`` junto con un reporte auditable.
        Para modelos fisicos no se corrige: la validacion debe fallar si RAMIS
        produce Q<0.
        """
        mt = self.ec.model_type
        apply_clip = mt in {"machine_learning", "hybrid", "hybrid_sequence"}

        def _summary(raw: np.ndarray) -> dict:
            arr = np.asarray(raw, dtype=float)
            finite = arr[np.isfinite(arr)]
            neg = finite[finite < 0]
            return {
                "n": int(len(arr)),
                "n_negative_raw": int(len(neg)),
                "min_raw": float(np.min(finite)) if len(finite) else None,
                "max_raw": float(np.max(finite)) if len(finite) else None,
                "min_after": float(max(0.0, np.min(finite))) if apply_clip and len(finite) else (float(np.min(finite)) if len(finite) else None),
            }

        report = {
            "applied": bool(apply_clip),
            "method": "clip_negative_to_zero" if apply_clip else "none",
            "model_type": mt,
            "train": _summary(sim_tr_raw),
            "validation": _summary(sim_v_raw),
        }
        if not apply_clip:
            return sim_tr_raw, sim_v_raw, report
        return np.clip(sim_tr_raw, 0.0, None), np.clip(sim_v_raw, 0.0, None), report

    def _values_for_dates(self, subset: pd.DataFrame, values, target_dates) -> np.ndarray:
        """Extrae valores del subconjunto original para fechas ya alineadas.

        Evita usar recortes tipo ``[-m:]`` cuando ML descarta filas internas por
        NaN. Si hay fechas repetidas, conserva la primera aparicion.
        """
        idx = pd.to_datetime(subset["date"])
        ser = pd.Series(np.asarray(values, dtype=float), index=idx)
        ser = ser[~ser.index.duplicated(keep="first")]
        out = ser.reindex(pd.to_datetime(target_dates)).to_numpy(dtype=float)
        return out

    # --------------------------------------------------------- alignment utils
    def _align_obs(self, subset, pred, which) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alinea obs/pred/fechas teniendo en cuenta recortes por lags/secuencias."""
        pred = np.asarray(pred, dtype=float)
        n_sub = len(subset)
        if len(pred) == n_sub:
            return (subset["Qobs"].to_numpy(), pred, subset["date"].to_numpy())

        # Hubo recorte: las predicciones corresponden a las ultimas filas.
        if hasattr(self, "_ml_index") and "train_y" in self._ml_index:
            y = self._ml_index["train_y" if which == "train" else "val_y"]
            idx_key = "train_target_idx" if which == "train" else "val_target_idx"
            if idx_key in self._ml_index:
                idx = np.asarray(self._ml_index[idx_key], dtype=int)
                dates = subset["date"].to_numpy()[idx]
            else:
                dates = subset["date"].to_numpy()[-len(y):]
            return (np.asarray(y, dtype=float), pred[: len(y)], dates)
        if hasattr(self, "_ml_index") and "ytr" in self._ml_index:
            y = self._ml_index["ytr" if which == "train" else "yv"]
            idx = self._ml_index["train" if which == "train" else "val"]
            dates = subset["date"].to_numpy()[idx]
            return (np.asarray(y, dtype=float), pred[: len(y)], dates)
        # Fallback robusto: recortar al minimo.
        m = min(len(pred), n_sub)
        return (subset["Qobs"].to_numpy()[-m:], pred[-m:], subset["date"].to_numpy()[-m:])

    def _align_band(self, subset, sim, which):
        n = len(subset)
        lo, up = sim.lower, sim.upper
        obs, _, _ = self._align_obs(subset, sim.q_total, which)
        if len(obs) == n:
            return lo, up
        m = len(obs)
        return lo[-m:], up[-m:]
