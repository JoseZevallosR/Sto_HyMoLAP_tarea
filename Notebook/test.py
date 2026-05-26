import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.optimize import differential_evolution

from models import StoHyMoLAPConfig, simulate_mean_response
from metrics import METRIC_REGISTRY, nse, kge, rmse, pbias
from utils import add_distance_to_ideal, plot_pareto_front


# ============================================================
# SINGLE-OBJECTIVE FUNCTION
# ============================================================

def objective_abtheta(
    x: np.ndarray,
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    metric: str = "nse",
    n_members: int = 30,
    summary_method: str = "median"
) -> float:
    a, b, theta = x

    if a <= 0 or b <= 0 or theta < 0:
        return 1e6

    metric = metric.lower()
    if metric not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {list(METRIC_REGISTRY.keys())}"
        )

    params = {"a": a, "b": b, "theta": theta}

    try:
        q_sim = simulate_mean_response(
            psi=psi,
            q0=q0,
            params=params,
            base_config=base_config,
            n_members=n_members,
            summary_method=summary_method
        )
    except Exception:
        return 1e6

    if not np.all(np.isfinite(q_sim)):
        return 1e6

    metric_func = METRIC_REGISTRY[metric]["func"]
    sense = METRIC_REGISTRY[metric]["sense"]

    val = metric_func(q_obs, q_sim)

    if not np.isfinite(val):
        return 1e6

    if sense == "max":
        return -val
    if sense == "min":
        return val

    raise ValueError(f"Invalid sense for metric '{metric}'")


# ============================================================
# MULTIOBJECTIVE EVALUATION
# ============================================================

def evaluate_metrics(
    x: np.ndarray,
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    metrics: List[str],
    n_members: int = 30,
    summary_method: str = "median"
) -> Dict[str, float]:
    metrics = [m.lower() for m in metrics]
    a, b, theta = x

    result: Dict[str, float] = {
        "a": np.nan,
        "b": np.nan,
        "theta": np.nan
    }

    for metric in metrics:
        result[metric] = np.nan
        result[f"f_{metric}"] = np.inf

    if a <= 0 or b <= 0 or theta < 0:
        return result

    result["a"] = a
    result["b"] = b
    result["theta"] = theta

    params = {"a": a, "b": b, "theta": theta}

    try:
        q_sim = simulate_mean_response(
            psi=psi,
            q0=q0,
            params=params,
            base_config=base_config,
            n_members=n_members,
            summary_method=summary_method
        )
    except Exception:
        return result

    if not np.all(np.isfinite(q_sim)):
        return result

    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{metric}'. Available: {list(METRIC_REGISTRY.keys())}"
            )

        metric_func = METRIC_REGISTRY[metric]["func"]
        sense = METRIC_REGISTRY[metric]["sense"]

        val = metric_func(q_obs, q_sim)

        if not np.isfinite(val):
            val = np.nan
            f_val = np.inf
        else:
            if sense == "max":
                f_val = 1.0 - val
            elif sense == "min":
                f_val = val
            else:
                raise ValueError(f"Invalid sense for metric '{metric}'")

        result[metric] = val
        result[f"f_{metric}"] = f_val

    return result


def objective_weighted_metrics(
    x: np.ndarray,
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    metrics: List[str],
    weights: List[float],
    n_members: int = 30,
    summary_method: str = "median"
) -> float:
    metrics = [m.lower() for m in metrics]

    if len(metrics) != len(weights):
        raise ValueError("metrics and weights must have the same length")

    weights_arr = np.asarray(weights, dtype=float)

    if np.any(weights_arr < 0):
        raise ValueError("weights must be non-negative")

    if np.sum(weights_arr) == 0:
        raise ValueError("sum of weights must be > 0")

    weights_arr = weights_arr / np.sum(weights_arr)

    res = evaluate_metrics(
        x=x,
        psi=psi,
        q_obs=q_obs,
        q0=q0,
        base_config=base_config,
        metrics=metrics,
        n_members=n_members,
        summary_method=summary_method
    )

    f_vals = np.array([res[f"f_{m}"] for m in metrics], dtype=float)

    if not np.all(np.isfinite(f_vals)):
        return 1e6

    return float(np.sum(weights_arr * f_vals))


# ============================================================
# PARETO UTILITIES
# ============================================================

def dominates_general(row_i: Dict[str, float], row_j: Dict[str, float], objective_cols: List[str]) -> bool:
    vals_i = np.array([row_i[col] for col in objective_cols], dtype=float)
    vals_j = np.array([row_j[col] for col in objective_cols], dtype=float)

    return np.all(vals_i <= vals_j) and np.any(vals_i < vals_j)


def extract_pareto_front_general(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    metrics = [m.lower() for m in metrics]
    objective_cols = [f"f_{m}" for m in metrics]
    is_nondominated = np.ones(len(df), dtype=bool)
    rows = df.to_dict("records")

    for i in range(len(rows)):
        if not is_nondominated[i]:
            continue
        for j in range(len(rows)):
            if i == j:
                continue
            if dominates_general(rows[j], rows[i], objective_cols):
                is_nondominated[i] = False
                break

    pareto_df = df.loc[is_nondominated].copy()
    pareto_df = pareto_df.reset_index(drop=True)
    return pareto_df


def calibrate_pareto_by_weight_scan(
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    bounds_abtheta: List[Tuple[float, float]],
    metrics: List[str],
    weight_sets: List[List[float]],
    n_members_calibration: int = 30,
    n_members_refit: int = 100,
    summary_method: str = "median",
    maxiter: int = 25,
    popsize: int = 12,
    seed: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [m.lower() for m in metrics]
    all_solutions = []

    for weights in weight_sets:
        weights_arr = np.asarray(weights, dtype=float)

        if len(weights_arr) != len(metrics):
            raise ValueError("Each weight vector must have the same length as metrics")

        if np.any(weights_arr < 0):
            raise ValueError("Weights must be non-negative")

        if np.sum(weights_arr) == 0:
            raise ValueError("The sum of weights must be > 0")

        weights_arr = weights_arr / np.sum(weights_arr)

        weight_msg = ", ".join(
            [f"{m}={w:.2f}" for m, w in zip(metrics, weights_arr)]
        )
        print(f"\nCalibrating for weights: {weight_msg}")

        result = differential_evolution(
            objective_weighted_metrics,
            bounds=bounds_abtheta,
            args=(
                psi,
                q_obs,
                q0,
                base_config,
                metrics,
                weights_arr.tolist(),
                n_members_calibration,
                summary_method
            ),
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            polish=True,
            workers=-1,
            updating="deferred"
        )

        x_best = result.x

        res = evaluate_metrics(
            x=x_best,
            psi=psi,
            q_obs=q_obs,
            q0=q0,
            base_config=base_config,
            metrics=metrics,
            n_members=n_members_refit,
            summary_method=summary_method
        )

        for m, w in zip(metrics, weights_arr):
            res[f"w_{m}"] = float(w)

        res["weighted_obj"] = float(result.fun)
        all_solutions.append(res)

    all_df = pd.DataFrame(all_solutions)
    pareto_df = extract_pareto_front_general(all_df, metrics=metrics)

    return all_df, pareto_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # 1. Read data
    # --------------------------------------------------------
    df = pd.read_csv("../data.csv", encoding="utf-8", sep=";")

    df["psi"] = (df["Precipitation"] - df["PET"]).clip(lower=0)

    psi_obs = df["psi"].to_numpy(dtype=float)
    q_obs = df["Qobs"].to_numpy(dtype=float)

    mask = np.isfinite(psi_obs) & np.isfinite(q_obs)
    psi_obs = psi_obs[mask]
    q_obs = q_obs[mask]

    if len(q_obs) == 0:
        raise ValueError("No valid observations found after filtering NaNs.")

    q0 = q_obs[0]

    # --------------------------------------------------------
    # 2. Base configuration
    # --------------------------------------------------------
    base_cfg = StoHyMoLAPConfig(
        a=1.0,
        b=20.0,
        theta=0.02,
        dt=1.0,
        noise_type="gaussian",   # change to "gaussian" or "levy"
        alpha_levy=1.8,
        beta_levy=0.0,
        levy_scale=0.01,
        q_min=1e-8,
        n_substeps=8,
        levy_clip=5.0,
        random_seed=123
    )

    bounds_abtheta = [
        (0.6, 2.0),     # a
        (1.0, 40.0),    # b
        (0.001, 0.05)   # theta
    ]

    # --------------------------------------------------------
    # 3. General multiobjective calibration
    # --------------------------------------------------------
    print("Starting multiobjective calibration...")

    metrics = ["nse", "kge", "rmse"]

    weight_sets = [
        [0.60, 0.20, 0.20],
        [0.50, 0.25, 0.25],
        [0.40, 0.30, 0.30],
        [0.34, 0.33, 0.33],
        [0.20, 0.40, 0.40],
        [0.20, 0.60, 0.20],
        [0.70, 0.15, 0.15]
    ]

    all_df, pareto_df = calibrate_pareto_by_weight_scan(
        psi=psi_obs,
        q_obs=q_obs,
        q0=q0,
        base_config=base_cfg,
        bounds_abtheta=bounds_abtheta,
        metrics=metrics,
        weight_sets=weight_sets,
        n_members_calibration=40,
        n_members_refit=100,
        summary_method="median",
        maxiter=25,
        popsize=12,
        seed=123
    )

    print("\nAll candidate solutions:")
    cols_show = ["a", "b", "theta"] + metrics + [f"w_{m}" for m in metrics] + ["weighted_obj"]
    print(all_df[cols_show])

    print("\nPareto front:")
    print(pareto_df[cols_show])

    # --------------------------------------------------------
    # 4. Choose best compromise from Pareto front
    # --------------------------------------------------------
    pareto_df = add_distance_to_ideal(pareto_df, metrics=metrics)
    best_compromise = pareto_df.loc[pareto_df["distance_to_ideal"].idxmin()]

    print("\nBest compromise solution:")
    print(best_compromise[["a", "b", "theta"] + metrics + ["distance_to_ideal"]])

    best_a = float(best_compromise["a"])
    best_b = float(best_compromise["b"])
    best_theta = float(best_compromise["theta"])

    # --------------------------------------------------------
    # 5. Final simulation
    # --------------------------------------------------------
    q_cal_pareto = simulate_mean_response(
        psi=psi_obs,
        q0=q0,
        params={"a": best_a, "b": best_b, "theta": best_theta},
        base_config=base_cfg,
        n_members=100,
        summary_method="median"
    )

    # --------------------------------------------------------
    # 6. Pareto plot
    # --------------------------------------------------------
    # Solo funciona si metrics tiene exactamente 2 objetivos
    # plot_pareto_front(all_df, pareto_df, metrics=metrics)

    # --------------------------------------------------------
    # 7. Final observed vs simulated plot
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(q_obs, label="Observed", color="black", alpha=0.6)
    plt.plot(q_cal_pareto, label="Best Pareto compromise", color="red", linewidth=1.5)
    plt.legend()
    plt.title(f"Stochastic Calibration Results ({base_cfg.noise_type.capitalize()} noise)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 8. Final metrics
    # --------------------------------------------------------
    print("\nFinal performance of best Pareto compromise:")
    print(f"NSE   = {nse(q_obs, q_cal_pareto):.4f}")
    print(f"KGE   = {kge(q_obs, q_cal_pareto):.4f}")
    print(f"RMSE  = {rmse(q_obs, q_cal_pareto):.4f}")
    print(f"PBIAS = {pbias(q_obs, q_cal_pareto):.2f}")