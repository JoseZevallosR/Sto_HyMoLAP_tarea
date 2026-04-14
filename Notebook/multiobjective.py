import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Dict
from scipy.stats import levy_stable
from scipy.optimize import differential_evolution


# ============================================================
# METRICS
# ============================================================

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan

    return 1.0 - np.sum((obs - sim) ** 2) / denom


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    return np.sqrt(np.mean((obs - sim) ** 2))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        return np.nan

    denom = np.sum(obs)
    if denom == 0:
        return np.nan

    return 100.0 * np.sum(sim - obs) / denom


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 2:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]

    s_obs = np.std(obs, ddof=1)
    s_sim = np.std(sim, ddof=1)
    m_obs = np.mean(obs)
    m_sim = np.mean(sim)

    alpha = s_sim / s_obs if s_obs > 0 else np.nan
    beta = m_sim / m_obs if m_obs != 0 else np.nan

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class StoHyMoLAPConfig:
    a: float = 1.0
    b: float = 10.0
    theta: float = 0.05
    dt: float = 1.0

    noise_type: str = "gaussian"   # "gaussian" or "levy"

    # Lévy parameters
    alpha_levy: float = 1.8
    beta_levy: float = 0.0
    levy_scale: float = 0.03

    # Stability controls
    q_min: float = 1e-8
    n_substeps: int = 8
    levy_clip: float = 8.0

    random_seed: Optional[int] = 42


# ============================================================
# MODEL
# ============================================================

class StoHyMoLAP:
    """
    Stochastic HyMoLAP model

    dQ = [-(a/b) Q^(2a-1) + (1/b) psi(t)] dt + theta * Q * dL
    """

    def __init__(self, config: StoHyMoLAPConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def forcing(self, psi: np.ndarray, t_idx: int) -> float:
        return float(psi[t_idx])

    def drift(self, q: float, psi_t: float) -> float:
        a = self.config.a
        b = self.config.b
        q_safe = max(q, self.config.q_min)
        return -(a / b) * (q_safe ** (2 * a - 1)) + (1.0 / b) * psi_t

    def noise_increment(self, dt_sub: float) -> float:
        noise_type = self.config.noise_type.lower()

        if noise_type == "gaussian":
            return np.sqrt(dt_sub) * self.rng.normal(0.0, 1.0)

        if noise_type == "levy":
            alpha = self.config.alpha_levy
            beta = self.config.beta_levy
            scale = self.config.levy_scale * (dt_sub ** (1.0 / alpha))

            raw_inc = levy_stable.rvs(
                alpha=alpha,
                beta=beta,
                loc=0.0,
                scale=scale,
                random_state=self.rng
            )

            inc = np.clip(raw_inc, -self.config.levy_clip, self.config.levy_clip)

            if abs(raw_inc) > 5:
                print(f"⚠️ Large raw Levy jump: {raw_inc} -> clipped to {inc}")

            return inc

        raise ValueError("noise_type must be 'gaussian' or 'levy'")

    def step(self, q: float, psi_t: float, t: int = None) -> float:
        n_sub = self.config.n_substeps
        dt_sub = self.config.dt / n_sub
        theta = self.config.theta

        q_new = max(q, self.config.q_min)

        for _ in range(n_sub):
            drift_term = self.drift(q_new, psi_t) * dt_sub
            diffusion_term = theta * q_new * self.noise_increment(dt_sub)

            q_new = q_new + drift_term + diffusion_term
            # 🔥 BLOW-UP CHECK (ADD THIS)
            if not np.isfinite(q_new) or q_new > 1e6:
                print(f"⚠️ Blow-up at t={t}")
                print(f"q={q_new}, drift={drift_term}, diff={diffusion_term}")
                raise RuntimeError("Simulation exploded")

            q_new = max(q_new, self.config.q_min)

        return q_new

    def simulate(self, psi: np.ndarray, q0: float) -> np.ndarray:
        psi = np.asarray(psi, dtype=float)
        q = np.empty(len(psi), dtype=float)
        q[0] = max(float(q0), self.config.q_min)

        for t in range(len(psi) - 1):
            psi_t = self.forcing(psi, t)
            q[t + 1] = self.step(q[t], psi_t, t=t)

        return q

    def simulate_ensemble(self, psi: np.ndarray, q0: float, n_members: int = 100) -> np.ndarray:
        out = []
        for i in range(n_members):
            # different seed per member
            member_cfg = StoHyMoLAPConfig(
                a=self.config.a,
                b=self.config.b,
                theta=self.config.theta,
                dt=self.config.dt,
                noise_type=self.config.noise_type,
                alpha_levy=self.config.alpha_levy,
                beta_levy=self.config.beta_levy,
                levy_scale=self.config.levy_scale,
                q_min=self.config.q_min,
                n_substeps=self.config.n_substeps,
                levy_clip=self.config.levy_clip,
                random_seed=(self.config.random_seed + i) if self.config.random_seed is not None else None
            )
            member_model = StoHyMoLAP(member_cfg)
            out.append(member_model.simulate(psi=psi, q0=q0))

        return np.vstack(out)


# ============================================================
# ROBUST ENSEMBLE SUMMARY
# ============================================================

def robust_ensemble_summary(ens: np.ndarray, method: str = "median", trim_fraction: float = 0.1) -> np.ndarray:
    ens = np.asarray(ens, dtype=float)

    if method == "mean":
        return np.mean(ens, axis=0)

    if method == "median":
        return np.median(ens, axis=0)

    if method == "trimmed_mean":
        n = ens.shape[0]
        k = int(np.floor(trim_fraction * n))
        ens_sorted = np.sort(ens, axis=0)
        if 2 * k >= n:
            return np.mean(ens_sorted, axis=0)
        return np.mean(ens_sorted[k:n-k, :], axis=0)

    raise ValueError("method must be 'mean', 'median', or 'trimmed_mean'")


def simulate_mean_response(
    psi: np.ndarray,
    q0: float,
    params: Dict[str, float],
    base_config: StoHyMoLAPConfig,
    n_members: int = 30,
    summary_method: str = "median"
) -> np.ndarray:
    cfg = StoHyMoLAPConfig(
        a=params.get("a", base_config.a),
        b=params.get("b", base_config.b),
        theta=params.get("theta", base_config.theta),
        dt=base_config.dt,
        noise_type=base_config.noise_type,
        alpha_levy=base_config.alpha_levy,
        beta_levy=base_config.beta_levy,
        levy_scale=base_config.levy_scale,
        q_min=base_config.q_min,
        n_substeps=base_config.n_substeps,
        levy_clip=base_config.levy_clip,
        random_seed=base_config.random_seed
    )

    model = StoHyMoLAP(cfg)
    ens = model.simulate_ensemble(psi=psi, q0=q0, n_members=n_members)

    return robust_ensemble_summary(ens, method=summary_method)


# ============================================================
# OBJECTIVE FUNCTION
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

    # hard safety
    if a <= 0 or b <= 0 or theta < 0:
        return 1e6

    params = {"a": a, "b": b, "theta": theta}

    q_sim = simulate_mean_response(
        psi=psi,
        q0=q0,
        params=params,
        base_config=base_config,
        n_members=n_members,
        summary_method=summary_method
    )

    if not np.all(np.isfinite(q_sim)):
        return 1e6

    metric = metric.lower()

    if metric == "nse":
        val = nse(q_obs, q_sim)
        return 1e6 if not np.isfinite(val) else -val

    if metric == "rmse":
        val = rmse(q_obs, q_sim)
        return 1e6 if not np.isfinite(val) else val

    if metric == "kge":
        val = kge(q_obs, q_sim)
        return 1e6 if not np.isfinite(val) else -val

    raise ValueError("metric must be 'nse', 'rmse', or 'kge'")

# ============================================================
# MULTIOBJECTIVE UTILITIES
# ============================================================

def evaluate_nse_kge(
    x: np.ndarray,
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    n_members: int = 30,
    summary_method: str = "median"
) -> Dict[str, float]:
    a, b, theta = x

    if a <= 0 or b <= 0 or theta < 0:
        return {
            "a": np.nan, "b": np.nan, "theta": np.nan,
            "nse": -np.inf, "kge": -np.inf,
            "f1": np.inf, "f2": np.inf
        }

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
        return {
            "a": a, "b": b, "theta": theta,
            "nse": -np.inf, "kge": -np.inf,
            "f1": np.inf, "f2": np.inf
        }

    if not np.all(np.isfinite(q_sim)):
        return {
            "a": a, "b": b, "theta": theta,
            "nse": -np.inf, "kge": -np.inf,
            "f1": np.inf, "f2": np.inf
        }

    nse_val = nse(q_obs, q_sim)
    kge_val = kge(q_obs, q_sim)

    if not np.isfinite(nse_val):
        nse_val = -np.inf
    if not np.isfinite(kge_val):
        kge_val = -np.inf

    # minimization form
    f1 = np.inf if not np.isfinite(nse_val) else (1.0 - nse_val)
    f2 = np.inf if not np.isfinite(kge_val) else (1.0 - kge_val)

    return {
        "a": a,
        "b": b,
        "theta": theta,
        "nse": nse_val,
        "kge": kge_val,
        "f1": f1,
        "f2": f2
    }

def objective_weighted_nse_kge(
    x: np.ndarray,
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    w_nse: float = 0.5,
    n_members: int = 30,
    summary_method: str = "median"
) -> float:
    res = evaluate_nse_kge(
        x=x,
        psi=psi,
        q_obs=q_obs,
        q0=q0,
        base_config=base_config,
        n_members=n_members,
        summary_method=summary_method
    )

    if not np.isfinite(res["f1"]) or not np.isfinite(res["f2"]):
        return 1e6

    w_kge = 1.0 - w_nse
    return w_nse * res["f1"] + w_kge * res["f2"]

def dominates(row_i, row_j) -> bool:
    return (
        (row_i["f1"] <= row_j["f1"]) and
        (row_i["f2"] <= row_j["f2"]) and
        ((row_i["f1"] < row_j["f1"]) or (row_i["f2"] < row_j["f2"]))
    )


def extract_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    is_nondominated = np.ones(len(df), dtype=bool)

    rows = df.to_dict("records")
    for i in range(len(rows)):
        if not is_nondominated[i]:
            continue
        for j in range(len(rows)):
            if i == j:
                continue
            if dominates(rows[j], rows[i]):
                is_nondominated[i] = False
                break

    pareto_df = df.loc[is_nondominated].copy()
    pareto_df = pareto_df.sort_values(["f1", "f2"]).reset_index(drop=True)
    return pareto_df

def calibrate_pareto_by_weight_scan(
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    bounds_abtheta,
    weights=None,
    n_members_calibration: int = 30,
    n_members_refit: int = 100,
    summary_method: str = "median",
    maxiter: int = 25,
    popsize: int = 12
) -> (pd.DataFrame, pd.DataFrame):

    if weights is None:
        weights = np.linspace(0.0, 1.0, 11)

    all_solutions = []

    for w in weights:
        print(f"\nCalibrating for weight NSE={w:.2f}, KGE={1.0-w:.2f}")

        result = differential_evolution(
            objective_weighted_nse_kge,
            bounds=bounds_abtheta,
            args=(psi, q_obs, q0, base_config, w, n_members_calibration, summary_method),
            seed=123,
            maxiter=maxiter,
            popsize=popsize,
            polish=True,
            workers=-1,               # safer with stochastic model
            updating="deferred"
        )

        x_best = result.x

        # re-evaluate with a larger ensemble for stability
        res = evaluate_nse_kge(
            x=x_best,
            psi=psi,
            q_obs=q_obs,
            q0=q0,
            base_config=base_config,
            n_members=n_members_refit,
            summary_method=summary_method
        )

        res["w_nse"] = w
        res["w_kge"] = 1.0 - w
        res["weighted_obj"] = result.fun

        all_solutions.append(res)

    all_df = pd.DataFrame(all_solutions)
    pareto_df = extract_pareto_front(all_df)

    return all_df, pareto_df

def plot_pareto_front(all_df: pd.DataFrame, pareto_df: pd.DataFrame):
    plt.figure(figsize=(7, 5))

    plt.scatter(
        all_df["nse"], all_df["kge"],
        alpha=0.7,
        label="Weighted solutions"
    )

    plt.scatter(
        pareto_df["nse"], pareto_df["kge"],
        s=80,
        label="Pareto front"
    )

    # connect Pareto points ordered by NSE
    pareto_sorted = pareto_df.sort_values("nse")
    plt.plot(pareto_sorted["nse"], pareto_sorted["kge"], linewidth=1)

    for _, row in pareto_sorted.iterrows():
        plt.annotate(
            f"w={row['w_nse']:.1f}",
            (row["nse"], row["kge"]),
            fontsize=8
        )

    plt.xlabel("NSE")
    plt.ylabel("KGE")
    plt.title("Pareto front: NSE vs KGE")
    plt.legend()
    plt.tight_layout()
    plt.show()
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

    q0 = q_obs[0]

    # --------------------------------------------------------
    # 2. Base configuration
    # --------------------------------------------------------
    base_cfg = StoHyMoLAPConfig(
        a=1.0,
        b=20.0,
        theta=0.02,
        dt=1.0,
        noise_type="gaussian",      # change to "gaussian" or "levy"
        alpha_levy=1.8,
        beta_levy=0.0,
        levy_scale=0.01,
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
    # 3. Multiobjective calibration: NSE + KGE
    # --------------------------------------------------------
    print("Starting multiobjective calibration...")

    all_df, pareto_df = calibrate_pareto_by_weight_scan(
        psi=psi_obs,
        q_obs=q_obs,
        q0=q0,
        base_config=base_cfg,
        bounds_abtheta=bounds_abtheta,
        weights=np.linspace(0.0, 1.0, 11),
        n_members_calibration=40,
        n_members_refit=100,
        summary_method="median",
        maxiter=25,
        popsize=12
    )

    print("\nAll candidate solutions:")
    print(all_df[["w_nse", "w_kge", "a", "b", "theta", "nse", "kge"]])

    print("\nPareto front:")
    print(pareto_df[["w_nse", "w_kge", "a", "b", "theta", "nse", "kge"]])

    # --------------------------------------------------------
    # 4. Choose best compromise from Pareto front
    # --------------------------------------------------------
    pareto_df = pareto_df.copy()
    pareto_df["distance_to_ideal"] = np.sqrt(
        (1.0 - pareto_df["nse"])**2 +
        (1.0 - pareto_df["kge"])**2
    )

    best_compromise = pareto_df.loc[pareto_df["distance_to_ideal"].idxmin()]

    print("\nBest compromise solution:")
    print(best_compromise[["a", "b", "theta", "nse", "kge", "distance_to_ideal"]])

    best_a = float(best_compromise["a"])
    best_b = float(best_compromise["b"])
    best_theta = float(best_compromise["theta"])

    # final simulation using the compromise solution
    q_cal_pareto = simulate_mean_response(
        psi=psi_obs,
        q0=q0,
        params={"a": best_a, "b": best_b, "theta": best_theta},
        base_config=base_cfg,
        n_members=100,
        summary_method="median"
    )

    # --------------------------------------------------------
    # 5. Pareto plot
    # --------------------------------------------------------
    plot_pareto_front(all_df, pareto_df)

    # --------------------------------------------------------
    # 6. Final observed vs simulated plot
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(q_obs, label="Observed", color="black", alpha=0.6)
    plt.plot(q_cal_pareto, label="Best Pareto compromise", color="red", linewidth=1.5)
    plt.legend()
    plt.title(f"Stochastic Calibration Results ({base_cfg.noise_type.capitalize()} noise)")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # 7. Final metrics
    # --------------------------------------------------------
    print("\nFinal performance of best Pareto compromise:")
    print(f"NSE   = {nse(q_obs, q_cal_pareto):.4f}")
    print(f"KGE   = {kge(q_obs, q_cal_pareto):.4f}")
    print(f"RMSE  = {rmse(q_obs, q_cal_pareto):.4f}")
    print(f"PBIAS = {pbias(q_obs, q_cal_pareto):.2f}")