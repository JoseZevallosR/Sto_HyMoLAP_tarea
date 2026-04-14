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
        levy_scale=0.01,        # much smaller than 1.0
        n_substeps=8,
        levy_clip=5.0,
        random_seed=123
    )

    # Tighter bounds for Lévy case
    bounds_abtheta = [
        (0.6, 2.0),     # a
        (1.0, 40.0),    # b
        (0.001, 0.05)   # theta
    ]

    # --------------------------------------------------------
    # 3. Calibration
    # --------------------------------------------------------
    print("Starting optimization...")

    result_abtheta = differential_evolution(
        objective_abtheta,
        bounds=bounds_abtheta,
        args=(psi_obs, q_obs, q0, base_cfg, "kge", 40, "median"),
        seed=123,
        maxiter=25,
        popsize=12,
        polish=True,
        workers=-1,              # safer first; later test workers=-1
        updating="deferred"
    )

    best_a2, best_b2, best_theta2 = result_abtheta.x

    q_cal_abtheta = simulate_mean_response(
        psi_obs,
        q0,
        {"a": best_a2, "b": best_b2, "theta": best_theta2},
        base_cfg,
        n_members=100,
        summary_method="median"
    )

    print(f"\nBest Parameters:")
    print(f"a     = {best_a2:.6f}")
    print(f"b     = {best_b2:.6f}")
    print(f"theta = {best_theta2:.6f}")

    print(f"\nPerformance:")
    print(f"NSE   = {nse(q_obs, q_cal_abtheta):.4f}")
    print(f"KGE   = {kge(q_obs, q_cal_abtheta):.4f}")
    print(f"RMSE  = {rmse(q_obs, q_cal_abtheta):.4f}")
    print(f"PBIAS = {pbias(q_obs, q_cal_abtheta):.2f}")

    # --------------------------------------------------------
    # 4. Plot
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(q_obs, label="Observed", color="black", alpha=0.6)
    plt.plot(q_cal_abtheta, label="Calibrated Summary", color="red", linewidth=1.5)
    plt.legend()
    plt.title(f"Stochastic Calibration Results ({base_cfg.noise_type.capitalize()} noise)")
    plt.tight_layout()
    plt.show()
