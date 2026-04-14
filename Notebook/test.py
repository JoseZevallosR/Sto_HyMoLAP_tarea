import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from scipy.stats import levy_stable
from scipy.optimize import differential_evolution

# --- METRICS (Defined here to ensure the script runs) ---

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    denom = np.sum((obs - np.mean(obs))**2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim)**2) / denom


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    return np.sqrt(np.mean((obs - sim)**2))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
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
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1) if np.std(obs, ddof=1) > 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    return 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
# --- MODEL CLASSES ---

@dataclass
class StoHyMoLAPConfig:
    a: float = 1.0
    b: float = 10.0
    theta: float = 0.05
    dt: float = 1.0
    noise_type: str = "gaussian"   # "gaussian" or "levy"
    alpha_levy: float = 1.7        # stability parameter for Levy noise
    beta_levy: float = 0.0         # skewness parameter
    levy_scale: float = 1.0
    q_min: float = 1e-8            # positivity floor
    random_seed: Optional[int] = 42


class StoHyMoLAP:
    """
    Stochastic HyMoLAP model.

    SDE:
        dQ = [-(a/b) Q^(2a-1) + (1/b) psi(t)] dt + theta * Q * dL

    where dL may be Gaussian or Levy-stable.
    """

    def __init__(self, config: StoHyMoLAPConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def forcing(self, psi: np.ndarray, t_idx: int) -> float:
        """
        Effective forcing term psi(q, t).
        In this implementation, it is provided directly as a time series.
        """
        return float(psi[t_idx])

    def drift(self, q: float, psi_t: float) -> float:
        a = self.config.a
        b = self.config.b
        q_safe = max(q, self.config.q_min)
        return -(a / b) * (q_safe ** (2 * a - 1)) + (1.0 / b) * psi_t

    def noise_increment(self) -> float:
        """
        Draw a stochastic increment dL for one time step dt.

        Gaussian:
            dL = sqrt(dt) * N(0,1)

        Levy:
            dL ~ levy_stable(alpha, beta, scale = levy_scale * dt^(1/alpha))
        """
        dt = self.config.dt

        if self.config.noise_type.lower() == "gaussian":
            return np.sqrt(dt) * self.rng.normal(0.0, 1.0)

        if self.config.noise_type.lower() == "levy":
            alpha = self.config.alpha_levy
            beta = self.config.beta_levy
            scale = self.config.levy_scale * (dt ** (1.0 / alpha))
            return levy_stable.rvs(
                alpha=alpha,
                beta=beta,
                loc=0.0,
                scale=scale,
                random_state=self.rng,
            )

        raise ValueError("noise_type must be 'gaussian' or 'levy'")

    def step(self, q: float, psi_t: float) -> float:
        """
        One Euler-Maruyama step.
        """
        dt = self.config.dt
        theta = self.config.theta

        drift_term = self.drift(q, psi_t) * dt
        diffusion_term = theta * q * self.noise_increment()

        q_next = q + drift_term + diffusion_term

        if not np.isfinite(q_next) or q_next > 1e6:
            print(f"\n⚠️ BLOW-UP DETECTED at t={t}")
            print(f"q        = {q}")
            print(f"psi_t    = {psi_t}")
            print(f"drift    = {drift_term}")
            print(f"noise    = {noise}")
            print(f"diff     = {diffusion_term}")
            print(f"q_next   = {q_next}")
     
        # positivity safeguard
        return max(q_next, self.config.q_min)

    def simulate(self, psi: np.ndarray, q0: float) -> np.ndarray:
        """
        Simulate one discharge trajectory.
        """
        psi = np.asarray(psi, dtype=float)
        q = np.empty(len(psi), dtype=float)
        q[0] = max(float(q0), self.config.q_min)

        for t in range(len(psi) - 1):
            psi_t = self.forcing(psi, t)
            q[t + 1] = self.step(q[t], psi_t)

        return q

    def simulate_ensemble(self, psi: np.ndarray, q0: float, n_members: int = 100) -> np.ndarray:
        """
        Simulate an ensemble of trajectories.
        Returns array of shape (n_members, n_time).
        """
        out = []
        for _ in range(n_members):
            out.append(self.simulate(psi=psi, q0=q0))
        return np.vstack(out)

# --- OBJECTIVE FUNCTIONS ---

def simulate_mean_response(
    psi: np.ndarray,
    q0: float,
    params: Dict[str, float],
    base_config: StoHyMoLAPConfig,
    n_members: int = 30
) -> np.ndarray:
    """
    Simulate multiple stochastic trajectories and return their ensemble mean.
    """
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
        random_seed=base_config.random_seed,
    )
    model = StoHyMoLAP(cfg)
    ens = model.simulate_ensemble(psi=psi, q0=q0, n_members=n_members)
    return np.mean(ens, axis=0)
    
def objective_abtheta(
    x: np.ndarray,
    psi: np.ndarray,
    q_obs: np.ndarray,
    q0: float,
    base_config: StoHyMoLAPConfig,
    metric: str = "nse",
    n_members: int = 30
) -> float:
    """
    Objective function for calibrating a, b, and theta.
    """
    a, b, theta = x
    params = {"a": a, "b": b, "theta": theta}
    q_sim = simulate_mean_response(
        psi=psi,
        q0=q0,
        params=params,
        base_config=base_config,
        n_members=n_members,
    )

    if metric.lower() == "nse":
        return -nse(q_obs, q_sim)
    elif metric.lower() == "rmse":
        return rmse(q_obs, q_sim)
    elif metric.lower() == "kge":
        return -kge(q_obs, q_sim)
    else:
        raise ValueError("metric must be 'nse', 'rmse', or 'kge'")
# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # 1. Create dummy data for demonstration (Replace this with your real data)
    df = pd.read_csv("../data.csv",encoding='utf-8', sep = ';')
    df['psi'] = np.maximum(df['Precipitation'] - df['PET'], 0)
    psi_obs = df["psi"].to_numpy()
    q_obs = df["Qobs"].to_numpy()
    q0 = q_obs[0]
    # 2. Setup Configuration
    base_cfg = StoHyMoLAPConfig(
        a=1.0, b=20.0, theta=0.05, dt=1.0,
        noise_type="gaussian", random_seed=123
    )

    bounds_abtheta = [(0.6, 2.0), (1.0, 40.0), (0.001, 0.3)]

    # 3. Run Differential Evolution
    print("Starting optimization...")
    result_abtheta = differential_evolution(
        objective_abtheta,
        bounds=bounds_abtheta,
        args=(psi_obs, q_obs, q_obs[0], base_cfg, "nse", 40),
        seed=123,
        maxiter=20,
        polish=True,
        workers=-1,          # Now safe to use parallel cores
        updating='deferred'  
    )

    # 4. Results & Plotting
    best_a2, best_b2, best_theta2 = result_abtheta.x
    q_cal_abtheta = simulate_mean_response(psi_obs, q_obs[0], 
                                          {"a": best_a2, "b": best_b2, "theta": best_theta2}, 
                                          base_cfg, 100)

    print(f"\nBest Parameters: a={best_a2:.4f}, b={best_b2:.4f}, theta={best_theta2:.4f}")
    print(f"Best NSE: {-result_abtheta.fun:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(q_obs, label="Observed", color='black', alpha=0.6)
    plt.plot(q_cal_abtheta, label="Calibrated Mean", color='red', linewidth=1.5)
    plt.legend()
    plt.title("Stochastic Calibration Results")
    plt.show()
