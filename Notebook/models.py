import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from scipy.stats import levy_stable


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

            return float(inc)

        raise ValueError("noise_type must be 'gaussian' or 'levy'")

    def step(self, q: float, psi_t: float, t: Optional[int] = None) -> float:
        n_sub = self.config.n_substeps
        dt_sub = self.config.dt / n_sub
        theta = self.config.theta

        q_new = max(q, self.config.q_min)

        for _ in range(n_sub):
            drift_term = self.drift(q_new, psi_t) * dt_sub
            diffusion_term = theta * q_new * self.noise_increment(dt_sub)

            q_new = q_new + drift_term + diffusion_term

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
            member_seed = None
            if self.config.random_seed is not None:
                member_seed = self.config.random_seed + i

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
                random_seed=member_seed
            )

            member_model = StoHyMoLAP(member_cfg)
            out.append(member_model.simulate(psi=psi, q0=q0))

        return np.vstack(out)


# ============================================================
# ENSEMBLE UTILITIES
# ============================================================

def robust_ensemble_summary(
    ens: np.ndarray,
    method: str = "median",
    trim_fraction: float = 0.1
) -> np.ndarray:
    ens = np.asarray(ens, dtype=float)

    if ens.ndim != 2:
        raise ValueError("ens must be a 2D array with shape (n_members, n_times)")

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