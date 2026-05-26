import time
from datetime import datetime

import numpy as np

from levy import stable_rvs_cms
from metrics import nse, kge, rmse, pbias, nse_vectorized
from model import state_basin_vectorized


def simulate_candidates_paper(
    mu, lambda_, sigma, peff, q0,
    levy_values,
    alpha_area,           # ← NUEVO: factor de escala Peff→Q
    clamp_negative_q=True
):
    mu          = np.asarray(mu,          dtype=float)
    lambda_     = np.asarray(lambda_,     dtype=float)
    sigma       = np.asarray(sigma,       dtype=float)
    peff        = np.asarray(peff,        dtype=float)
    levy_values = np.asarray(levy_values, dtype=float)

    n_params  = len(mu)
    n         = len(peff)
    q0_safe   = max(float(q0), 1e-6)

    qsim      = np.zeros((n_params, n), dtype=float)
    qsim[:,0] = q0_safe

    x_state   = state_basin_vectorized(mu, lambda_, peff)

    ratio     = mu / lambda_
    inv_lambda= 1.0 / lambda_
    exponent  = 2.0 * mu - 1.0

    dlev = levy_values[1:] - levy_values[:-1]

    for k in range(1, n):
        q_prev = qsim[:, k - 1]
        if clamp_negative_q:
            q_prev       = np.maximum(q_prev, 0.0)
            qsim[:, k-1] = q_prev

        q_norm  = q_prev / q0_safe
        q_power = np.power(np.maximum(q_norm, 0.0), exponent) * q0_safe

        qsim[:, k] = (
            q_prev
            - ratio     * q_power
            + inv_lambda * x_state[:, k-1] * alpha_area   # ← escala aplicada
            + sigma     * q_prev * dlev[k-1]
        )

        bad = ~np.isfinite(qsim[:, k])
        if np.any(bad):
            qsim[bad, k] = np.nan

    if clamp_negative_q:
        qsim[:, -1] = np.maximum(qsim[:, -1], 0.0)

    return qsim


def calibrate_paper_heuristic(discharge, peff, config):
    discharge = np.asarray(discharge, dtype=float)
    peff      = np.asarray(peff,      dtype=float)

    # Factor de escala: convierte Peff (mm/día) a unidades de Q
    peff_mean  = np.mean(peff[peff > 0]) if np.any(peff > 0) else 1.0
    alpha_area = np.mean(discharge) / peff_mean

    n   = len(discharge)
    q0  = discharge[0]
    rng = np.random.default_rng(config.random_seed)

    sig_best    = np.zeros(config.max_traj_cal, dtype=float)
    mu_best     = np.zeros(config.max_traj_cal, dtype=float)
    lambda_best = np.zeros(config.max_traj_cal, dtype=float)
    nse_best    = np.zeros(config.max_traj_cal, dtype=float)
    qq_best     = np.zeros((n, config.max_traj_cal), dtype=float)

    start_time = time.time()

    print("=" * 72)
    print("CALIBRACIÓN HEURÍSTICA STO-HYMOLAP")
    print(f"  alpha Lévy        = {config.alpha_levy}")
    print(f"  beta Lévy         = {config.beta_levy}")
    print(f"  trayectorias      = {config.max_traj_cal}")
    print(f"  candidatos/tray.  = {config.n_param_samples}")
    print(f"  longitud serie    = {n}")
    print(f"  alpha_area        = {alpha_area:.4f}")
    print("=" * 72, flush=True)

    for traj in range(config.max_traj_cal):
        lev = stable_rvs_cms(
            alpha=config.alpha_levy,
            beta=config.beta_levy,
            loc=0.0,
            scale=1.0,
            size=n,
            rng=rng
        )

        mu_candidates = rng.uniform(
            config.mu_bounds[0],
            config.mu_bounds[1],
            size=config.n_param_samples
        )
        lambda_candidates = rng.uniform(
            config.lambda_bounds[0],
            config.lambda_bounds[1],
            size=config.n_param_samples
        )
        sigma_candidates = rng.uniform(
            config.sigma_bounds[0],
            config.sigma_bounds[1],
            size=config.n_param_samples
        )

        q_candidates = simulate_candidates_paper(
            mu=mu_candidates,
            lambda_=lambda_candidates,
            sigma=sigma_candidates,
            peff=peff,
            q0=q0,
            levy_values=lev,
            alpha_area=alpha_area,
            clamp_negative_q=config.clamp_negative_q
        )

        scores = nse_vectorized(discharge, q_candidates)

        if np.all(~np.isfinite(scores)):
            raise RuntimeError(
                f"Todas las simulaciones fallaron en la trayectoria {traj}."
            )

        idx_best = int(np.nanargmax(scores))

        mu_best[traj]     = mu_candidates[idx_best]
        lambda_best[traj] = lambda_candidates[idx_best]
        sig_best[traj]    = sigma_candidates[idx_best]
        nse_best[traj]    = scores[idx_best]
        qq_best[:, traj]  = q_candidates[idx_best, :]

        if (traj + 1) % max(1, config.max_traj_cal // 10) == 0 or traj == 0:
            elapsed = time.time() - start_time
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Trayectoria {traj + 1}/{config.max_traj_cal} | "
                f"NSE mejor={nse_best[traj]:.4f} | "
                f"mu={mu_best[traj]:.4f}, "
                f"lambda={lambda_best[traj]:.4f}, "
                f"sigma={sig_best[traj]:.6f} | "
                f"tiempo={elapsed:.1f}s",
                flush=True
            )

    mean_mu     = float(np.mean(mu_best))
    mean_lambda = float(np.mean(lambda_best))
    mean_sigma  = float(np.mean(sig_best))

    mean_trajectory = np.mean(qq_best, axis=1)
    inf_trajectory  = np.percentile(qq_best, 2.5,  axis=1)
    sup_trajectory  = np.percentile(qq_best, 97.5, axis=1)

    metrics = {
        "NSE":   nse(discharge,   mean_trajectory),
        "KGE":   kge(discharge,   mean_trajectory),
        "RMSE":  rmse(discharge,  mean_trajectory),
        "PBIAS": pbias(discharge, mean_trajectory)
    }

    results = {
        "mean_mu":        mean_mu,
        "mean_lambda":    mean_lambda,
        "mean_sigma":     mean_sigma,
        "mu_best":        mu_best,
        "lambda_best":    lambda_best,
        "sigma_best":     sig_best,
        "nse_best":       nse_best,
        "qq_best":        qq_best,
        "mean_trajectory":mean_trajectory,
        "inf_trajectory": inf_trajectory,
        "sup_trajectory": sup_trajectory,
        "metrics":        metrics
    }

    return results, alpha_area