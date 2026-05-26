import numpy as np

from levy import stable_rvs_cms
from metrics import nse, kge, rmse, pbias
from model import state_basin_scalar


def simulate_fixed_params_ensemble(
    mu, lambda_, sigma, peff, q0,
    alpha_levy, beta_levy, n_traj, rng,
    alpha_area,           # ← NUEVO
    chunk_size=1000, clamp_negative_q=True
):
    peff    = np.asarray(peff, dtype=float)
    n       = len(peff)
    q0_safe = max(float(q0), 1e-6)
    qq      = np.zeros((n, n_traj), dtype=float)

    x_state = state_basin_scalar(mu, lambda_, peff)

    ratio     = mu / lambda_
    inv_lambda= 1.0 / lambda_
    exponent  = 2.0 * mu - 1.0

    written = 0
    while written < n_traj:
        current_chunk = min(chunk_size, n_traj - written)

        lev = stable_rvs_cms(
            alpha=alpha_levy, beta=beta_levy,
            loc=0.0, scale=1.0,
            size=(current_chunk, n), rng=rng
        )

        s       = np.zeros((current_chunk, n), dtype=float)
        s[:, 0] = q0_safe

        for k in range(1, n):
            s_prev = s[:, k - 1]
            if clamp_negative_q:
                s_prev    = np.maximum(s_prev, 0.0)
                s[:, k-1] = s_prev

            dlev    = lev[:, k] - lev[:, k - 1]
            q_norm  = s_prev / q0_safe
            q_power = np.power(np.maximum(q_norm, 0.0), exponent) * q0_safe

            s[:, k] = (
                s_prev
                - ratio     * q_power
                + inv_lambda * x_state[k-1] * alpha_area  # ← escala aplicada
                + sigma     * s_prev * dlev
            )

            bad = ~np.isfinite(s[:, k])
            if np.any(bad):
                s[bad, k] = np.nan

        if clamp_negative_q:
            s[:, -1] = np.maximum(s[:, -1], 0.0)

        qq[:, written:written + current_chunk] = s.T
        written += current_chunk
        print(f"  Validación ensemble: {written}/{n_traj} trayectorias", flush=True)

    return qq


def run_validation(
    discharge_valid,
    precip_valid,
    peff_valid,
    mean_mu,
    mean_lambda,
    mean_sigma,
    alpha_area,
    config
):
    if len(discharge_valid) <= 5:
        print(
            "\nNo hay suficientes datos después del periodo de calibración para validar."
        )
        return None

    print("\n" + "=" * 72)
    print("VALIDACIÓN CON PARÁMETROS MEDIOS")
    print(f"  n_traj_validation = {config.n_traj_validation}")
    print("=" * 72, flush=True)

    rng_valid = np.random.default_rng(
        config.random_seed + 999
    )

    qq_valid = simulate_fixed_params_ensemble(
        mu=mean_mu,
        lambda_=mean_lambda,
        sigma=mean_sigma,
        peff=peff_valid,
        q0=discharge_valid[0],
        alpha_levy=config.alpha_levy,
        beta_levy=config.beta_levy,
        n_traj=config.n_traj_validation,
        rng=rng_valid,
        alpha_area=alpha_area,
        chunk_size=config.validation_chunk_size,
        clamp_negative_q=config.clamp_negative_q
    )

    mean_valid = np.nanmean(
        qq_valid,
        axis=1
    )

    inf_valid = np.nanpercentile(
        qq_valid,
        2.5,
        axis=1
    )

    sup_valid = np.nanpercentile(
        qq_valid,
        97.5,
        axis=1
    )

    valid_metrics = {
        "NSE": nse(discharge_valid, mean_valid),
        "KGE": kge(discharge_valid, mean_valid),
        "RMSE": rmse(discharge_valid, mean_valid),
        "PBIAS": pbias(discharge_valid, mean_valid)
    }

    results = {
        "qq_valid": qq_valid,
        "mean_valid": mean_valid,
        "inf_valid": inf_valid,
        "sup_valid": sup_valid,
        "metrics": valid_metrics
    }

    return results
