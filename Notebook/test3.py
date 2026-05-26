import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


# ============================================================
# CONFIGURACIÓN
# ============================================================

@dataclass
class PaperHyMoLAPConfig:
    # Ruta de datos
    data_path: str = "../data.csv"
    csv_sep: str = ";"

    # División igual al paper:
    # primeras 1461 observaciones para calibración,
    # el resto para validación.
    train_size: int = 1461

    # Parámetros Lévy del paper
    alpha_levy: float = 1.3
    beta_levy: float = -0.8

    # Rango de parámetros del paper
    mu_bounds: Tuple[float, float] = (0.75, 0.95)
    lambda_bounds: Tuple[float, float] = (2.0, 3.4)
    sigma_bounds: Tuple[float, float] = (0.0, 0.005)

    # Configuración rápida.
    # Para reproducir el paper en modo pesado:
    # max_traj_cal = 1000
    # n_param_samples = 5000
    # n_traj_validation = 300000
    max_traj_cal: int = 50
    n_param_samples: int = 500
    n_traj_validation: int = 5000
    validation_chunk_size: int = 1000

    # Reproducibilidad
    random_seed: int = 10

    # Control físico
    clamp_negative_q: bool = True

    # Salida
    output_dir: str = "outputs_stohymolap_paper"


# ============================================================
# GENERADOR ALPHA-ESTABLE SIN scipy.stats.levy_stable
# Método Chambers-Mallows-Stuck
# ============================================================

def stable_rvs_cms(alpha, beta, loc=0.0, scale=1.0, size=None, rng=None):
    """
    Genera variables aleatorias alpha-estables usando el método
    Chambers-Mallows-Stuck, con NumPy.

    Esto reemplaza a scipy.stats.levy_stable.rvs.
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha = float(alpha)
    beta = float(beta)
    scale = float(scale)

    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha debe estar en (0, 2].")

    if not (-1.0 <= beta <= 1.0):
        raise ValueError("beta debe estar en [-1, 1].")

    if scale <= 0:
        raise ValueError("scale debe ser positivo.")

    if np.isclose(alpha, 2.0):
        x = np.sqrt(2.0) * rng.normal(0.0, 1.0, size=size)
        return loc + scale * x

    u = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=size)
    w = rng.exponential(1.0, size=size)

    if np.isclose(alpha, 1.0):
        part1 = (np.pi / 2.0 + beta * u) * np.tan(u)
        part2 = beta * np.log(
            (np.pi / 2.0 * w * np.cos(u)) /
            (np.pi / 2.0 + beta * u)
        )
        x = (2.0 / np.pi) * (part1 - part2)
        return loc + scale * x

    zeta = beta * np.tan(np.pi * alpha / 2.0)

    b_alpha = np.arctan(zeta) / alpha
    s_alpha = (1.0 + zeta ** 2) ** (1.0 / (2.0 * alpha))

    numerator = np.sin(alpha * (u + b_alpha))
    denominator = np.cos(u) ** (1.0 / alpha)

    factor = (
        np.cos(u - alpha * (u + b_alpha)) / w
    ) ** ((1.0 - alpha) / alpha)

    x = s_alpha * numerator / denominator * factor

    return loc + scale * x


# ============================================================
# MÉTRICAS
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


def nse_vectorized(obs: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula NSE para muchas simulaciones simultáneamente.

    sim_matrix debe tener forma:
    (n_simulaciones, n_tiempos)
    """
    obs = np.asarray(obs, dtype=float)
    sim_matrix = np.asarray(sim_matrix, dtype=float)

    denom = np.sum((obs - np.mean(obs)) ** 2)

    if denom == 0:
        return np.full(sim_matrix.shape[0], -np.inf)

    finite_rows = np.all(np.isfinite(sim_matrix), axis=1)

    sse = np.sum((sim_matrix - obs[None, :]) ** 2, axis=1)
    out = 1.0 - sse / denom

    out[~finite_rows] = -np.inf

    return out


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

    std_obs = np.std(obs, ddof=1)
    std_sim = np.std(sim, ddof=1)

    if std_obs == 0 or std_sim == 0:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]

    if not np.isfinite(r):
        return np.nan

    alpha = std_sim / std_obs
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan

    if not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan

    return 1.0 - np.sqrt(
        (r - 1.0) ** 2 +
        (alpha - 1.0) ** 2 +
        (beta - 1.0) ** 2
    )


# ============================================================
# LECTURA DE DATOS
# ============================================================

def load_hydrological_data(config: PaperHyMoLAPConfig):
    """
    Lee datos desde CSV o Excel.

    Soporta dos formatos:

    1. Formato con nombres:
       Qobs, Precipitation, PET

    2. Formato tipo paper:
       primera columna = discharge
       segunda columna = precipitation
       tercera columna = PET
    """
    path = config.data_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path, sep=config.csv_sep, encoding="utf-8")

        if df.shape[1] < 3:
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    else:
        raise ValueError("Formato no soportado. Usa .csv, .txt, .xlsx o .xls")

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    if (
        "qobs" in cols_lower and
        "precipitation" in cols_lower and
        "pet" in cols_lower
    ):
        q_col = cols_lower["qobs"]
        p_col = cols_lower["precipitation"]
        pet_col = cols_lower["pet"]

        discharge = pd.to_numeric(df[q_col], errors="coerce").to_numpy(dtype=float)
        precip = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
        pet = pd.to_numeric(df[pet_col], errors="coerce").to_numpy(dtype=float)

    else:
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = numeric_df.columns[numeric_df.notna().sum() > 0].tolist()

        if len(numeric_cols) < 3:
            raise ValueError(
                "No se encontraron al menos tres columnas numéricas. "
                "Se requieren Q, precipitación y PET."
            )

        discharge = numeric_df[numeric_cols[0]].to_numpy(dtype=float)
        precip = numeric_df[numeric_cols[1]].to_numpy(dtype=float)
        pet = numeric_df[numeric_cols[2]].to_numpy(dtype=float)

    mask = np.isfinite(discharge) & np.isfinite(precip) & np.isfinite(pet)

    discharge = discharge[mask]
    precip = precip[mask]
    pet = pet[mask]

    if len(discharge) < 20:
        raise ValueError("Muy pocos datos válidos para ejecutar el modelo.")

    peff = precip - pet
    peff[peff < 0.0] = 0.0

    return discharge, precip, pet, peff


# ============================================================
# ESTADO DE CUENCA X(t)
# Fiel al código del paper
# ============================================================

def state_basin_scalar(mu: float, lambda_: float, peff: np.ndarray) -> np.ndarray:
    peff = np.asarray(peff, dtype=float)
    n = len(peff)
    x = np.zeros(n, dtype=float)
    x[0] = peff[0]

    decay = mu / lambda_          # tasa de vaciado por paso

    for i in range(1, n):
        x[i] = x[i - 1] * (1.0 - decay) + peff[i]   # ← CORREGIDO

    return x

def state_basin_vectorized(
    mu: np.ndarray,
    lambda_: np.ndarray,
    peff: np.ndarray
) -> np.ndarray:
    mu      = np.asarray(mu,      dtype=float)
    lambda_ = np.asarray(lambda_, dtype=float)
    peff    = np.asarray(peff,    dtype=float)

    n_params = len(mu)
    n        = len(peff)
    decay    = mu / lambda_          # shape (n_params,)

    x       = np.zeros((n_params, n), dtype=float)
    x[:, 0] = peff[0]

    for i in range(1, n):
        x[:, i] = x[:, i - 1] * (1.0 - decay) + peff[i]   # ← CORREGIDO

    return x

# ============================================================
# SIMULACIÓN FIEL AL PAPER
# ============================================================

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


# ============================================================
# CALIBRACIÓN HEURÍSTICA FIEL AL PAPER, PERO VECTORIZADA
# ============================================================

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
# ============================================================
# GRÁFICOS
# ============================================================

def plot_hydrograph_with_ci(
    discharge: np.ndarray,
    precip: np.ndarray,
    qmean: np.ndarray,
    qinf: np.ndarray,
    qsup: np.ndarray,
    title: str,
    output_path: str
):
    discharge = np.asarray(discharge, dtype=float)
    precip = np.asarray(precip, dtype=float)
    qmean = np.asarray(qmean, dtype=float)
    qinf = np.asarray(qinf, dtype=float)
    qsup = np.asarray(qsup, dtype=float)

    t = np.arange(len(discharge))

    fig, ax1 = plt.subplots(figsize=(13, 6))

    ax1.bar(t, precip, color="black", alpha=0.50, label="Precipitación")
    ax1.set_ylabel("Precipitación")
    ax1.invert_yaxis()
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(t, discharge, color="blue", linewidth=1.2, label="Qobs")
    ax2.fill_between(
        t,
        qinf,
        qsup,
        color="gray",
        alpha=0.30,
        label="IC 95%"
    )
    ax2.plot(t, qmean, color="red", linewidth=1.4, label="Qmean/Qsim")

    ax2.set_ylabel("Caudal")
    ax1.set_xlabel("Paso de tiempo")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def save_results_csv(
    output_dir: str,
    prefix: str,
    discharge: np.ndarray,
    precip: np.ndarray,
    peff: np.ndarray,
    qmean: np.ndarray,
    qinf: np.ndarray,
    qsup: np.ndarray
):
    out = pd.DataFrame({
        "Qobs": discharge,
        "Precipitation": precip,
        "Peff": peff,
        "Qmean": qmean,
        "Qinf_2_5": qinf,
        "Qsup_97_5": qsup
    })

    out_path = os.path.join(output_dir, f"{prefix}_series.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Serie guardada en: {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    config = PaperHyMoLAPConfig(
        data_path="../data.csv",

        # Estos valores son rápidos para probar.
        # Luego puedes subirlos.
        max_traj_cal=500,
        n_param_samples=5000,
        n_traj_validation=50000,
        validation_chunk_size=1000,

        # Parámetros Lévy del código original
        alpha_levy=1.3,
        beta_levy=-0.8,

        # Rangos del código original
        mu_bounds=(0.75, 0.95),
        lambda_bounds=(2.0, 3.4),
        sigma_bounds=(0.0, 0.1),

        random_seed=10,
        clamp_negative_q=True,
        output_dir="outputs_stohymolap_paper"
    )

    os.makedirs(config.output_dir, exist_ok=True)

    discharge_all, precip_all, pet_all, peff_all = load_hydrological_data(config)

    # Agrega esto justo después de load_hydrological_data en el main:
    import pandas as pd
    print(pd.DataFrame({
        "Q": discharge_all[:10],
        "P": precip_all[:10],
        "PET": pet_all[:10],
        "Peff": peff_all[:10]
    }))
    print(f"\nQ: min={discharge_all.min():.4f}, max={discharge_all.max():.4f}, mean={discharge_all.mean():.4f}")
    print(f"P: min={precip_all.min():.4f}, max={precip_all.max():.4f}, mean={precip_all.mean():.4f}")
    print(f"PET: min={pet_all.min():.4f}, max={pet_all.max():.4f}, mean={pet_all.mean():.4f}")
    print(f"Peff: min={peff_all.min():.4f}, max={peff_all.max():.4f}, mean={peff_all.mean():.4f}")

    n_total = len(discharge_all)
    n_train = min(config.train_size, n_total)

    discharge_train = discharge_all[:n_train]
    precip_train = precip_all[:n_train]
    peff_train = peff_all[:n_train]

    discharge_valid = discharge_all[n_train:]
    precip_valid = precip_all[n_train:]
    peff_valid = peff_all[n_train:]

    print("=" * 72)
    print("DATOS CARGADOS")
    print(f"  Total de datos       : {n_total}")
    print(f"  Datos calibración    : {len(discharge_train)}")
    print(f"  Datos validación     : {len(discharge_valid)}")
    print("=" * 72, flush=True)

    # --------------------------------------------------------
    # 1. Calibración
    # --------------------------------------------------------

    cal_results, alpha_area = calibrate_paper_heuristic(
        discharge=discharge_train,
        peff=peff_train,
        config=config
    )

    mean_mu = cal_results["mean_mu"]
    mean_lambda = cal_results["mean_lambda"]
    mean_sigma = cal_results["mean_sigma"]

    print("\n" + "=" * 72)
    print("RESULTADOS DE CALIBRACIÓN")
    print(f"  mean_MU     = {mean_mu:.6f}")
    print(f"  mean_LAMBDA = {mean_lambda:.6f}")
    print(f"  mean_SIGMA  = {mean_sigma:.8f}")
    print("\nMétricas con trayectoria media")
    print(f"  NSE   = {cal_results['metrics']['NSE']:.4f}")
    print(f"  KGE   = {cal_results['metrics']['KGE']:.4f}")
    print(f"  RMSE  = {cal_results['metrics']['RMSE']:.4f}")
    print(f"  PBIAS = {cal_results['metrics']['PBIAS']:.2f}%")
    print("=" * 72)

    params_df = pd.DataFrame({
        "mu": cal_results["mu_best"],
        "lambda": cal_results["lambda_best"],
        "sigma": cal_results["sigma_best"],
        "nse_best": cal_results["nse_best"]
    })

    params_path = os.path.join(config.output_dir, "calibration_best_parameters.csv")
    params_df.to_csv(params_path, index=False, encoding="utf-8")
    print(f"Parámetros de calibración guardados en: {params_path}")

    save_results_csv(
        output_dir=config.output_dir,
        prefix="calibration",
        discharge=discharge_train,
        precip=precip_train,
        peff=peff_train,
        qmean=cal_results["mean_trajectory"],
        qinf=cal_results["inf_trajectory"],
        qsup=cal_results["sup_trajectory"]
    )

    plot_hydrograph_with_ci(
        discharge=discharge_train,
        precip=precip_train,
        qmean=cal_results["mean_trajectory"],
        qinf=cal_results["inf_trajectory"],
        qsup=cal_results["sup_trajectory"],
        title=(
            "StoHyMoLAP calibración - versión fiel al paper\n"
            f"NSE={cal_results['metrics']['NSE']:.3f}, "
            f"KGE={cal_results['metrics']['KGE']:.3f}, "
            f"PBIAS={cal_results['metrics']['PBIAS']:.1f}%"
        ),
        output_path=os.path.join(config.output_dir, "calibration_hydrograph.png")
    )

    # --------------------------------------------------------
    # 2. Validación
    # --------------------------------------------------------

    if len(discharge_valid) > 5:
        print("\n" + "=" * 72)
        print("VALIDACIÓN CON PARÁMETROS MEDIOS")
        print(f"  n_traj_validation = {config.n_traj_validation}")
        print("=" * 72, flush=True)

        rng_valid = np.random.default_rng(config.random_seed + 999)

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
            alpha_area=alpha_area,              # ← pasar aquí
            chunk_size=config.validation_chunk_size,
            clamp_negative_q=config.clamp_negative_q
        )

        mean_valid = np.nanmean(qq_valid, axis=1)
        inf_valid = np.nanpercentile(qq_valid, 2.5, axis=1)
        sup_valid = np.nanpercentile(qq_valid, 97.5, axis=1)

        valid_metrics = {
            "NSE": nse(discharge_valid, mean_valid),
            "KGE": kge(discharge_valid, mean_valid),
            "RMSE": rmse(discharge_valid, mean_valid),
            "PBIAS": pbias(discharge_valid, mean_valid)
        }

        print("\n" + "=" * 72)
        print("RESULTADOS DE VALIDACIÓN")
        print(f"  NSE   = {valid_metrics['NSE']:.4f}")
        print(f"  KGE   = {valid_metrics['KGE']:.4f}")
        print(f"  RMSE  = {valid_metrics['RMSE']:.4f}")
        print(f"  PBIAS = {valid_metrics['PBIAS']:.2f}%")
        print("=" * 72)

        save_results_csv(
            output_dir=config.output_dir,
            prefix="validation",
            discharge=discharge_valid,
            precip=precip_valid,
            peff=peff_valid,
            qmean=mean_valid,
            qinf=inf_valid,
            qsup=sup_valid
        )

        plot_hydrograph_with_ci(
            discharge=discharge_valid,
            precip=precip_valid,
            qmean=mean_valid,
            qinf=inf_valid,
            qsup=sup_valid,
            title=(
                "StoHyMoLAP validación - versión fiel al paper\n"
                f"NSE={valid_metrics['NSE']:.3f}, "
                f"KGE={valid_metrics['KGE']:.3f}, "
                f"PBIAS={valid_metrics['PBIAS']:.1f}%"
            ),
            output_path=os.path.join(config.output_dir, "validation_hydrograph.png")
        )

    else:
        print("\nNo hay suficientes datos después del periodo de calibración para validar.")

    print("\nProceso terminado.")