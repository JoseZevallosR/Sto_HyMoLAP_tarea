import os
import pandas as pd

from config import PaperHyMoLAPConfig
from data_io import load_hydrological_data
from diagnostics import print_basic_statistics
from calibration import calibrate_paper_heuristic
from validation import run_validation
from outputs import save_results_csv
from plots import plot_hydrograph_with_ci


def main():

    config = PaperHyMoLAPConfig(
        data_path="../data.csv",

        max_traj_cal=1000,
        n_param_samples=5000,
        n_traj_validation=50000,
        validation_chunk_size=2000,

        alpha_levy=1.3,
        beta_levy=-0.8,

        mu_bounds=(0.75, 0.95),
        lambda_bounds=(2.0, 3.4),
        sigma_bounds=(0.0, 0.1),

        random_seed=10,
        clamp_negative_q=True,
        output_dir="outputs_stohymolap_paper"
    )

    os.makedirs(config.output_dir, exist_ok=True)

    discharge_all, precip_all, pet_all, peff_all = load_hydrological_data(config)

    print_basic_statistics(
        discharge=discharge_all,
        precip=precip_all,
        pet=pet_all,
        peff=peff_all
    )

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

    params_path = os.path.join(
        config.output_dir,
        "calibration_best_parameters.csv"
    )

    params_df.to_csv(
        params_path,
        index=False,
        encoding="utf-8"
    )

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
        output_path=os.path.join(
            config.output_dir,
            "calibration_hydrograph.png"
        )
    )

    # --------------------------------------------------------
    # 2. Validación
    # --------------------------------------------------------

    valid_results = run_validation(
        discharge_valid=discharge_valid,
        precip_valid=precip_valid,
        peff_valid=peff_valid,
        mean_mu=mean_mu,
        mean_lambda=mean_lambda,
        mean_sigma=mean_sigma,
        alpha_area=alpha_area,
        config=config
    )

    if valid_results is not None:

        valid_metrics = valid_results["metrics"]

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
            qmean=valid_results["mean_valid"],
            qinf=valid_results["inf_valid"],
            qsup=valid_results["sup_valid"]
        )

        plot_hydrograph_with_ci(
            discharge=discharge_valid,
            precip=precip_valid,
            qmean=valid_results["mean_valid"],
            qinf=valid_results["inf_valid"],
            qsup=valid_results["sup_valid"],
            title=(
                "StoHyMoLAP validación - versión fiel al paper\n"
                f"NSE={valid_metrics['NSE']:.3f}, "
                f"KGE={valid_metrics['KGE']:.3f}, "
                f"PBIAS={valid_metrics['PBIAS']:.1f}%"
            ),
            output_path=os.path.join(
                config.output_dir,
                "validation_hydrograph.png"
            )
        )

    print("\nProceso terminado.")


if __name__ == "__main__":
    main()