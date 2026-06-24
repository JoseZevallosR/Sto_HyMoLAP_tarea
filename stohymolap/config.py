from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PaperHyMoLAPConfig:
    # Ruta de datos
    data_path: str = "../ramis_hydro.csv"
    csv_sep: str = ","          # to_csv(index=False) escribe con coma

    # Latitud del centroide de la cuenca, en grados decimales.
    # NEGATIVA en el hemisferio sur. La cuenca del Ramis (Puno) está
    # aproximadamente entre -14.8 y -15.4. AJUSTA este valor al de tu cuenca.
    latitude_deg: float = -15.0

    # Si es True, los huecos (NaN) de flow_obs se completan con flow_sim.
    fill_obs_with_sim: bool = True

    # --------------------------------------------------------------
    # División calibración / validación
    # --------------------------------------------------------------
    # Si train_fraction no es None, se usa esa fracción del total para
    # calibrar (p.ej. 0.7 = 70%). Si es None, se usa train_size (modo paper).
    train_fraction: Optional[float] = 0.7
    train_size: int = 1461

    # --------------------------------------------------------------
    # Parámetros Lévy: AHORA SE CALIBRAN
    # --------------------------------------------------------------
    # Valores del paper (se usan como punto de partida / fallback):
    alpha_levy: float = 1.3
    beta_levy: float = -0.8

    # Rangos de búsqueda para la calibración de alpha y beta de Lévy.
    # alpha debe estar en (0, 2]; beta en [-1, 1]. Se evitan los extremos
    # 1.0 y 2.0 porque activan ramas especiales del generador estable.
    alpha_bounds: Tuple[float, float] = (1.1, 1.9)
    beta_bounds: Tuple[float, float] = (-1.0, 0.0)

    # Fracción de trayectorias (con mejor NSE) usada para estimar los
    # parámetros calibrados finales. 0.1 = mejor 10%. Pon 1.0 para volver
    # al promedio sobre TODAS las trayectorias (no calibra alpha/beta).
    cal_select_top_frac: float = 0.1

    # Rango de parámetros del paper
    mu_bounds: Tuple[float, float] = (0.75, 0.95)
    lambda_bounds: Tuple[float, float] = (2.0, 3.4)
    sigma_bounds: Tuple[float, float] = (0.0, 0.005)

    # Configuración rápida.
    # Para reproducir el paper en modo pesado:
    #max_traj_cal = 1000
    #n_param_samples = 5000
    #n_traj_validation = 300000
    max_traj_cal: int = 1000
    n_param_samples: int = 5000
    n_traj_validation: int = 300000
    validation_chunk_size: int = 1000

    # Reproducibilidad
    random_seed: int = 10

    # Control físico
    clamp_negative_q: bool = True

    # Salida
    output_dir: str = "outputs_stohymolap_paper"
