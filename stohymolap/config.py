from dataclasses import dataclass
from typing import Tuple


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