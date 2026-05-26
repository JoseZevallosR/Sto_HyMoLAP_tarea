import os
import pandas as pd
import numpy as np


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
