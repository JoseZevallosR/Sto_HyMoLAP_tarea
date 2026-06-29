# Contexto de publicación — StoHyMoLAP

Generado: 20260628_125846
Proyecto: /home/jose/Documents/githubs/Sto_HyMoLAP_tarea/stohymolap_project

## Objetivo del paquete

Preparar un contexto autosuficiente para redactar las secciones de Resultados y Discusión de un manuscrito basado en StoHyMoLAP/RAMIS, usando los artefactos reproducibles de Fase 3.4B y Fase 3.4C/3.5.

## Regla crítica de redacción

No afirmar que E7/E8 son GRU reales si el backend indica `fallback_mlp_on_flattened_sequences`. Deben describirse como configuraciones GRU-like/declaradas como GRU, ejecutadas con backend fallback por ausencia de TensorFlow/PyTorch, salvo que se reinstale un backend recurrente real y se regenere la evidencia.

## Relación con el paper base

El artículo de Houénafa et al. (2025) propone la hibridación entre un modelo hidrológico estocástico y ML mediante propiedades estadísticas simuladas de la distribución diaria de caudal, incluyendo media y cuantiles. Su estructura útil para esta redacción combina: comparación de modelos físicos/ML/híbridos, tabla de métricas, hidrogramas y scatter, análisis por clases de caudal, sensibilidad a rangos de incertidumbre y discusión de la dependencia de desempeño respecto de las propiedades de distribución usadas como entrada.

## Pregunta científica sugerida

¿En qué medida la incorporación de memoria hidrológica/baseflow, perturbaciones estocásticas tipo Lévy y predictores híbridos basados en salidas físicas mejora la simulación diaria de caudales frente a modelos físicos y ML puros bajo una ventana de validación común sin evidencia de leakage temporal?

## Mensaje central preliminar

El desempeño mejora de forma acumulativa cuando el modelo conserva estructura física e incorpora información adicional útil para ML. El mejor resultado común corresponde a E8_HYB_GRU_QUANTILES; sin embargo, la discusión debe matizarse porque el backend ejecutado es fallback MLP sobre secuencias aplanadas y no una GRU recurrente real.

## Artefactos clave esperados

- `outputs/reports/phase34/phase34_results_report.md`: reporte narrativo y tablas base.
- `outputs/reports/phase34/table_final_ranking_common_window.csv`: ranking común E0-E8.
- `outputs/reports/phase34/table_ablation_effects_common_window.csv`: efectos de ablación.
- `outputs/reports/phase34/table_regime_metrics_selected.csv`: métricas por régimen.
- `outputs/reports/phase34/table_backend_manuscript_notes.csv`: notas de backend para redacción.
- `outputs/figures/phase34/phase34_figure_manifest.csv`: disponibilidad de figuras.
- `outputs/figures/phase34/*.png`: figuras paper-ready.


### Git — rama actual

```text
main
```

### Git — últimos commits

```text
35706d6 Add Phase 3.4C paper-ready result report
c56c7de Add Phase 3.4B paper-ready figures
e75439c fix: make leakage audit compatible with legacy outputs
d369114 feat: add phase 3.4A leakage audit and backend QA
69b73c9 chore: remove python cache artifacts from repository
13f482e chore: remove python cache artifacts from repository
1fdb8c6 fix: audit nonnegative ML predictions and add common intersection leaderboard
97af7c7 feat: enforce common evaluation window for phase 3 comparisons
```

### Git — estado corto

```text
 M generar_contexto.sh
?? contexto_stohymolap_fase3_4B.md
?? contextos_publicacion/
?? generate_context_publicacion.sh
?? prompt_siguiente_chat_fase3_4B.md
?? ../texto/
```

### Árbol relevante del proyecto

```text
configs/base.yaml
configs/experiments.yaml
configs/model_bounds.yaml
contextos_publicacion/publicacion_20260628_125846/contexto_publicacion_stohymolap.md
contextos_publicacion/publicacion_20260628_125846/manifest_publicacion.txt
contextos/stohymolap_contexto_fase3_4B_20260628_123956_filelist.txt
contextos/stohymolap_contexto_fase3_4B_20260628_123956.tar.gz
contexto_stohymolap_fase3_4B.md
data/ramis_hydro.csv
docs/diagrams/00_arquitectura_general.mmd
docs/diagrams/E0_RAMIS_DET_NOBF.mmd
docs/diagrams/E1_RAMIS_DET_BF.mmd
docs/diagrams/E2_RAMIS_LEVY_NOBF.mmd
docs/diagrams/E3_RAMIS_LEVY_BF.mmd
docs/diagrams/E4_ML_PURE_XGB.mmd
docs/diagrams/E5_HYB_XGB_MEAN.mmd
docs/diagrams/E6_HYB_XGB_QUANTILES.mmd
docs/diagrams/E7_HYB_GRU_MEAN.mmd
docs/diagrams/E8_HYB_GRU_QUANTILES.mmd
generar_contexto.sh
generate_context_publicacion.sh
.gitignore
outputs/comparison/ablation_effects_common_intersection.csv
outputs/comparison/ablation_effects.csv
outputs/comparison/best_model_summary.md
outputs/comparison/evaluation_window_comparison.csv
outputs/comparison/experiment_matrix.csv
outputs/comparison/leaderboard_common_intersection.csv
outputs/comparison/leaderboard.csv
outputs/comparison/leakage_audit_issues.csv
outputs/comparison/leakage_audit.json
outputs/comparison/leakage_audit_report.md
outputs/comparison/leakage_audit_status.txt
outputs/comparison/leakage_audit_summary.csv
outputs/comparison/ml_backend_summary.csv
outputs/comparison/phase26_audit.log
outputs/comparison/physical_audit_issues.csv
outputs/comparison/physical_audit_report.md
outputs/comparison/physical_audit_status.txt
outputs/comparison/physical_audit_summary.csv
outputs/comparison/regime_metrics_comparison.csv
outputs/comparison/run_all.log
outputs/comparison/uncertainty_comparison.csv
outputs/comparison/validation_metrics_common_intersection.csv
outputs/comparison/validation_metrics_comparison.csv
prompt_siguiente_chat_fase3_4B.md
pyproject.toml
.pytest_cache/CACHEDIR.TAG
.pytest_cache/.gitignore
.pytest_cache/README.md
README.md
requirements.txt
scripts/audit_leakage_phase34.py
scripts/audit_physical_phase26.py
scripts/generate_phase34_figures.py
scripts/generate_phase34_report.py
scripts/run_all_experiments.py
scripts/run_experiment.py
scripts/summarize_results.py
src/stohymolap.egg-info/dependency_links.txt
src/stohymolap.egg-info/PKG-INFO
src/stohymolap.egg-info/requires.txt
src/stohymolap.egg-info/SOURCES.txt
src/stohymolap.egg-info/top_level.txt
src/stohymolap/__init__.py
src/stohymolap/validation_checks.py
stohymolap_phase3_4B_figures.patch
stohymolap_phase3_4C_report.patch
tests/test_phase34_figures.py
tests/test_phase34_report.py
tests/test_smoke.py
```

### README

Archivo: `README.md`

```text
# StoHyMoLAP / RAMIS — Plataforma experimental de modelado lluvia-escorrentía

Plataforma modular para comparar, sobre una **única base de código** controlada
por archivos YAML, distintos enfoques de modelado del caudal diario de la cuenca
**Ramis (Puno, Perú)**:

- **RAMIS determinístico** (físico) con reservorio de **flujo base**.
- **RAMIS estocástico** con ruido **α-estable (Lévy)** y cuantificación de incertidumbre.
- **Machine Learning puro** (XGBoost).
- **Híbridos RAMIS-ML** (XGBoost y GRU) que post-procesan el ensemble físico.

La plataforma añade **memoria hidrológica** mediante un reservorio lineal de flujo
base, una **calibración multiobjetivo** (NSE, KGE, PBIAS y cobertura), métricas
**por régimen** (caudales bajos/medios/altos) y de **incertidumbre**, además de
salvaguardas **anti-fuga temporal (anti-leakage)**.

> Nota sobre backends ML: el código usa `xgboost`, TensorFlow o PyTorch **si están
> instalados**. Si no lo están, recurre automáticamente a equivalentes de
> scikit-learn (`HistGradientBoostingRegressor` para XGBoost; `MLPRegressor` sobre
> secuencias aplanadas para GRU). El backend efectivamente usado se registra en
> cada corrida (`backend=...`) y en el log.

---

## 1. Estructura del proyecto

```
stohymolap_project/
├── configs/
│   ├── base.yaml                 # bloques globales compartidos
│   ├── experiments.yaml          # matriz de 9 experimentos (autocontenido)
│   └── model_bounds.yaml         # ficha de límites de parámetros
├── data/
│   └── ramis_hydro.csv           # serie diaria (date, P, Tmin, Tmax, flow_obs)
├── docs/diagrams/                # 9 diagramas Mermaid (.mmd)
├── scripts/
│   ├── run_experiment.py         # ejecuta UN experimento
│   ├── run_all_experiments.py    # ejecuta la matriz + comparación
│   └── summarize_results.py      # regenera la comparación desde salidas
├── src/stohymolap/
│   ├── data/        (io, preprocessing)
│   ├── hydro/       (pet, ramis, baseflow, water_balance)
│   ├── stochastic/  (levy, monte_carlo)
│   ├── calibration/ (objective, monte_carlo_search, parameter_store)
│   ├── features/    (lagged, uncertainty_features, feature_builder)
│   ├── ml/          (xgboost_model, gru_model, pure_ml, train_predict)
│   ├── metrics/     (deterministic, uncertainty, regime_metrics)
│   ├── plotting/    (hydrographs, scatter, flow_duration, uncertainty, diagrams)
│   ├── experiments/ (runner, registry, comparison)
│   ├── utils/       (config, logging, reproducibility)
│   └── validation_checks.py      # comprobaciones anti-fuga e integridad
├── outputs/
│   ├── experiments/E*/           # salidas por experimento
│   └── comparison/               # leaderboard + tablas + figuras
├── requirements.txt
└── README.md
```

## 2. Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Opcional (mejores backends): pip install xgboost tensorflow
```

## 3. Datos de entrada

`data/ramis_hydro.csv` con columnas (los nombres son configurables en `global:`):

| columna              | descripción                         |
|----------------------|-------------------------------------|
| `date`               | fecha diaria (YYYY-MM-DD)           |
| `precipitation_mean` | precipitación areal media (mm)      |
| `tmin`, `tmax`       | temperaturas mín/máx (°C)           |
| `flow_obs`           | caudal observado (m³/s)             |
| `flow_sim` (opcional)| caudal de referencia                |

> El CSV incluido es un **dataset sintético** plausible para verificar el pipeline
> de extremo a extremo. Reemplázalo por la serie real de Ramis manteniendo las
> columnas (o ajusta los nombres en `configs/*.yaml`).

## 4. Ejecución

Un experimento:

```bash
python scripts/run_experiment.py --config configs/experiments.yaml \
       --experiment E1_RAMIS_DET_BF
```

Toda la matriz + comparación final:

```bash
python scripts/run_all_experiments.py --config configs/experiments.yaml
```

Subconjunto o tolerante a fallos:

```bash
python scripts/run_all_experiments.py --config configs/experiments.yaml \
       --only E2_RAMIS_LEVY_NOBF E3_RAMIS_LEVY_BF --continue-on-error
```

Regenerar solo la comparación (sin recalcular):

```bash
python scripts/summarize_results.py --config configs/experiments.yaml
```

## 5. Matriz de experimentos

Fase 3.2 usa una matriz canónica sin duplicados y una ventana común de evaluación. Los siete primeros son la
matriz mínima publicable; E7/E8 son extensiones secuenciales opcionales.

| ID | Modelo | Estocástico | Flujo base | ML | Rol |
|----|--------|:----------:|:----------:|----|----|
| **E0_RAMIS_DET_NOBF** | físico determinístico sin memoria | – | ✖ | – | mínimo |
| **E1_RAMIS_DET_BF** | físico determinístico con memoria | – | ✔ | – | mínimo |
| **E2_RAMIS_LEVY_NOBF** | físico estocástico Lévy sin memoria | ✔ | ✖ | – | mínimo |
| **E3_RAMIS_LEVY_BF** | físico estocástico Lévy con memoria | ✔ | ✔ | – | mínimo |
| **E4_ML_PURE_XGB** | ML puro sobre forzantes | – | ✖ | XGBoost | mínimo |
| **E5_HYB_XGB_MEAN** | híbrido con media física | ✔ | ✔ | XGBoost | mínimo |
| **E6_HYB_XGB_QUANTILES** | híbrido con cuantiles físicos | ✔ | ✔ | XGBoost | mínimo |
| **E7_HYB_GRU_MEAN** | híbrido secuencial con media | ✔ | ✔ | GRU | extendido |
| **E8_HYB_GRU_QUANTILES** | híbrido secuencial con cuantiles | ✔ | ✔ | GRU | extendido |

Lecturas directas de ablación:

- **Memoria hidrológica determinística:** E1 - E0.
- **Memoria hidrológica estocástica:** E3 - E2.
- **Ruido Lévy con memoria:** E3 - E1.
- **ML puro vs física:** E4 - E3.
- **Hibridación:** E5/E6/E7/E8 frente a E3 y E4.

## 5.1. Ventana común de evaluación

La comparación Fase 3.2 recorta todos los experimentos a una misma ventana de
fechas objetivo. Esto evita que los modelos físicos compitan con más días de
validación que los modelos ML/híbridos, que pierden días iniciales por lags o
longitud de secuencia. La política se controla en:

```yaml
evaluation:
  common_window:
    enabled: true
    mode: declared_matrix_max_warmup
    start_offset: auto
    end_trim: 0
```

Con la matriz E0-E8 actual, `start_offset: auto` infiere 7 días porque la GRU usa
`sequence_length=7` y `forecast_horizon=1`. Cada experimento guarda
`evaluation_window.json` y la comparación consolida
`evaluation_window_comparison.csv`.

## 6. Nuevos componentes científicos

**Reservorio de flujo base** (memoria hidrológica), activable por config:

```
Ep_t   = max(P_t - PET_t, 0)
R_t    = c_r * Ep_t
S_b[t+1] = max(0, S_b[t] + R_t - k_b * S_b[t])
Q_b[t] = k_b * S_b[t]
Q_total = max(Q_fast + Q_b, 0)
```
con `c_r∈[0,1]`, `k_b∈[0.001,0.5]`, `S0_b∈[0,10]`.

**Calibración multiobjetivo** (reemplaza el criterio solo-NSE):

```
J = w_nse·(1-NSE) + w_kge·(1-KGE) + w_pbias·|PBIAS/100| + w_cov·|0.95-PICP|
```
pesos por defecto `0.40 / 0.30 / 0.15 / 0.15`. Si un experimento no produce banda,
el término de cobertura se omite y los pesos se renormalizan.

**Anti-fuga (anti-leakage):** split temporal (no aleatorio); escaladores ML
ajustados **solo en train**; predictores en `t` usan `t, t-1, t-2` (nunca futuro);
umbrales de régimen fijados en calibración; secuencias GRU que no mezclan validación.
Todas estas invariantes se verifican en `validation_checks.py`.

## 7. Salidas por experimento (`outputs/experiments/<ID>/`)

`config_used.yaml`, `best_parameters.csv`, `top_k_parameters.csv`,
`metrics_train.csv`, `metrics_validation.csv`, `metrics_by_regime.csv`,
`uncertainty_metrics.csv` (solo físico-estocásticos donde la banda corresponde a Qsim), `physical_uncertainty_reference.csv` (híbridos), `evaluation_window.json`, `predictions_train.csv`,
`predictions_validation.csv`, `ensemble_summary.csv` (estocásticos),
`model_artifact/` (modelos ML), `run.log`, y `figures/` con hidrogramas,
dispersión, curva de duración, banda de incertidumbre, residuos por régimen y
diagrama de Taylor.

## 8. Salidas de comparación (`outputs/comparison/`)

`leaderboard.csv`, `experiment_matrix.csv`, `validation_metrics_comparison.csv`,
`regime_metrics_comparison.csv`, `uncertainty_comparison.csv`,
`ablation_effects.csv`, `evaluation_window_comparison.csv`, `best_model_summary.md` y `figures/` (barras de métricas, hidrogramas de los
mejores modelos, comparación de curvas de duración y RMSE por régimen).

## 9. Cómo interpretar los resultados

- **¿El flujo base ayuda?** Compara **E1-E0** en RAMIS determinístico y
  **E3-E2** en RAMIS-Lévy. Un Δ(NSE/KGE) > 0 y menor RMSE en caudales bajos
  (`metrics_by_regime.csv`, régimen `low`) indica que la memoria hidrológica
  mejora la recesión. `ablation_effects.csv` calcula estos Δ.
- **¿La hibridación ayuda?** Compara **E3** (físico-estocástico completo) y
  **E4** (ML puro) contra **E5/E6/E7/E8**. Si los híbridos superan a ambos,
  el post-procesamiento ML aporta sobre la física; si E4 ya gana, el ML domina
  la señal.
- **Incertidumbre:** revisa `PICP` (idealmente ≈ 0.95) junto con `PINAW/MPIW`
  y `Winkler` solo en modelos físico-estocásticos. En híbridos, la banda física
  se guarda como referencia y no se reporta como incertidumbre final del Qsim ML.
- **Cuidado con la calibración:** los valores de ejemplo usan números modestos
  (`n_param_samples`, `n_iter`, `n_trajectories`) para correr rápido. Para
  conclusiones científicas, súbelos (p. ej. 3000 / 1000 / 2000) en `configs`.
- Mira siempre la métrica de **validación** (no la de calibración) y contrasta con
  la inspección visual de hidrogramas y curvas de duración.

> Las salidas incluidas se generaron con la configuración de ejemplo y sirven como
> demostración; vuelve a ejecutar con tus datos y tu presupuesto de cómputo.

## Fase 3.4A — Auditoría anti-leakage y QA

La Fase 3.4A agrega una auditoría reproducible antes de preparar figuras y tablas publicables. La auditoría revisa:

- separación temporal estricta entre calibración y validación;
- lags no negativos y secuencias formadas solo con información hasta `t` para predecir `t+horizon`;
- ausencia de `Qobs`/target observado dentro de las variables predictoras;
- consistencia entre `evaluation_window.json` y los CSV de predicciones;
- origen de umbrales de régimen como `train_Qobs_only`;
- backend ML usado realmente, distinguiendo GRU real (`tensorflow`/`torch`) de fallback `sklearn_mlp`;
- aplicación del postproceso `Qsim = max(Qsim_raw, 0)` solo a ML/híbridos.

Uso recomendado después de ejecutar o resumir la matriz:

```bash
python scripts/audit_leakage_phase34.py --config configs/experiments.yaml
```

Artefactos generados en `outputs/comparison/`:

- `leakage_audit_report.md`
- `leakage_audit_summary.csv`
- `leakage_audit_issues.csv`
- `leakage_audit.json`

Además, cada corrida nueva guarda en el directorio de cada experimento:

- `ml_backend.json`
- `regime_thresholds.json`

[TRUNCADO: se muestran 250 de 327 líneas]
```

### Reporte paper-ready Fase 3.4C/3.5

Archivo: `outputs/reports/phase34/phase34_results_report.md`

```text
# StoHyMoLAP Phase 3.4C/3.5 — Paper-ready result package

Generated at: `2026-06-28T17:58:50+00:00`
Output root: `outputs`

## 1. Status

- Anti-leakage audit status: `WARN`
- Physical audit status: `WARN`
- Package warnings: `1`

## 2. Best common-window model

The best model on the common validation intersection is `E8_HYB_GRU_QUANTILES` with NSE=0.8491, KGE=0.8986, RMSE=0.2097, MAE=0.1240 and PBIAS=1.5791 over n=4367 samples (`2009-01-08 to 2020-12-31`).

## 3. Final ranking table

|   rank | experiment           |   n_eval | common_start_date   | common_end_date   |    NSE |    KGE |   RMSE |    MAE |   PBIAS |     R2 | backend_label                       |
|-------:|:---------------------|---------:|:--------------------|:------------------|-------:|-------:|-------:|-------:|--------:|-------:|:------------------------------------|
|      1 | E8_HYB_GRU_QUANTILES |     4367 | 2009-01-08          | 2020-12-31        | 0.8491 | 0.8986 | 0.2097 | 0.124  |  1.5791 | 0.8495 | fallback_mlp_on_flattened_sequences |
|      2 | E6_HYB_XGB_QUANTILES |     4367 | 2009-01-08          | 2020-12-31        | 0.7399 | 0.8098 | 0.2753 | 0.1617 |  0.9266 | 0.7401 | sklearn_hgb                         |
|      3 | E7_HYB_GRU_MEAN      |     4367 | 2009-01-08          | 2020-12-31        | 0.6865 | 0.7891 | 0.3023 | 0.192  |  1.6436 | 0.6888 | fallback_mlp_on_flattened_sequences |
|      4 | E5_HYB_XGB_MEAN      |     4367 | 2009-01-08          | 2020-12-31        | 0.6262 | 0.7146 | 0.3301 | 0.2099 |  2.0068 | 0.6267 | sklearn_hgb                         |
|      5 | E4_ML_PURE_XGB       |     4367 | 2009-01-08          | 2020-12-31        | 0.5752 | 0.6609 | 0.3519 | 0.2228 |  0.7423 | 0.5753 | sklearn_hgb                         |
|      6 | E1_RAMIS_DET_BF      |     4367 | 2009-01-08          | 2020-12-31        | 0.4385 | 0.7137 | 0.4046 | 0.2478 |  0.8602 | 0.5104 | physical                            |
|      7 | E3_RAMIS_LEVY_BF     |     4367 | 2009-01-08          | 2020-12-31        | 0.4286 | 0.725  | 0.4081 | 0.2554 | -1.2176 | 0.5593 | physical                            |
|      8 | E2_RAMIS_LEVY_NOBF   |     4367 | 2009-01-08          | 2020-12-31        | 0.4269 | 0.7167 | 0.4087 | 0.2678 | -7.5511 | 0.5529 | physical                            |
|      9 | E0_RAMIS_DET_NOBF    |     4367 | 2009-01-08          | 2020-12-31        | 0.4142 | 0.7134 | 0.4132 | 0.2696 | -5.8648 | 0.554  | physical                            |

## 4. Selected model metrics for figures

| experiment           |    NSE |    KGE |   RMSE |    MAE |   PBIAS |     R2 | backend_label                       |
|:---------------------|-------:|-------:|-------:|-------:|--------:|-------:|:------------------------------------|
| E0_RAMIS_DET_NOBF    | 0.4142 | 0.7134 | 0.4132 | 0.2696 | -5.8648 | 0.554  | physical                            |
| E1_RAMIS_DET_BF      | 0.4385 | 0.7137 | 0.4046 | 0.2478 |  0.8602 | 0.5104 | physical                            |
| E3_RAMIS_LEVY_BF     | 0.4286 | 0.725  | 0.4081 | 0.2554 | -1.2176 | 0.5593 | physical                            |
| E4_ML_PURE_XGB       | 0.5752 | 0.6609 | 0.3519 | 0.2228 |  0.7423 | 0.5753 | sklearn_hgb                         |
| E6_HYB_XGB_QUANTILES | 0.7399 | 0.8098 | 0.2753 | 0.1617 |  0.9266 | 0.7401 | sklearn_hgb                         |
| E8_HYB_GRU_QUANTILES | 0.8491 | 0.8986 | 0.2097 | 0.124  |  1.5791 | 0.8495 | fallback_mlp_on_flattened_sequences |

## 5. Ablation effects

| ablation                       | baseline             | candidate            |   delta_NSE |   delta_KGE |   delta_RMSE |   delta_MAE |   delta_abs_PBIAS_improvement |
|:-------------------------------|:---------------------|:---------------------|------------:|------------:|-------------:|------------:|------------------------------:|
| deterministic_baseflow         | E0_RAMIS_DET_NOBF    | E1_RAMIS_DET_BF      |      0.0242 |      0.0003 |      -0.0086 |     -0.0219 |                        5.0045 |
| stochastic_baseflow            | E2_RAMIS_LEVY_NOBF   | E3_RAMIS_LEVY_BF     |      0.0017 |      0.0082 |      -0.0006 |     -0.0124 |                        6.3335 |
| levy_given_baseflow            | E1_RAMIS_DET_BF      | E3_RAMIS_LEVY_BF     |     -0.0098 |      0.0112 |       0.0035 |      0.0076 |                       -0.3573 |
| pure_ml_vs_physical            | E3_RAMIS_LEVY_BF     | E4_ML_PURE_XGB       |      0.1466 |     -0.064  |      -0.0562 |     -0.0326 |                        0.4753 |
| hybrid_mean_vs_physical        | E3_RAMIS_LEVY_BF     | E5_HYB_XGB_MEAN      |      0.1976 |     -0.0104 |      -0.078  |     -0.0455 |                       -0.7893 |
| hybrid_quantiles_vs_physical   | E3_RAMIS_LEVY_BF     | E6_HYB_XGB_QUANTILES |      0.3113 |      0.0848 |      -0.1328 |     -0.0937 |                        0.291  |
| quantiles_vs_mean_xgb          | E5_HYB_XGB_MEAN      | E6_HYB_XGB_QUANTILES |      0.1137 |      0.0952 |      -0.0548 |     -0.0482 |                        1.0803 |
| gru_mean_vs_xgb_mean           | E5_HYB_XGB_MEAN      | E7_HYB_GRU_MEAN      |      0.0603 |      0.0745 |      -0.0278 |     -0.0179 |                        0.3632 |
| gru_quantiles_vs_xgb_quantiles | E6_HYB_XGB_QUANTILES | E8_HYB_GRU_QUANTILES |      0.1092 |      0.0888 |      -0.0656 |     -0.0377 |                       -0.6525 |

## 6. Regime metrics

| experiment           | regime   |    n |       NSE |      KGE |   RMSE |    MAE |    PBIAS |
|:---------------------|:---------|-----:|----------:|---------:|-------:|-------:|---------:|
| E0_RAMIS_DET_NOBF    | low      | 1368 | -167.437  | -10.8169 | 0.1767 | 0.0971 |  49.9406 |
| E0_RAMIS_DET_NOBF    | mid      | 1922 |   -6.402  |  -0.9594 | 0.3919 | 0.2635 |   6.5969 |
| E0_RAMIS_DET_NOBF    | high     | 1077 |   -0.4647 |   0.394  | 0.6153 | 0.4998 | -12.9142 |
| E1_RAMIS_DET_BF      | low      | 1368 | -208.954  | -10.6068 | 0.1973 | 0.1009 | 195.043  |
| E1_RAMIS_DET_BF      | mid      | 1922 |   -5.4252 |  -0.7472 | 0.3651 | 0.2064 |  32.1287 |
| E1_RAMIS_DET_BF      | high     | 1077 |   -0.456  |   0.4339 | 0.6135 | 0.5082 | -19.7381 |
| E3_RAMIS_LEVY_BF     | low      | 1368 | -184.113  | -11.3523 | 0.1853 | 0.092  |  68.0296 |
| E3_RAMIS_LEVY_BF     | mid      | 1922 |   -6.2186 |  -0.9384 | 0.387  | 0.2396 |  19.0256 |
| E3_RAMIS_LEVY_BF     | high     | 1077 |   -0.4099 |   0.4115 | 0.6037 | 0.4911 | -11.5188 |
| E4_ML_PURE_XGB       | low      | 1368 | -191.175  |  -8.7538 | 0.1888 | 0.124  | 239.646  |
| E4_ML_PURE_XGB       | mid      | 1922 |   -2.6842 |  -0.2562 | 0.2765 | 0.1654 |  40.4909 |
| E4_ML_PURE_XGB       | high     | 1077 |   -0.2396 |   0.3761 | 0.5661 | 0.4508 | -25.0157 |
| E6_HYB_XGB_QUANTILES | low      | 1368 |  -29.2244 |  -2.4206 | 0.0749 | 0.0471 |  90.2341 |
| E6_HYB_XGB_QUANTILES | mid      | 1922 |   -1.3689 |   0.0497 | 0.2217 | 0.1336 |  30.9098 |
| E6_HYB_XGB_QUANTILES | high     | 1077 |    0.1777 |   0.5722 | 0.4611 | 0.3574 | -13.6187 |
| E8_HYB_GRU_QUANTILES | low      | 1368 |  -10.6944 |  -0.8546 | 0.0466 | 0.0324 |  56.9967 |
| E8_HYB_GRU_QUANTILES | mid      | 1922 |   -0.4053 |   0.3175 | 0.1708 | 0.1082 |  16.1341 |
| E8_HYB_GRU_QUANTILES | high     | 1077 |    0.522  |   0.7392 | 0.3515 | 0.2685 |  -6.13   |

## 7. Uncertainty and stochastic references

| experiment         |   PICP |   PINAW |   MPIW |   Winkler |
|:-------------------|-------:|--------:|-------:|----------:|
| E2_RAMIS_LEVY_NOBF | 0.1601 |  0.0841 | 0.2187 |    7.2971 |
| E3_RAMIS_LEVY_BF   | 0.0037 |  0.0034 | 0.0088 |   10.05   |

Note: `E2_RAMIS_LEVY_NOBF, E3_RAMIS_LEVY_BF` should be described as underdispersive stochastic references, not calibrated uncertainty bands.

## 8. Figure availability

| status    |   count |
|:----------|--------:|
| fallback  |       1 |
| generated |      14 |
| skipped   |       6 |

| figure_id                                | status    | path                                                                 | warnings                                                                                                                                                                                                                                                        |
|:-----------------------------------------|:----------|:---------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| hydrograph_validation_common             | generated | outputs/figures/phase34/hydrograph_validation_common.png             | backend=sklearn_hgb | backend=sklearn_hgb | declared GRU uses backend=fallback_mlp_on_flattened_sequences; not a real GRU in this environment                                                                                                                   |
| fdc_common                               | generated | outputs/figures/phase34/fdc_common.png                               |                                                                                                                                                                                                                                                                 |
| scatter_E0_RAMIS_DET_NOBF                | generated | outputs/figures/phase34/scatter_E0_RAMIS_DET_NOBF.png                |                                                                                                                                                                                                                                                                 |
| residuals_by_regime_E0_RAMIS_DET_NOBF    | generated | outputs/figures/phase34/residuals_by_regime_E0_RAMIS_DET_NOBF.png    |                                                                                                                                                                                                                                                                 |
| scatter_E1_RAMIS_DET_BF                  | generated | outputs/figures/phase34/scatter_E1_RAMIS_DET_BF.png                  |                                                                                                                                                                                                                                                                 |
| residuals_by_regime_E1_RAMIS_DET_BF      | generated | outputs/figures/phase34/residuals_by_regime_E1_RAMIS_DET_BF.png      |                                                                                                                                                                                                                                                                 |
| scatter_E3_RAMIS_LEVY_BF                 | generated | outputs/figures/phase34/scatter_E3_RAMIS_LEVY_BF.png                 |                                                                                                                                                                                                                                                                 |
| residuals_by_regime_E3_RAMIS_LEVY_BF     | generated | outputs/figures/phase34/residuals_by_regime_E3_RAMIS_LEVY_BF.png     |                                                                                                                                                                                                                                                                 |
| scatter_E4_ML_PURE_XGB                   | generated | outputs/figures/phase34/scatter_E4_ML_PURE_XGB.png                   |                                                                                                                                                                                                                                                                 |
| residuals_by_regime_E4_ML_PURE_XGB       | generated | outputs/figures/phase34/residuals_by_regime_E4_ML_PURE_XGB.png       |                                                                                                                                                                                                                                                                 |
| scatter_E6_HYB_XGB_QUANTILES             | generated | outputs/figures/phase34/scatter_E6_HYB_XGB_QUANTILES.png             |                                                                                                                                                                                                                                                                 |
| residuals_by_regime_E6_HYB_XGB_QUANTILES | generated | outputs/figures/phase34/residuals_by_regime_E6_HYB_XGB_QUANTILES.png |                                                                                                                                                                                                                                                                 |
| scatter_E8_HYB_GRU_QUANTILES             | generated | outputs/figures/phase34/scatter_E8_HYB_GRU_QUANTILES.png             |                                                                                                                                                                                                                                                                 |
| residuals_by_regime_E8_HYB_GRU_QUANTILES | generated | outputs/figures/phase34/residuals_by_regime_E8_HYB_GRU_QUANTILES.png |                                                                                                                                                                                                                                                                 |
| physical_components_E0_RAMIS_DET_NOBF    | skipped   |                                                                      | physical component columns are absent in predictions_validation.csv                                                                                                                                                                                             |
| physical_components_E1_RAMIS_DET_BF      | skipped   |                                                                      | physical component columns are absent in predictions_validation.csv                                                                                                                                                                                             |
| physical_components_E3_RAMIS_LEVY_BF     | skipped   |                                                                      | physical component columns are absent in predictions_validation.csv                                                                                                                                                                                             |
| physical_components_E4_ML_PURE_XGB       | skipped   |                                                                      | physical component columns are absent in predictions_validation.csv                                                                                                                                                                                             |
| physical_components_E6_HYB_XGB_QUANTILES | skipped   |                                                                      | physical component columns are absent in predictions_validation.csv                                                                                                                                                                                             |
| physical_components_E8_HYB_GRU_QUANTILES | skipped   |                                                                      | physical component columns are absent in predictions_validation.csv                                                                                                                                                                                             |
| stochastic_reference_E2_E3               | fallback  | outputs/figures/phase34/stochastic_reference_E2_E3.png               | E2_RAMIS_LEVY_NOBF: PICP=0.160; warning: stochastic band is strongly underdispersive | E3_RAMIS_LEVY_BF: PICP=0.004; warning: stochastic band is strongly underdispersive | lower/upper stochastic band columns are absent; generated point-reference plot only |

## 9. Backend and manuscript wording

| experiment           | ml_model   | backend     | backend_label                       | feature_source      | manuscript_claim                                                                      |
|:---------------------|:-----------|:------------|:------------------------------------|:--------------------|:--------------------------------------------------------------------------------------|
| E0_RAMIS_DET_NOBF    |            | physical    | physical                            | physical_only       | Report according to backend metadata                                                  |
| E1_RAMIS_DET_BF      |            | physical    | physical                            | physical_only       | Report according to backend metadata                                                  |
| E2_RAMIS_LEVY_NOBF   |            | physical    | physical                            | physical_only       | Report according to backend metadata                                                  |
| E3_RAMIS_LEVY_BF     |            | physical    | physical                            | physical_only       | Report according to backend metadata                                                  |
| E4_ML_PURE_XGB       | xgboost    | sklearn_hgb | sklearn_hgb                         | observed_forcing    | Report according to backend metadata                                                  |
| E5_HYB_XGB_MEAN      | xgboost    | sklearn_hgb | sklearn_hgb                         | stochastic_ensemble | Report according to backend metadata                                                  |
| E6_HYB_XGB_QUANTILES | xgboost    | sklearn_hgb | sklearn_hgb                         | stochastic_ensemble | Report according to backend metadata                                                  |
| E7_HYB_GRU_MEAN      | gru        | sklearn_mlp | fallback_mlp_on_flattened_sequences | stochastic_ensemble | GRU-labelled experiment; report actual backend as fallback MLP on flattened sequences |
| E8_HYB_GRU_QUANTILES | gru        | sklearn_mlp | fallback_mlp_on_flattened_sequences | stochastic_ensemble | GRU-labelled experiment; report actual backend as fallback MLP on flattened sequences |

Use backend-aware wording: when E7/E8 are declared as GRU but run with `fallback_mlp_on_flattened_sequences`, describe them as GRU-labelled hybrid experiments executed with a flattened-sequence MLP fallback, not as real recurrent GRU models.

## 10. Reproducibility checklist

| item                                    | status   | evidence                                                                                               | action                                                         |
|:----------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
| Common validation intersection          | PASS     | leaderboard_common_intersection.csv                                                                    | Use only common-window metrics in the manuscript.              |
| Anti-leakage audit                      | WARN     | leakage_audit_status.txt=WARN                                                                          | Do not submit if status becomes FAIL; document WARN items.     |
| Physical audit                          | WARN     | physical_audit_status.txt=WARN                                                                         | Discuss physical limitations when WARN is present.             |
| Figure manifest                         | PASS     | outputs/figures/phase34/phase34_figure_manifest.csv                                                    | Regenerate figures before final manuscript package if missing. |
| GRU wording                             | WARN     | GRU over-claim risk: E7_HYB_GRU_MEAN;E8_HYB_GRU_QUANTILES use fallback backend, not real recurrent GRU | Use backend-aware wording for E7/E8.                           |
| Optional physical components in figures | WARN     | skipped entries in phase34_figure_manifest.csv                                                         | Only claim Qfast/Qbase/Qtotal figures if columns are exported. |

## 11. Generated package artefacts

| artifact_id                             | status    |   rows | path                                                                | notes                                       |
|:----------------------------------------|:----------|-------:|:--------------------------------------------------------------------|:--------------------------------------------|
| table_final_ranking_common_window       | generated |      9 | outputs/reports/phase34/table_final_ranking_common_window.csv       | Main quantitative result table              |
| table_selected_models_common_window     | generated |      6 | outputs/reports/phase34/table_selected_models_common_window.csv     | Models used in Phase 3.4B figures           |
| table_ablation_effects_common_window    | generated |      9 | outputs/reports/phase34/table_ablation_effects_common_window.csv    | Ablation deltas on common validation window |
| table_regime_metrics_selected           | generated |     18 | outputs/reports/phase34/table_regime_metrics_selected.csv           | Regime metrics for selected models          |
| table_uncertainty_stochastic_references | generated |      2 | outputs/reports/phase34/table_uncertainty_stochastic_references.csv | Stochastic reference diagnostics            |
| table_figure_status_summary             | generated |      3 | outputs/reports/phase34/table_figure_status_summary.csv             | Generated/fallback/skipped figure counts    |
| table_backend_manuscript_notes          | generated |      9 | outputs/reports/phase34/table_backend_manuscript_notes.csv          | Backend-aware wording notes                 |
| table_reproducibility_checklist         | generated |      6 | outputs/reports/phase34/table_reproducibility_checklist.csv         | Submission-readiness checks                 |
| phase34_report_manifest                 | generated |      0 | outputs/reports/phase34/phase34_report_manifest.json                | Machine-readable report package manifest    |

## 12. Warnings

- GRU over-claim risk: E7_HYB_GRU_MEAN;E8_HYB_GRU_QUANTILES use fallback backend, not real recurrent GRU

## 13. Suggested manuscript framing

- Use the common-window leaderboard as the main quantitative result table.
- Present E8 as the best current model only under the exact backend used in this environment.
- Discuss E2/E3 stochastic bands as references if PICP is low, not as reliable predictive uncertainty.
- Do not claim physical component plots until Qfast/Qbase/Qtotal are exported by the experiment runners.
- Keep audit WARN items visible in supplementary reproducibility material.
```

### Ranking final en ventana común

Archivo: `outputs/reports/phase34/table_final_ranking_common_window.csv`

```csv
rank,experiment,n_eval,common_start_date,common_end_date,NSE,KGE,RMSE,MAE,PBIAS,R2,model_type,ml_model,backend_label,feature_source
1,E8_HYB_GRU_QUANTILES,4367,2009-01-08,2020-12-31,0.8491160018721119,0.8985560770853956,0.2097227434174252,0.1239717132746818,1.5791389617175808,0.84951314368649,hybrid_sequence,gru,fallback_mlp_on_flattened_sequences,stochastic_ensemble
2,E6_HYB_XGB_QUANTILES,4367,2009-01-08,2020-12-31,0.7399273544058875,0.80977759904721,0.2753412477742982,0.1616946172969598,0.9265927312935628,0.7400981468460716,hybrid,xgboost,sklearn_hgb,stochastic_ensemble
3,E7_HYB_GRU_MEAN,4367,2009-01-08,2020-12-31,0.6864827857787669,0.7890942412234975,0.3023114763475741,0.1919919830612535,1.6435975392447295,0.6887968694151526,hybrid_sequence,gru,fallback_mlp_on_flattened_sequences,stochastic_ensemble
4,E5_HYB_XGB_MEAN,4367,2009-01-08,2020-12-31,0.6262062864359661,0.7145963848249611,0.3300957516669702,0.20987887748592,2.0068432918117,0.6266527505471495,hybrid,xgboost,sklearn_hgb,stochastic_ensemble
5,E4_ML_PURE_XGB,4367,2009-01-08,2020-12-31,0.5752172486548741,0.6609452130021479,0.3518903645671916,0.2227895504887427,0.7422587700568847,0.5752636980045411,machine_learning,xgboost,sklearn_hgb,observed_forcing
6,E1_RAMIS_DET_BF,4367,2009-01-08,2020-12-31,0.4384715015927956,0.7137185090693596,0.4045850580202305,0.2477720418508602,0.860236359127669,0.5103514291817111,physical,,physical,physical_only
7,E3_RAMIS_LEVY_BF,4367,2009-01-08,2020-12-31,0.4286477539033839,0.7249663584026212,0.408108751516014,0.2554212403680974,-1.2175665522260202,0.5592798595377673,stochastic_physical,,physical,physical_only
8,E2_RAMIS_LEVY_NOBF,4367,2009-01-08,2020-12-31,0.4269286762130487,0.7167407376700682,0.4087222467178347,0.2677771377611023,-7.551079226355426,0.5528836595701413,stochastic_physical,,physical,physical_only
9,E0_RAMIS_DET_NOBF,4367,2009-01-08,2020-12-31,0.4142486498117365,0.7134014251980563,0.4132192904027306,0.2696486039164401,-5.864756204694348,0.5540337015777467,physical,,physical,physical_only
```

### Modelos seleccionados para figuras

Archivo: `outputs/reports/phase34/table_selected_models_common_window.csv`

```csv
rank,experiment,n_eval,common_start_date,common_end_date,NSE,KGE,RMSE,MAE,PBIAS,R2,model_type,ml_model,backend_label,feature_source
9,E0_RAMIS_DET_NOBF,4367,2009-01-08,2020-12-31,0.4142486498117365,0.7134014251980563,0.4132192904027306,0.2696486039164401,-5.864756204694348,0.5540337015777467,physical,,physical,physical_only
6,E1_RAMIS_DET_BF,4367,2009-01-08,2020-12-31,0.4384715015927956,0.7137185090693596,0.4045850580202305,0.2477720418508602,0.860236359127669,0.5103514291817111,physical,,physical,physical_only
7,E3_RAMIS_LEVY_BF,4367,2009-01-08,2020-12-31,0.4286477539033839,0.7249663584026212,0.408108751516014,0.2554212403680974,-1.2175665522260202,0.5592798595377673,stochastic_physical,,physical,physical_only
5,E4_ML_PURE_XGB,4367,2009-01-08,2020-12-31,0.5752172486548741,0.6609452130021479,0.3518903645671916,0.2227895504887427,0.7422587700568847,0.5752636980045411,machine_learning,xgboost,sklearn_hgb,observed_forcing
2,E6_HYB_XGB_QUANTILES,4367,2009-01-08,2020-12-31,0.7399273544058875,0.80977759904721,0.2753412477742982,0.1616946172969598,0.9265927312935628,0.7400981468460716,hybrid,xgboost,sklearn_hgb,stochastic_ensemble
1,E8_HYB_GRU_QUANTILES,4367,2009-01-08,2020-12-31,0.8491160018721119,0.8985560770853956,0.2097227434174252,0.1239717132746818,1.5791389617175808,0.84951314368649,hybrid_sequence,gru,fallback_mlp_on_flattened_sequences,stochastic_ensemble
```

### Efectos de ablación

Archivo: `outputs/reports/phase34/table_ablation_effects_common_window.csv`

```csv
ablation,baseline,candidate,question,delta_NSE,delta_KGE,delta_RMSE,delta_MAE,delta_abs_PBIAS_improvement,n_eval_baseline,n_eval_candidate
deterministic_baseflow,E0_RAMIS_DET_NOBF,E1_RAMIS_DET_BF,Efecto del reservorio baseflow en RAMIS deterministico,0.024222851781059,0.0003170838713033,-0.0086342323825001,-0.0218765620655798,5.004519845566678,4367,4367
stochastic_baseflow,E2_RAMIS_LEVY_NOBF,E3_RAMIS_LEVY_BF,Efecto del reservorio baseflow en RAMIS-Levy,0.0017190776903351,0.008225620732553,-0.0006134952018206,-0.0123558973930049,6.333512674129405,4367,4367
levy_given_baseflow,E1_RAMIS_DET_BF,E3_RAMIS_LEVY_BF,Efecto del ruido Levy manteniendo baseflow,-0.0098237476894116,0.0112478493332616,0.0035236934957835,0.0076491985172371,-0.3573301930983511,4367,4367
pure_ml_vs_physical,E3_RAMIS_LEVY_BF,E4_ML_PURE_XGB,ML puro frente al fisico-estocastico completo,0.1465694947514901,-0.0640211454004733,-0.0562183869488223,-0.0326316898793546,0.4753077821691355,4367,4367
hybrid_mean_vs_physical,E3_RAMIS_LEVY_BF,E5_HYB_XGB_MEAN,Hibrido XGB-media frente al fisico-estocastico completo,0.1975585325325821,-0.0103699735776601,-0.0780129998490438,-0.0455423628821773,-0.7892767395856799,4367,4367
hybrid_quantiles_vs_physical,E3_RAMIS_LEVY_BF,E6_HYB_XGB_QUANTILES,Hibrido XGB-cuantiles frente al fisico-estocastico completo,0.3112796005025036,0.0848112406445887,-0.1327675037417157,-0.0937266230711375,0.2909738209324572,4367,4367
quantiles_vs_mean_xgb,E5_HYB_XGB_MEAN,E6_HYB_XGB_QUANTILES,Aporte de cuantiles frente a solo Qmean en XGB,0.1137210679699214,0.0951812142222489,-0.0547545038926719,-0.0481842601889602,1.0802505605181372,4367,4367
gru_mean_vs_xgb_mean,E5_HYB_XGB_MEAN,E7_HYB_GRU_MEAN,Extension GRU-media frente a XGB-media,0.0602764993428007,0.0744978563985363,-0.027784275319396,-0.0178868944246665,0.3632457525669703,4367,4367
gru_quantiles_vs_xgb_quantiles,E6_HYB_XGB_QUANTILES,E8_HYB_GRU_QUANTILES,Extension GRU-cuantiles frente a XGB-cuantiles,0.1091886474662243,0.0887784780381856,-0.065618504356873,-0.037722904022278,-0.6525462304240179,4367,4367
```

### Métricas por régimen

Archivo: `outputs/reports/phase34/table_regime_metrics_selected.csv`

```csv
experiment,regime,n,NSE,KGE,RMSE,MAE,PBIAS
E0_RAMIS_DET_NOBF,low,1368,-167.43677557797142,-10.816876301970098,0.1767114116970202,0.0970683570186978,49.9406057294248
E0_RAMIS_DET_NOBF,mid,1922,-6.401957311027732,-0.9594228392035358,0.3918888660349847,0.2635050281721482,6.596922959679679
E0_RAMIS_DET_NOBF,high,1077,-0.464693839589563,0.3940230089245748,0.6153203785247593,0.4998229124927079,-12.914225121670954
E1_RAMIS_DET_BF,low,1368,-208.9540601084904,-10.606830488841574,0.1972914549459516,0.1009025424539017,195.04307267382964
E1_RAMIS_DET_BF,mid,1922,-5.425164455474709,-0.7471705888982636,0.3651167533177685,0.2063543296095859,32.128698971824264
E1_RAMIS_DET_BF,high,1077,-0.4560269035323228,0.4338767166132065,0.6134971801452394,0.5082384467745079,-19.738124439742784
E3_RAMIS_LEVY_BF,low,1368,-184.1125418169712,-11.35225354325656,0.1852524920725675,0.0920493468602603,68.02961278628021
E3_RAMIS_LEVY_BF,mid,1922,-6.218593844310093,-0.9383909370942378,0.3870044334800824,0.2396332108971476,19.025645454609744
E3_RAMIS_LEVY_BF,high,1077,-0.4099266132518224,0.4114519834305418,0.6037068799213294,0.4911105095991897,-11.51882644973357
E4_ML_PURE_XGB,low,1368,-191.1747258052382,-8.753772780498208,0.1887531777323919,0.1239713715799234,239.64621908395776
E4_ML_PURE_XGB,mid,1922,-2.6842205773807,-0.2561784304171952,0.2764791902234826,0.1653755896396804,40.49094883614825
E4_ML_PURE_XGB,high,1077,-0.2395550118508707,0.3760995391085419,0.5660577931605073,0.4507681034127561,-25.01574246456537
E6_HYB_XGB_QUANTILES,low,1368,-29.22442519355636,-2.420551671441954,0.0748557492951145,0.0471194931440789,90.2340822605086
E6_HYB_XGB_QUANTILES,mid,1922,-1.368890761226266,0.0497136089426166,0.2216981933232863,0.1335961414045237,30.909796146980952
E6_HYB_XGB_QUANTILES,high,1077,0.177652862817409,0.5721983470691147,0.4610578953783853,0.3573715351302034,-13.618719979953555
E8_HYB_GRU_QUANTILES,low,1368,-10.694359702084371,-0.854566060475247,0.0465622931562704,0.0323675138068657,56.99673890908074
E8_HYB_GRU_QUANTILES,mid,1922,-0.4053481111011086,0.3174873364166684,0.1707582101598029,0.1081690059186078,16.134121411776878
E8_HYB_GRU_QUANTILES,high,1077,0.522027438223039,0.7392319382157331,0.3515032576319902,0.2685282113344281,-6.129959341368462
```

### Referencias estocásticas/incertidumbre

Archivo: `outputs/reports/phase34/table_uncertainty_stochastic_references.csv`

```csv
experiment,PICP,PINAW,MPIW,Winkler
E2_RAMIS_LEVY_NOBF,0.1600641172429585,0.0840696217014992,0.2186650860455995,7.297140056301506
E3_RAMIS_LEVY_BF,0.0036638424547744,0.0033684917097607,0.0087614469370875,10.050000041406184
```

### Resumen de estado de figuras

Archivo: `outputs/reports/phase34/table_figure_status_summary.csv`

```csv
status,count
fallback,1
generated,14
skipped,6
```

### Notas de backend para manuscrito

Archivo: `outputs/reports/phase34/table_backend_manuscript_notes.csv`

```csv
experiment,model_type,ml_model,backend,backend_label,feature_source,postprocess_method,manuscript_claim
E0_RAMIS_DET_NOBF,physical,,physical,physical,physical_only,none,Report according to backend metadata
E1_RAMIS_DET_BF,physical,,physical,physical,physical_only,none,Report according to backend metadata
E2_RAMIS_LEVY_NOBF,stochastic_physical,,physical,physical,physical_only,none,Report according to backend metadata
E3_RAMIS_LEVY_BF,stochastic_physical,,physical,physical,physical_only,none,Report according to backend metadata
E4_ML_PURE_XGB,machine_learning,xgboost,sklearn_hgb,sklearn_hgb,observed_forcing,clip_negative_to_zero,Report according to backend metadata
E5_HYB_XGB_MEAN,hybrid,xgboost,sklearn_hgb,sklearn_hgb,stochastic_ensemble,clip_negative_to_zero,Report according to backend metadata
E6_HYB_XGB_QUANTILES,hybrid,xgboost,sklearn_hgb,sklearn_hgb,stochastic_ensemble,clip_negative_to_zero,Report according to backend metadata
E7_HYB_GRU_MEAN,hybrid_sequence,gru,sklearn_mlp,fallback_mlp_on_flattened_sequences,stochastic_ensemble,clip_negative_to_zero,GRU-labelled experiment; report actual backend as fallback MLP on flattened sequences
E8_HYB_GRU_QUANTILES,hybrid_sequence,gru,sklearn_mlp,fallback_mlp_on_flattened_sequences,stochastic_ensemble,clip_negative_to_zero,GRU-labelled experiment; report actual backend as fallback MLP on flattened sequences
```

### Checklist de reproducibilidad

Archivo: `outputs/reports/phase34/table_reproducibility_checklist.csv`

```csv
item,status,evidence,action
Common validation intersection,PASS,leaderboard_common_intersection.csv,Use only common-window metrics in the manuscript.
Anti-leakage audit,WARN,leakage_audit_status.txt=WARN,Do not submit if status becomes FAIL; document WARN items.
Physical audit,WARN,physical_audit_status.txt=WARN,Discuss physical limitations when WARN is present.
Figure manifest,PASS,outputs/figures/phase34/phase34_figure_manifest.csv,Regenerate figures before final manuscript package if missing.
GRU wording,WARN,"GRU over-claim risk: E7_HYB_GRU_MEAN;E8_HYB_GRU_QUANTILES use fallback backend, not real recurrent GRU",Use backend-aware wording for E7/E8.
Optional physical components in figures,WARN,skipped entries in phase34_figure_manifest.csv,Only claim Qfast/Qbase/Qtotal figures if columns are exported.
```

### Manifest de figuras

Archivo: `outputs/figures/phase34/phase34_figure_manifest.csv`

```csv
figure_id,status,path,experiments,required_columns,missing_columns,warnings,notes
hydrograph_validation_common,generated,outputs/figures/phase34/hydrograph_validation_common.png,E0_RAMIS_DET_NOBF;E1_RAMIS_DET_BF;E3_RAMIS_LEVY_BF;E4_ML_PURE_XGB;E6_HYB_XGB_QUANTILES;E8_HYB_GRU_QUANTILES,date;Qobs;Qsim,,backend=sklearn_hgb | backend=sklearn_hgb | declared GRU uses backend=fallback_mlp_on_flattened_sequences; not a real GRU in this environment,n_common=4367
fdc_common,generated,outputs/figures/phase34/fdc_common.png,E0_RAMIS_DET_NOBF;E1_RAMIS_DET_BF;E3_RAMIS_LEVY_BF;E4_ML_PURE_XGB;E6_HYB_XGB_QUANTILES;E8_HYB_GRU_QUANTILES,date;Qobs;Qsim,,,n_common=4367
scatter_E0_RAMIS_DET_NOBF,generated,outputs/figures/phase34/scatter_E0_RAMIS_DET_NOBF.png,E0_RAMIS_DET_NOBF,date;Qobs;Qsim,,,n=4367
residuals_by_regime_E0_RAMIS_DET_NOBF,generated,outputs/figures/phase34/residuals_by_regime_E0_RAMIS_DET_NOBF.png,E0_RAMIS_DET_NOBF,date;Qobs;Qsim;regime_thresholds.json:p25;regime_thresholds.json:p75,,,"p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
scatter_E1_RAMIS_DET_BF,generated,outputs/figures/phase34/scatter_E1_RAMIS_DET_BF.png,E1_RAMIS_DET_BF,date;Qobs;Qsim,,,n=4367
residuals_by_regime_E1_RAMIS_DET_BF,generated,outputs/figures/phase34/residuals_by_regime_E1_RAMIS_DET_BF.png,E1_RAMIS_DET_BF,date;Qobs;Qsim;regime_thresholds.json:p25;regime_thresholds.json:p75,,,"p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
scatter_E3_RAMIS_LEVY_BF,generated,outputs/figures/phase34/scatter_E3_RAMIS_LEVY_BF.png,E3_RAMIS_LEVY_BF,date;Qobs;Qsim,,,n=4367
residuals_by_regime_E3_RAMIS_LEVY_BF,generated,outputs/figures/phase34/residuals_by_regime_E3_RAMIS_LEVY_BF.png,E3_RAMIS_LEVY_BF,date;Qobs;Qsim;regime_thresholds.json:p25;regime_thresholds.json:p75,,,"p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
scatter_E4_ML_PURE_XGB,generated,outputs/figures/phase34/scatter_E4_ML_PURE_XGB.png,E4_ML_PURE_XGB,date;Qobs;Qsim,,,n=4367
residuals_by_regime_E4_ML_PURE_XGB,generated,outputs/figures/phase34/residuals_by_regime_E4_ML_PURE_XGB.png,E4_ML_PURE_XGB,date;Qobs;Qsim;regime_thresholds.json:p25;regime_thresholds.json:p75,,,"p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
scatter_E6_HYB_XGB_QUANTILES,generated,outputs/figures/phase34/scatter_E6_HYB_XGB_QUANTILES.png,E6_HYB_XGB_QUANTILES,date;Qobs;Qsim,,,n=4367
residuals_by_regime_E6_HYB_XGB_QUANTILES,generated,outputs/figures/phase34/residuals_by_regime_E6_HYB_XGB_QUANTILES.png,E6_HYB_XGB_QUANTILES,date;Qobs;Qsim;regime_thresholds.json:p25;regime_thresholds.json:p75,,,"p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
scatter_E8_HYB_GRU_QUANTILES,generated,outputs/figures/phase34/scatter_E8_HYB_GRU_QUANTILES.png,E8_HYB_GRU_QUANTILES,date;Qobs;Qsim,,,n=4367
residuals_by_regime_E8_HYB_GRU_QUANTILES,generated,outputs/figures/phase34/residuals_by_regime_E8_HYB_GRU_QUANTILES.png,E8_HYB_GRU_QUANTILES,date;Qobs;Qsim;regime_thresholds.json:p25;regime_thresholds.json:p75,,,"p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
physical_components_E0_RAMIS_DET_NOBF,skipped,,E0_RAMIS_DET_NOBF,Qfast;Qbase;Qtotal,Qfast;Qbase;Qtotal,physical component columns are absent in predictions_validation.csv,
physical_components_E1_RAMIS_DET_BF,skipped,,E1_RAMIS_DET_BF,Qfast;Qbase;Qtotal,Qfast;Qbase;Qtotal,physical component columns are absent in predictions_validation.csv,
physical_components_E3_RAMIS_LEVY_BF,skipped,,E3_RAMIS_LEVY_BF,Qfast;Qbase;Qtotal,Qfast;Qbase;Qtotal,physical component columns are absent in predictions_validation.csv,
physical_components_E4_ML_PURE_XGB,skipped,,E4_ML_PURE_XGB,Qfast;Qbase;Qtotal,Qfast;Qbase;Qtotal,physical component columns are absent in predictions_validation.csv,
physical_components_E6_HYB_XGB_QUANTILES,skipped,,E6_HYB_XGB_QUANTILES,Qfast;Qbase;Qtotal,Qfast;Qbase;Qtotal,physical component columns are absent in predictions_validation.csv,
physical_components_E8_HYB_GRU_QUANTILES,skipped,,E8_HYB_GRU_QUANTILES,Qfast;Qbase;Qtotal,Qfast;Qbase;Qtotal,physical component columns are absent in predictions_validation.csv,
stochastic_reference_E2_E3,fallback,outputs/figures/phase34/stochastic_reference_E2_E3.png,E2_RAMIS_LEVY_NOBF;E3_RAMIS_LEVY_BF,date;Qobs;Qsim;lower stochastic band;upper stochastic band,E2_RAMIS_LEVY_NOBF:lower/upper;E3_RAMIS_LEVY_BF:lower/upper,E2_RAMIS_LEVY_NOBF: PICP=0.160; warning: stochastic band is strongly underdispersive | E3_RAMIS_LEVY_BF: PICP=0.004; warning: stochastic band is strongly underdispersive | lower/upper stochastic band columns are absent; generated point-reference plot only,fallback: no uncertainty band columns in predictions_validation.csv
```

### Manifest JSON de reporte

Archivo: `outputs/reports/phase34/phase34_report_manifest.json`

```text
{
  "artifacts": [
    {
      "artifact_id": "table_final_ranking_common_window",
      "status": "generated",
      "path": "outputs/reports/phase34/table_final_ranking_common_window.csv",
      "rows": 9,
      "warnings": [],
      "notes": "Main quantitative result table"
    },
    {
      "artifact_id": "table_selected_models_common_window",
      "status": "generated",
      "path": "outputs/reports/phase34/table_selected_models_common_window.csv",
      "rows": 6,
      "warnings": [],
      "notes": "Models used in Phase 3.4B figures"
    },
    {
      "artifact_id": "table_ablation_effects_common_window",
      "status": "generated",
      "path": "outputs/reports/phase34/table_ablation_effects_common_window.csv",
      "rows": 9,
      "warnings": [],
      "notes": "Ablation deltas on common validation window"
    },
    {
      "artifact_id": "table_regime_metrics_selected",
      "status": "generated",
      "path": "outputs/reports/phase34/table_regime_metrics_selected.csv",
      "rows": 18,
      "warnings": [],
      "notes": "Regime metrics for selected models"
    },
    {
      "artifact_id": "table_uncertainty_stochastic_references",
      "status": "generated",
      "path": "outputs/reports/phase34/table_uncertainty_stochastic_references.csv",
      "rows": 2,
      "warnings": [],
      "notes": "Stochastic reference diagnostics"
    },
    {
      "artifact_id": "table_figure_status_summary",
      "status": "generated",
      "path": "outputs/reports/phase34/table_figure_status_summary.csv",
      "rows": 3,
      "warnings": [],
      "notes": "Generated/fallback/skipped figure counts"
    },
    {
      "artifact_id": "table_backend_manuscript_notes",
      "status": "generated",
      "path": "outputs/reports/phase34/table_backend_manuscript_notes.csv",
      "rows": 9,
      "warnings": [],
      "notes": "Backend-aware wording notes"
    },
    {
      "artifact_id": "table_reproducibility_checklist",
      "status": "generated",
      "path": "outputs/reports/phase34/table_reproducibility_checklist.csv",
      "rows": 6,
      "warnings": [],
      "notes": "Submission-readiness checks"
    }
  ],
  "warnings": [
    "GRU over-claim risk: E7_HYB_GRU_MEAN;E8_HYB_GRU_QUANTILES use fallback backend, not real recurrent GRU"
  ]
}```

### Manifest JSON de figuras

Archivo: `outputs/figures/phase34/phase34_figure_manifest.json`

```text
[
  {
    "figure_id": "hydrograph_validation_common",
    "status": "generated",
    "path": "outputs/figures/phase34/hydrograph_validation_common.png",
    "experiments": [
      "E0_RAMIS_DET_NOBF",
      "E1_RAMIS_DET_BF",
      "E3_RAMIS_LEVY_BF",
      "E4_ML_PURE_XGB",
      "E6_HYB_XGB_QUANTILES",
      "E8_HYB_GRU_QUANTILES"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [
      "backend=sklearn_hgb",
      "backend=sklearn_hgb",
      "declared GRU uses backend=fallback_mlp_on_flattened_sequences; not a real GRU in this environment"
    ],
    "notes": "n_common=4367"
  },
  {
    "figure_id": "fdc_common",
    "status": "generated",
    "path": "outputs/figures/phase34/fdc_common.png",
    "experiments": [
      "E0_RAMIS_DET_NOBF",
      "E1_RAMIS_DET_BF",
      "E3_RAMIS_LEVY_BF",
      "E4_ML_PURE_XGB",
      "E6_HYB_XGB_QUANTILES",
      "E8_HYB_GRU_QUANTILES"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "n_common=4367"
  },
  {
    "figure_id": "scatter_E0_RAMIS_DET_NOBF",
    "status": "generated",
    "path": "outputs/figures/phase34/scatter_E0_RAMIS_DET_NOBF.png",
    "experiments": [
      "E0_RAMIS_DET_NOBF"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "n=4367"
  },
  {
    "figure_id": "residuals_by_regime_E0_RAMIS_DET_NOBF",
    "status": "generated",
    "path": "outputs/figures/phase34/residuals_by_regime_E0_RAMIS_DET_NOBF.png",
    "experiments": [
      "E0_RAMIS_DET_NOBF"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim",
      "regime_thresholds.json:p25",
      "regime_thresholds.json:p75"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
  },
  {
    "figure_id": "scatter_E1_RAMIS_DET_BF",
    "status": "generated",
    "path": "outputs/figures/phase34/scatter_E1_RAMIS_DET_BF.png",
    "experiments": [
      "E1_RAMIS_DET_BF"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "n=4367"
  },
  {
    "figure_id": "residuals_by_regime_E1_RAMIS_DET_BF",
    "status": "generated",
    "path": "outputs/figures/phase34/residuals_by_regime_E1_RAMIS_DET_BF.png",
    "experiments": [
      "E1_RAMIS_DET_BF"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim",
      "regime_thresholds.json:p25",
      "regime_thresholds.json:p75"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
  },
  {
    "figure_id": "scatter_E3_RAMIS_LEVY_BF",
    "status": "generated",
    "path": "outputs/figures/phase34/scatter_E3_RAMIS_LEVY_BF.png",
    "experiments": [
      "E3_RAMIS_LEVY_BF"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "n=4367"
  },
  {
    "figure_id": "residuals_by_regime_E3_RAMIS_LEVY_BF",
    "status": "generated",
    "path": "outputs/figures/phase34/residuals_by_regime_E3_RAMIS_LEVY_BF.png",
    "experiments": [
      "E3_RAMIS_LEVY_BF"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim",
      "regime_thresholds.json:p25",
      "regime_thresholds.json:p75"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
  },
  {
    "figure_id": "scatter_E4_ML_PURE_XGB",
    "status": "generated",
    "path": "outputs/figures/phase34/scatter_E4_ML_PURE_XGB.png",
    "experiments": [
      "E4_ML_PURE_XGB"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "n=4367"
  },
  {
    "figure_id": "residuals_by_regime_E4_ML_PURE_XGB",
    "status": "generated",
    "path": "outputs/figures/phase34/residuals_by_regime_E4_ML_PURE_XGB.png",
    "experiments": [
      "E4_ML_PURE_XGB"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim",
      "regime_thresholds.json:p25",
      "regime_thresholds.json:p75"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
  },
  {
    "figure_id": "scatter_E6_HYB_XGB_QUANTILES",
    "status": "generated",
    "path": "outputs/figures/phase34/scatter_E6_HYB_XGB_QUANTILES.png",
    "experiments": [
      "E6_HYB_XGB_QUANTILES"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "n=4367"
  },
  {
    "figure_id": "residuals_by_regime_E6_HYB_XGB_QUANTILES",
    "status": "generated",
    "path": "outputs/figures/phase34/residuals_by_regime_E6_HYB_XGB_QUANTILES.png",
    "experiments": [
      "E6_HYB_XGB_QUANTILES"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim",
      "regime_thresholds.json:p25",
      "regime_thresholds.json:p75"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"
  },
  {
    "figure_id": "scatter_E8_HYB_GRU_QUANTILES",
    "status": "generated",
    "path": "outputs/figures/phase34/scatter_E8_HYB_GRU_QUANTILES.png",
    "experiments": [
      "E8_HYB_GRU_QUANTILES"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "n=4367"
  },
  {
    "figure_id": "residuals_by_regime_E8_HYB_GRU_QUANTILES",
    "status": "generated",
    "path": "outputs/figures/phase34/residuals_by_regime_E8_HYB_GRU_QUANTILES.png",
    "experiments": [
      "E8_HYB_GRU_QUANTILES"
    ],
    "required_columns": [
      "date",
      "Qobs",
      "Qsim",
      "regime_thresholds.json:p25",
      "regime_thresholds.json:p75"
    ],
    "missing_columns": [],
    "warnings": [],
    "notes": "p25=0.074; p75=0.576; counts={'low': 1368, 'mid': 1922, 'high': 1077}"

[TRUNCADO: se muestran 250 de 409 líneas]
```

### Leaderboard común original

Archivo: `outputs/comparison/leaderboard_common_intersection.csv`

```csv
rank,experiment,n_eval,common_start_date,common_end_date,NSE,KGE,RMSE,MAE,PBIAS,R2
1,E8_HYB_GRU_QUANTILES,4367,2009-01-08,2020-12-31,0.8491160018721119,0.8985560770853956,0.20972274341742525,0.12397171327468186,1.5791389617175808,0.84951314368649
2,E6_HYB_XGB_QUANTILES,4367,2009-01-08,2020-12-31,0.7399273544058875,0.80977759904721,0.27534124777429825,0.16169461729695986,0.9265927312935629,0.7400981468460716
3,E7_HYB_GRU_MEAN,4367,2009-01-08,2020-12-31,0.6864827857787669,0.7890942412234975,0.30231147634757416,0.19199198306125353,1.6435975392447297,0.6887968694151526
4,E5_HYB_XGB_MEAN,4367,2009-01-08,2020-12-31,0.6262062864359661,0.7145963848249611,0.3300957516669702,0.20987887748592007,2.0068432918117,0.6266527505471495
5,E4_ML_PURE_XGB,4367,2009-01-08,2020-12-31,0.5752172486548741,0.6609452130021479,0.35189036456719164,0.22278955048874277,0.7422587700568847,0.5752636980045411
6,E1_RAMIS_DET_BF,4367,2009-01-08,2020-12-31,0.4384715015927956,0.7137185090693596,0.4045850580202305,0.24777204185086024,0.860236359127669,0.5103514291817111
7,E3_RAMIS_LEVY_BF,4367,2009-01-08,2020-12-31,0.42864775390338394,0.7249663584026212,0.408108751516014,0.2554212403680974,-1.2175665522260202,0.5592798595377673
8,E2_RAMIS_LEVY_NOBF,4367,2009-01-08,2020-12-31,0.42692867621304875,0.7167407376700682,0.4087222467178347,0.26777713776110235,-7.551079226355426,0.5528836595701413
9,E0_RAMIS_DET_NOBF,4367,2009-01-08,2020-12-31,0.4142486498117365,0.7134014251980563,0.41321929040273064,0.26964860391644013,-5.8647562046943476,0.5540337015777467
```

### Leaderboard general

Archivo: `outputs/comparison/leaderboard.csv`

```csv
rank,experiment,n_eval,NSE,KGE,RMSE,MAE,PBIAS,R2,PICP,PINAW,MPIW,Winkler
1,E8_HYB_GRU_QUANTILES,4367,0.8491160018721119,0.8985560770853956,0.2097227434174252,0.1239717132746818,1.5791389617175875,0.84951314368649,,,,
2,E6_HYB_XGB_QUANTILES,4367,0.7399273544058875,0.80977759904721,0.2753412477742982,0.1616946172969598,0.926592731293568,0.7400981468460716,,,,
3,E7_HYB_GRU_MEAN,4367,0.686482785778767,0.7890942412234974,0.3023114763475741,0.1919919830612535,1.6435975392447366,0.6887968694151526,,,,
4,E5_HYB_XGB_MEAN,4367,0.6262062864359661,0.7145963848249611,0.3300957516669702,0.20987887748592,2.0068432918117054,0.6266527505471495,,,,
5,E4_ML_PURE_XGB,4367,0.5752172486548741,0.6609452130021479,0.3518903645671916,0.2227895504887427,0.7422587700568917,0.5752636980045411,,,,
6,E1_RAMIS_DET_BF,4376,0.4384715015927956,0.7137185090693596,0.4045850580202305,0.2477720418508602,0.8602363591276737,0.5103514291817111,,,,
7,E3_RAMIS_LEVY_BF,4376,0.4286477539033839,0.7249663584026212,0.408108751516014,0.2554212403680974,-1.2175665522260115,0.5592798595377673,0.0036638424547744,0.0033684917097607,0.0087614469370875,10.050000041406184
8,E2_RAMIS_LEVY_NOBF,4376,0.4269286762130487,0.7167407376700682,0.4087222467178347,0.2677771377611023,-7.551079226355424,0.5528836595701413,0.1600641172429585,0.0840696217014992,0.2186650860455995,7.297140056301506
9,E0_RAMIS_DET_NOBF,4376,0.4142486498117365,0.7134014251980563,0.4132192904027306,0.2696486039164401,-5.864756204694343,0.5540337015777467,,,,
```

### Auditoría anti-leakage

Archivo: `outputs/comparison/leakage_audit_report.md`

```text
# Auditoria anti-leakage y QA — Fase 3.4A

**Estado:** `WARN`

## Resumen por experimento

| experiment | status | train_start | train_end | validation_start | validation_end | n_validation_predictions | backend_label | postprocess_method | regime_threshold_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_RAMIS_DET_NOBF | WARN | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4376 | physical | none | train_Qobs_only |
| E1_RAMIS_DET_BF | WARN | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4376 | physical | none | train_Qobs_only |
| E2_RAMIS_LEVY_NOBF | WARN | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4376 | physical | none | train_Qobs_only |
| E3_RAMIS_LEVY_BF | WARN | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4376 | physical | none | train_Qobs_only |
| E4_ML_PURE_XGB | PASS | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4367 | sklearn_hgb | clip_negative_to_zero | train_Qobs_only |
| E5_HYB_XGB_MEAN | PASS | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4367 | sklearn_hgb | clip_negative_to_zero | train_Qobs_only |
| E6_HYB_XGB_QUANTILES | PASS | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4367 | sklearn_hgb | clip_negative_to_zero | train_Qobs_only |
| E7_HYB_GRU_MEAN | WARN | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4367 | fallback_mlp_on_flattened_sequences | clip_negative_to_zero | train_Qobs_only |
| E8_HYB_GRU_QUANTILES | WARN | 1991-09-08 | 2008-12-31 | 2009-01-08 | 2020-12-31 | 4367 | fallback_mlp_on_flattened_sequences | clip_negative_to_zero | train_Qobs_only |

## Hallazgos WARN/ERROR

| severity | experiment | check | message | evidence |
| --- | --- | --- | --- | --- |
| WARN | E0_RAMIS_DET_NOBF | predictions_train_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 177, "n_rows": 6325} |
| WARN | E0_RAMIS_DET_NOBF | predictions_validation_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 9, "n_rows": 4376} |
| WARN | E1_RAMIS_DET_BF | predictions_train_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 177, "n_rows": 6325} |
| WARN | E1_RAMIS_DET_BF | predictions_validation_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 9, "n_rows": 4376} |
| WARN | E2_RAMIS_LEVY_NOBF | predictions_train_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 177, "n_rows": 6325} |
| WARN | E2_RAMIS_LEVY_NOBF | predictions_validation_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 9, "n_rows": 4376} |
| WARN | E3_RAMIS_LEVY_BF | predictions_train_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 177, "n_rows": 6325} |
| WARN | E3_RAMIS_LEVY_BF | predictions_validation_Qobs_finite | Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas. | {"n_missing": 9, "n_rows": 4376} |
| WARN | E7_HYB_GRU_MEAN | gru_backend_real | El experimento declarado GRU no debe redactarse como GRU real salvo que backend sea tensorflow/torch. | {"backend": "sklearn_mlp", "label": "fallback_mlp_on_flattened_sequences"} |
| WARN | E8_HYB_GRU_QUANTILES | gru_backend_real | El experimento declarado GRU no debe redactarse como GRU real salvo que backend sea tensorflow/torch. | {"backend": "sklearn_mlp", "label": "fallback_mlp_on_flattened_sequences"} |

## Criterios auditados

- Separacion temporal estricta entre calibracion y validacion.
- Lags no negativos y secuencias formadas con informacion hasta t para predecir t+horizon.
- Variables predictoras sin Qobs/target observado.
- Ventana comun declarada consistente con los CSV de prediccion.
- Umbrales de regimen documentados como `train_Qobs_only`.
- Backend ML registrado para distinguir GRU real de fallback MLP.
- Postproceso no-negativo aplicado solo a ML/hibridos.
```

### Estado auditoría anti-leakage

Archivo: `outputs/comparison/leakage_audit_status.txt`

```text
WARN
```

### Auditoría física

Archivo: `outputs/comparison/physical_audit_report.md`

```text
# Auditoria fisica RAMIS/baseflow - Fase 2.6

**Estado:** `WARN`

## Resumen E1/E2

| experiment       | n_train | train_NSE          | train_KGE          | train_PBIAS         | n_validation | validation_NSE      | validation_KGE     | validation_PBIAS    | validation_Qsim_Qobs_ratio |
| ---------------- | ------- | ------------------ | ------------------ | ------------------- | ------------ | ------------------- | ------------------ | ------------------- | -------------------------- |
| E1_RAMIS_DET_BF  | 6155    | 0.5068619208250867 | 0.7588246222120646 | -0.6176661696586848 | 4374         | 0.43729950676316187 | 0.7140983793742237 | 1.088078596216255   | 1.0093395681367194         |
| E2_RAMIS_LEVY_BF | 6155    | 0.4927915231475505 | 0.738354829030989  | -1.7356366407650279 | 4374         | 0.4262742835843092  | 0.723712049338845  | -0.9194266793142531 | 0.9888524524147835         |

## Incidencias

| experiment      | level | check        | message                                    |
| --------------- | ----- | ------------ | ------------------------------------------ |
| E1_RAMIS_DET_BF | WARN  | regime_pbias | PBIAS absoluto >100% en regimenes ['low']. |

## Criterio de cierre

Fase 2 se considera cerrada si el estado es `PASS` o, como maximo, `WARN` con advertencias metodologicamente explicables. Un estado `FAIL` requiere corregir salidas, parametros o componentes antes de pasar a ablaciones/ML.
```

### Estado auditoría física

Archivo: `outputs/comparison/physical_audit_status.txt`

```text
WARN
```

## Columnas disponibles en predictions_validation.csv

```text
outputs/experiments/A1_RAMIS_LEVY_NOBF/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/A2_RAMIS_LEVY_BF/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E0_RAMIS_DET_NOBF/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E1_RAMIS_DET_BF/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E2_RAMIS_LEVY_BF/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E2_RAMIS_LEVY_NOBF/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E3_ML_PURE/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E3_RAMIS_LEVY_BF/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E4_ML_PURE_XGB/predictions_validation.csv: date, Qobs, Qsim, Qsim_raw
outputs/experiments/E4_RAMIS_XGB_MEAN/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E5_HYB_XGB_MEAN/predictions_validation.csv: date, Qobs, Qsim, Qsim_raw
outputs/experiments/E5_RAMIS_XGB_QUANTILE/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E6_HYB_XGB_QUANTILES/predictions_validation.csv: date, Qobs, Qsim, Qsim_raw
outputs/experiments/E6_RAMIS_GRU_MEAN/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E7_HYB_GRU_MEAN/predictions_validation.csv: date, Qobs, Qsim, Qsim_raw
outputs/experiments/E7_RAMIS_GRU_QUANTILE/predictions_validation.csv: date, Qobs, Qsim
outputs/experiments/E8_HYB_GRU_QUANTILES/predictions_validation.csv: date, Qobs, Qsim, Qsim_raw
```

## Figuras disponibles

```text
outputs/figures/phase34/fdc_common.png
outputs/figures/phase34/hydrograph_validation_common.png
outputs/figures/phase34/phase34_figure_manifest.csv
outputs/figures/phase34/phase34_figure_manifest.json
outputs/figures/phase34/residuals_by_regime_E0_RAMIS_DET_NOBF.png
outputs/figures/phase34/residuals_by_regime_E1_RAMIS_DET_BF.png
outputs/figures/phase34/residuals_by_regime_E3_RAMIS_LEVY_BF.png
outputs/figures/phase34/residuals_by_regime_E4_ML_PURE_XGB.png
outputs/figures/phase34/residuals_by_regime_E6_HYB_XGB_QUANTILES.png
outputs/figures/phase34/residuals_by_regime_E8_HYB_GRU_QUANTILES.png
outputs/figures/phase34/scatter_E0_RAMIS_DET_NOBF.png
outputs/figures/phase34/scatter_E1_RAMIS_DET_BF.png
outputs/figures/phase34/scatter_E3_RAMIS_LEVY_BF.png
outputs/figures/phase34/scatter_E4_ML_PURE_XGB.png
outputs/figures/phase34/scatter_E6_HYB_XGB_QUANTILES.png
outputs/figures/phase34/scatter_E8_HYB_GRU_QUANTILES.png
outputs/figures/phase34/stochastic_reference_E2_E3.png
```

## Archivos copiados al paquete

```text
.gitignore
outputs/comparison/ablation_effects_common_intersection.csv
outputs/comparison/ablation_effects.csv
outputs/comparison/best_model_summary.md
outputs/comparison/evaluation_window_comparison.csv
outputs/comparison/experiment_matrix.csv
outputs/comparison/leaderboard_common_intersection.csv
outputs/comparison/leaderboard.csv
outputs/comparison/leakage_audit_issues.csv
outputs/comparison/leakage_audit.json
outputs/comparison/leakage_audit_report.md
outputs/comparison/leakage_audit_status.txt
outputs/comparison/leakage_audit_summary.csv
outputs/comparison/ml_backend_summary.csv
outputs/comparison/physical_audit_issues.csv
outputs/comparison/physical_audit_report.md
outputs/comparison/physical_audit_status.txt
outputs/comparison/physical_audit_summary.csv
outputs/comparison/regime_metrics_comparison.csv
outputs/comparison/uncertainty_comparison.csv
outputs/comparison/validation_metrics_common_intersection.csv
outputs/comparison/validation_metrics_comparison.csv
outputs/experiments/A1_RAMIS_LEVY_NOBF/predictions_validation.csv
outputs/experiments/A2_RAMIS_LEVY_BF/predictions_validation.csv
outputs/experiments/E0_RAMIS_DET_NOBF/predictions_validation.csv
outputs/experiments/E1_RAMIS_DET_BF/predictions_validation.csv
outputs/experiments/E2_RAMIS_LEVY_BF/predictions_validation.csv
outputs/experiments/E2_RAMIS_LEVY_NOBF/predictions_validation.csv
outputs/experiments/E3_ML_PURE/predictions_validation.csv
outputs/experiments/E3_RAMIS_LEVY_BF/predictions_validation.csv
outputs/experiments/E4_ML_PURE_XGB/ml_backend.json
outputs/experiments/E4_ML_PURE_XGB/predictions_validation.csv
outputs/experiments/E4_RAMIS_XGB_MEAN/predictions_validation.csv
outputs/experiments/E5_HYB_XGB_MEAN/ml_backend.json
outputs/experiments/E5_HYB_XGB_MEAN/predictions_validation.csv
outputs/experiments/E5_RAMIS_XGB_QUANTILE/predictions_validation.csv
outputs/experiments/E6_HYB_XGB_QUANTILES/ml_backend.json
outputs/experiments/E6_HYB_XGB_QUANTILES/predictions_validation.csv
outputs/experiments/E6_RAMIS_GRU_MEAN/predictions_validation.csv
outputs/experiments/E7_HYB_GRU_MEAN/ml_backend.json
outputs/experiments/E7_HYB_GRU_MEAN/predictions_validation.csv
outputs/experiments/E7_RAMIS_GRU_QUANTILE/predictions_validation.csv
outputs/experiments/E8_HYB_GRU_QUANTILES/ml_backend.json
outputs/experiments/E8_HYB_GRU_QUANTILES/predictions_validation.csv
outputs/figures/phase34/fdc_common.png
outputs/figures/phase34/hydrograph_validation_common.png
outputs/figures/phase34/phase34_figure_manifest.csv
outputs/figures/phase34/phase34_figure_manifest.json
outputs/figures/phase34/residuals_by_regime_E0_RAMIS_DET_NOBF.png
outputs/figures/phase34/residuals_by_regime_E1_RAMIS_DET_BF.png
outputs/figures/phase34/residuals_by_regime_E3_RAMIS_LEVY_BF.png
outputs/figures/phase34/residuals_by_regime_E4_ML_PURE_XGB.png
outputs/figures/phase34/residuals_by_regime_E6_HYB_XGB_QUANTILES.png
outputs/figures/phase34/residuals_by_regime_E8_HYB_GRU_QUANTILES.png
outputs/figures/phase34/scatter_E0_RAMIS_DET_NOBF.png
outputs/figures/phase34/scatter_E1_RAMIS_DET_BF.png
outputs/figures/phase34/scatter_E3_RAMIS_LEVY_BF.png
outputs/figures/phase34/scatter_E4_ML_PURE_XGB.png
outputs/figures/phase34/scatter_E6_HYB_XGB_QUANTILES.png
outputs/figures/phase34/scatter_E8_HYB_GRU_QUANTILES.png
outputs/figures/phase34/stochastic_reference_E2_E3.png
outputs/reports/phase34/phase34_report_manifest.json
outputs/reports/phase34/phase34_results_report.md
outputs/reports/phase34/table_ablation_effects_common_window.csv
outputs/reports/phase34/table_backend_manuscript_notes.csv
outputs/reports/phase34/table_figure_status_summary.csv
outputs/reports/phase34/table_final_ranking_common_window.csv
outputs/reports/phase34/table_regime_metrics_selected.csv
outputs/reports/phase34/table_reproducibility_checklist.csv
outputs/reports/phase34/table_selected_models_common_window.csv
outputs/reports/phase34/table_uncertainty_stochastic_references.csv
pyproject.toml
README.md
requirements.txt
scripts/generate_phase34_figures.py
scripts/generate_phase34_report.py
src/stohymolap/calibration/__init__.py
src/stohymolap/calibration/monte_carlo_search.py
src/stohymolap/calibration/objective.py
src/stohymolap/calibration/parameter_store.py
src/stohymolap/data/__init__.py
src/stohymolap/data/io.py
src/stohymolap/data/preprocessing.py
src/stohymolap/diagnostics/__init__.py
src/stohymolap/diagnostics/leakage_audit.py
src/stohymolap/diagnostics/physical_audit.py
src/stohymolap/experiments/comparison.py
src/stohymolap/experiments/evaluation_window.py
src/stohymolap/experiments/__init__.py
src/stohymolap/experiments/registry.py
src/stohymolap/experiments/runner.py
src/stohymolap/features/feature_builder.py
src/stohymolap/features/__init__.py
src/stohymolap/features/lagged.py
src/stohymolap/features/uncertainty_features.py
src/stohymolap/hydro/baseflow.py
src/stohymolap/hydro/__init__.py
src/stohymolap/hydro/pet.py
src/stohymolap/hydro/ramis.py
src/stohymolap/hydro/water_balance.py
src/stohymolap/__init__.py
src/stohymolap/metrics/deterministic.py
src/stohymolap/metrics/__init__.py
src/stohymolap/metrics/regime_metrics.py
src/stohymolap/metrics/uncertainty.py
src/stohymolap/ml/gru_model.py
src/stohymolap/ml/__init__.py
src/stohymolap/ml/pure_ml.py
src/stohymolap/ml/train_predict.py
src/stohymolap/ml/xgboost_model.py
src/stohymolap/plotting/diagrams.py
src/stohymolap/plotting/flow_duration.py
src/stohymolap/plotting/hydrographs.py
src/stohymolap/plotting/__init__.py
src/stohymolap/plotting/scatter.py
src/stohymolap/plotting/uncertainty.py
src/stohymolap/stochastic/__init__.py
src/stohymolap/stochastic/levy.py
src/stohymolap/stochastic/monte_carlo.py
src/stohymolap/utils/config.py
src/stohymolap/utils/__init__.py
src/stohymolap/utils/logging.py
src/stohymolap/utils/reproducibility.py
src/stohymolap/validation_checks.py
tests/test_phase34_figures.py
tests/test_phase34_report.py
tests/test_smoke.py
```

## Estructura LaTeX generada

Archivo: `contextos_publicacion/publicacion_20260628_125846/paper_stohymolap_template.tex`

La plantilla contiene secciones para Introduction, Materials and methods, Results, Discussion, Conclusions, Code/Data availability y referencias. Incluye placeholders para tablas y figuras de Fase 3.4B/3.4C.

## Prompt generado

Archivo: `contextos_publicacion/publicacion_20260628_125846/prompt_redaccion_resultados_discusion.md`
