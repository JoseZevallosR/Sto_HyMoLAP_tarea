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
- `leakage_manifest.json`

## Fase 3.4B — Figuras paper-ready

La Fase 3.4B agrega un generador reproducible de figuras de validación sobre la
misma ventana común usada por `outputs/comparison/leaderboard_common_intersection.csv`.
No agrega dependencias pesadas: usa `matplotlib`, `pandas` y `numpy`.

Uso recomendado después de correr la matriz E0-E8 y la auditoría Fase 3.4A:

```bash
python scripts/generate_phase34_figures.py --output-root outputs
```

Artefactos generados en `outputs/figures/phase34/`:

- `hydrograph_validation_common.png`: hidrograma de validación para
  `E0_RAMIS_DET_NOBF`, `E1_RAMIS_DET_BF`, `E3_RAMIS_LEVY_BF`,
  `E4_ML_PURE_XGB`, `E6_HYB_XGB_QUANTILES` y `E8_HYB_GRU_QUANTILES`.
- `fdc_common.png`: curva de duración de caudales comparando `Qobs` y `Qsim`.
- `scatter_<EXPERIMENT>.png`: diagramas `Qobs`-`Qsim` por experimento.
- `residuals_by_regime_<EXPERIMENT>.png`: residuos por régimen hidrológico,
  usando `regime_thresholds.json` cuando está disponible.
- `physical_components_<EXPERIMENT>.png`: componentes `Qfast`, `Qbase` y
  `Qtotal` solo si existen en `predictions_validation.csv`; si no existen,
  el script no falla y registra la omisión en el manifest.
- `stochastic_reference_E2_E3.png`: referencia estocástica E2/E3. Si no existen
  columnas de banda inferior/superior, se genera una figura fallback con las
  trayectorias puntuales y una advertencia de subdispersión cuando
  `uncertainty_comparison.csv` lo indica.
- `phase34_figure_manifest.csv` y `phase34_figure_manifest.json`: inventario de
  figuras, estado (`generated`, `fallback`, `skipped`, `warning`), columnas
  requeridas, columnas faltantes, advertencias y notas.

Nota de redacción científica: `E7_HYB_GRU_MEAN` y `E8_HYB_GRU_QUANTILES` deben
reportarse según el backend real de `ml_backend.json`. Si el backend es
`fallback_mlp_on_flattened_sequences`, no deben describirse como GRU real.
