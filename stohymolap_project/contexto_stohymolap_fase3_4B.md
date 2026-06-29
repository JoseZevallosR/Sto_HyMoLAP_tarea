# Contexto StoHyMoLAP — Fase 3.4B

Generado: 20260628_123956
Proyecto: /home/jose/Documents/githubs/Sto_HyMoLAP_tarea/stohymolap_project

## Estado de avance

Fase 3.4A cerrada funcionalmente:

- Auditor anti-leakage y QA implementado.
- Compatibilidad con outputs legacy corregida.
- Tests reportados por el usuario: 26 passed, 2 warnings.
- Commit principal: d369114 — feat: add phase 3.4A leakage audit and backend QA.
- Commit correctivo: e75439c — fix: make leakage audit compatible with legacy outputs.
- Auditoría actual: WARN, no FAIL.

Interpretación del WARN:

- No se observó leakage temporal: train termina antes de validation.
- Las ventanas de evaluación son consistentes.
- La intersección común cubre los 9 experimentos.
- Persisten brechas observacionales de Qobs en físicos E0-E3.
- E7/E8 no deben redactarse como GRU real en este entorno; el backend registrado es fallback_mlp_on_flattened_sequences.
- Algunos outputs legacy pueden requerir rerun para completar trazabilidad de postprocessing_report.json y regime_thresholds.json.

## Objetivo de la Fase 3.4B

Generar figuras paper-ready para StoHyMoLAP:

1. Hidrograma de validación para E0, E1, E3, E4, E6 y E8.
2. Curva de duración de caudales, FDC, comparando Qobs y Qsim.
3. Scatter Qobs-Qsim.
4. Residuos por régimen hidrológico.
5. Componentes físicos Qfast, Qbase y Qtotal si están disponibles.
6. Bandas estocásticas E2/E3 como referencia física, con advertencia explícita de subdispersión.
7. Manifest de figuras con rutas, modelos usados, columnas requeridas y notas metodológicas.

## Reglas metodológicas para 3.4B

- Usar la misma ventana común/intersección exacta usada por el leaderboard.
- No recalcular métricas con fechas distintas a las de comparación común.
- Excluir Qobs faltante de métricas y gráficos estadísticos que lo requieran.
- Documentar explícitamente que E7/E8 son fallback MLP si no hay TensorFlow/PyTorch.
- No vender las bandas E2/E3 como incertidumbre calibrada; tratarlas como referencia estocástica subdispersa.
- Guardar figuras en outputs/figures/phase34/.
- Guardar un manifest en outputs/figures/phase34/figure_manifest.csv y/o JSON.

## pwd

```text
/home/jose/Documents/githubs/Sto_HyMoLAP_tarea/stohymolap_project
```

## git status --short

```text
 M generar_contexto.sh
?? contexto_stohymolap_fase3_4B.md
?? ../texto/
```

## ultimos commits

```text
e75439c fix: make leakage audit compatible with legacy outputs
d369114 feat: add phase 3.4A leakage audit and backend QA
69b73c9 chore: remove python cache artifacts from repository
13f482e chore: remove python cache artifacts from repository
1fdb8c6 fix: audit nonnegative ML predictions and add common intersection leaderboard
97af7c7 feat: enforce common evaluation window for phase 3 comparisons
4c03718 chore: ignore python cache files
2b3e411 feat: define canonical phase 3 ablation matrix
```

## archivos pyc versionados

```text
```

## estructura resumida

```text
configs/base.yaml
configs/experiments.yaml
configs/model_bounds.yaml
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
pyproject.toml
.pytest_cache/CACHEDIR.TAG
.pytest_cache/.gitignore
.pytest_cache/README.md
README.md
requirements.txt
scripts/audit_leakage_phase34.py
scripts/audit_physical_phase26.py
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
tests/__pycache__/test_smoke.cpython-310-pytest-7.4.3.pyc
tests/test_smoke.py
```

## README

Archivo: README.md

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
- `leakage_manifest.json`
```

## Configuracion experimental

Archivo: configs/experiments.yaml

```text
# =====================================================================
# experiments.yaml  -  Matriz experimental StoHyMoLAP / RAMIS
# ---------------------------------------------------------------------
# Matriz Fase 3.2: ablaciones canonicas con ventana comun de evaluacion.
# Objetivo: aislar memoria hidrologica (baseflow), ruido Levy y
# post-procesamiento ML/hibrido manteniendo outputs comparables.
#
# Ejecucion:
#   python scripts/run_experiment.py --config configs/experiments.yaml --list
#   python scripts/run_experiment.py --config configs/experiments.yaml --experiment E3_RAMIS_LEVY_BF
#   python scripts/run_all_experiments.py --config configs/experiments.yaml --continue-on-error
# =====================================================================

global:
  data_path: data/ramis_hydro.csv
  date_col: date
  qobs_col: flow_obs
  p_col: precipitation_mean
  tmin_col: tmin
  tmax_col: tmax
  qsim_col: flow_sim
  fill_obs_with_sim: false
  csv_sep: ","
  train_fraction: 0.7
  ramis_state_mode: continuous_train_validation
  forecast_horizon: 1
  lags: [0, 1, 2]
  seed: 42
  output_root: outputs/experiments

pet:
  method: hargreaves_samani
  latitude: -15.0

calibration:
  n_param_samples: 800
  n_iter: 400
  top_frac: 0.10
  parameter_selection: best_j
  mu_bounds: [0.75, 0.95]
  lambda_bounds: [2.0, 3.4]
  sigma_bounds: [0.0, 0.10]
  objective:
    weights:
      nse: 0.40
      kge: 0.30
      pbias: 0.15
      coverage: 0.15
    target_coverage: 0.95

stochastic:
  n_trajectories: 400
  levy:
    alpha_bounds: [1.1, 1.9]
    beta_bounds: [-1.0, 0.0]

baseflow:
  enabled: false
  calibrate: true
  c_r_bounds: [0.0, 1.0]
  k_b_bounds: [0.001, 0.5]
  S0_b_bounds: [0.0, 10.0]

# Ventana comun de evaluacion:
# Los modelos fisicos pueden simular desde el primer dia de validacion, pero
# ML/hibridos pierden dias iniciales por lags o secuencias. Esta politica
# recorta TODOS los experimentos a la misma ventana objetivo, inferida desde
# el mayor warm-up de la matriz declarada. Para la matriz E0-E8 actual:
# max(sequence_length=7, horizon=1) => start_offset = 7.
evaluation:
  common_window:
    enabled: true
    mode: declared_matrix_max_warmup
    start_offset: auto
    end_trim: 0

# ---------------------------------------------------------------------
# Matriz canonica Fase 3.2
# ---------------------------------------------------------------------
experiments:

  # ===================================================================
  # Bloque fisico minimo: memoria hidrologica y Levy aislados
  # ===================================================================

  E0_RAMIS_DET_NOBF:
    description: "Control fisico: RAMIS deterministico SIN reservorio de flujo base."
    role: minimal
    model_type: physical
    stochastic: false
    baseflow: false

  E1_RAMIS_DET_BF:
    description: "RAMIS deterministico CON reservorio de flujo base."
    role: minimal
    model_type: physical
    stochastic: false
    baseflow: true

  E2_RAMIS_LEVY_NOBF:
    description: "RAMIS estocastico Levy SIN reservorio de flujo base."
    role: minimal
    model_type: stochastic_physical
    stochastic: true
    baseflow: false

  E3_RAMIS_LEVY_BF:
    description: "RAMIS estocastico Levy CON reservorio de flujo base; ensemble + cuantiles."
    role: minimal
    model_type: stochastic_physical
    stochastic: true
    baseflow: true

  # ===================================================================
  # Bloque ML/hibrido tabular minimo
  # ===================================================================

  E4_ML_PURE_XGB:
    description: "XGBoost puro sobre forzantes observadas (P, PET, Tmin, Tmax) + lags."
    role: minimal
    model_type: machine_learning
    ml_model: xgboost
    stochastic: false
    baseflow: false
    features:
      source: observed_forcing
      variables: [P, PET, Tmin, Tmax]
      lags: [0, 1, 2]

  E5_HYB_XGB_MEAN:
    description: "Hibrido: XGBoost corrige la media del ensemble RAMIS-Levy+BF."
    role: minimal
    model_type: hybrid
    ml_model: xgboost
    stochastic: true
    baseflow: true
    features:
      source: stochastic_ensemble
      variables: [Qmean]
      lags: [0, 1, 2]

  E6_HYB_XGB_QUANTILES:
    description: "Hibrido: XGBoost usa media, cuantiles y anchos del ensemble RAMIS-Levy+BF."
    role: minimal
    model_type: hybrid
    ml_model: xgboost
    stochastic: true
    baseflow: true
    features:
      source: stochastic_ensemble
      variables: [Qmean, q05, q25, q50, q75, q95, width_q95_q05, width_q75_q25]
      lags: [0, 1, 2]

  # ===================================================================
  # Extension secuencial: opcional para paper si hay presupuesto computacional
  # ===================================================================

  E7_HYB_GRU_MEAN:
    description: "Extension secuencial: GRU sobre secuencias de Qmean del ensemble RAMIS-Levy+BF."
    role: extended
    model_type: hybrid_sequence
    ml_model: gru
    stochastic: true
    baseflow: true
    sequence_length: 7
    features:
      source: stochastic_ensemble
      variables: [Qmean]

  E8_HYB_GRU_QUANTILES:
    description: "Extension secuencial: GRU sobre Qmean, cuantiles, anchos y Qbase."
    role: extended
    model_type: hybrid_sequence
    ml_model: gru
    stochastic: true
    baseflow: true
    sequence_length: 7
    features:
      source: stochastic_ensemble
      variables: [Qmean, q05, q25, q50, q75, q95, width_q95_q05, width_q75_q25, Qbase]
```

## Auditoria leakage status

Archivo: outputs/comparison/leakage_audit_status.txt

```text
WARN
```

## Auditoria leakage report

Archivo: outputs/comparison/leakage_audit_report.md

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

## Leaderboard comun

Archivo: outputs/comparison/leaderboard_common_intersection.csv

```text
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

## Metricas validation comun

Archivo: outputs/comparison/validation_metrics_common_intersection.csv

```text
experiment,n_eval,common_start_date,common_end_date,NSE,KGE,RMSE,MAE,PBIAS,R2
E0_RAMIS_DET_NOBF,4367,2009-01-08,2020-12-31,0.4142486498117365,0.7134014251980563,0.41321929040273064,0.26964860391644013,-5.8647562046943476,0.5540337015777467
E1_RAMIS_DET_BF,4367,2009-01-08,2020-12-31,0.4384715015927956,0.7137185090693596,0.4045850580202305,0.24777204185086024,0.860236359127669,0.5103514291817111
E2_RAMIS_LEVY_NOBF,4367,2009-01-08,2020-12-31,0.42692867621304875,0.7167407376700682,0.4087222467178347,0.26777713776110235,-7.551079226355426,0.5528836595701413
E3_RAMIS_LEVY_BF,4367,2009-01-08,2020-12-31,0.42864775390338394,0.7249663584026212,0.408108751516014,0.2554212403680974,-1.2175665522260202,0.5592798595377673
E4_ML_PURE_XGB,4367,2009-01-08,2020-12-31,0.5752172486548741,0.6609452130021479,0.35189036456719164,0.22278955048874277,0.7422587700568847,0.5752636980045411
E5_HYB_XGB_MEAN,4367,2009-01-08,2020-12-31,0.6262062864359661,0.7145963848249611,0.3300957516669702,0.20987887748592007,2.0068432918117,0.6266527505471495
E6_HYB_XGB_QUANTILES,4367,2009-01-08,2020-12-31,0.7399273544058875,0.80977759904721,0.27534124777429825,0.16169461729695986,0.9265927312935629,0.7400981468460716
E7_HYB_GRU_MEAN,4367,2009-01-08,2020-12-31,0.6864827857787669,0.7890942412234975,0.30231147634757416,0.19199198306125353,1.6435975392447297,0.6887968694151526
E8_HYB_GRU_QUANTILES,4367,2009-01-08,2020-12-31,0.8491160018721119,0.8985560770853956,0.20972274341742525,0.12397171327468186,1.5791389617175808,0.84951314368649
```

## Efectos de ablacion

Archivo: outputs/comparison/ablation_effects_common_intersection.csv

```text
ablation,baseline,candidate,question,delta_NSE,delta_KGE,delta_RMSE,delta_MAE,delta_abs_PBIAS_improvement,n_eval_baseline,n_eval_candidate
deterministic_baseflow,E0_RAMIS_DET_NOBF,E1_RAMIS_DET_BF,Efecto del reservorio baseflow en RAMIS deterministico,0.024222851781059096,0.0003170838713033586,-0.008634232382500162,-0.02187656206557989,5.004519845566678,4367,4367
stochastic_baseflow,E2_RAMIS_LEVY_NOBF,E3_RAMIS_LEVY_BF,Efecto del reservorio baseflow en RAMIS-Levy,0.0017190776903351912,0.008225620732553063,-0.0006134952018206907,-0.012355897393004966,6.333512674129405,4367,4367
levy_given_baseflow,E1_RAMIS_DET_BF,E3_RAMIS_LEVY_BF,Efecto del ruido Levy manteniendo baseflow,-0.009823747689411633,0.011247849333261617,0.0035236934957835198,0.007649198517237138,-0.3573301930983511,4367,4367
pure_ml_vs_physical,E3_RAMIS_LEVY_BF,E4_ML_PURE_XGB,ML puro frente al fisico-estocastico completo,0.14656949475149017,-0.06402114540047332,-0.05621838694882236,-0.03263168987935461,0.4753077821691355,4367,4367
hybrid_mean_vs_physical,E3_RAMIS_LEVY_BF,E5_HYB_XGB_MEAN,Hibrido XGB-media frente al fisico-estocastico completo,0.19755853253258215,-0.010369973577660119,-0.07801299984904381,-0.045542362882177306,-0.7892767395856799,4367,4367
hybrid_quantiles_vs_physical,E3_RAMIS_LEVY_BF,E6_HYB_XGB_QUANTILES,Hibrido XGB-cuantiles frente al fisico-estocastico completo,0.3112796005025036,0.08481124064458878,-0.13276750374171575,-0.09372662307113752,0.29097382093245727,4367,4367
quantiles_vs_mean_xgb,E5_HYB_XGB_MEAN,E6_HYB_XGB_QUANTILES,Aporte de cuantiles frente a solo Qmean en XGB,0.11372106796992143,0.0951812142222489,-0.054754503892671946,-0.048184260188960215,1.0802505605181372,4367,4367
gru_mean_vs_xgb_mean,E5_HYB_XGB_MEAN,E7_HYB_GRU_MEAN,Extension GRU-media frente a XGB-media,0.06027649934280077,0.07449785639853634,-0.027784275319396035,-0.01788689442466654,0.36324575256697034,4367,4367
gru_quantiles_vs_xgb_quantiles,E6_HYB_XGB_QUANTILES,E8_HYB_GRU_QUANTILES,Extension GRU-cuantiles frente a XGB-cuantiles,0.10918864746622436,0.08877847803818562,-0.065618504356873,-0.037722904022278,-0.6525462304240179,4367,4367
```

## Metricas por regimen

Archivo: outputs/comparison/regime_metrics_comparison.csv

```text
experiment,regime,n,NSE,KGE,RMSE,MAE,PBIAS
E0_RAMIS_DET_NOBF,low,1368,-167.43677557797142,-10.816876301970098,0.1767114116970202,0.0970683570186978,49.9406057294248
E0_RAMIS_DET_NOBF,mid,1922,-6.401957311027732,-0.9594228392035358,0.3918888660349847,0.2635050281721482,6.596922959679679
E0_RAMIS_DET_NOBF,high,1077,-0.464693839589563,0.3940230089245748,0.6153203785247593,0.4998229124927079,-12.914225121670954
E1_RAMIS_DET_BF,low,1368,-208.9540601084904,-10.606830488841574,0.1972914549459516,0.1009025424539017,195.04307267382964
E1_RAMIS_DET_BF,mid,1922,-5.425164455474709,-0.7471705888982636,0.3651167533177685,0.2063543296095859,32.128698971824264
E1_RAMIS_DET_BF,high,1077,-0.4560269035323228,0.4338767166132065,0.6134971801452394,0.5082384467745079,-19.738124439742784
E2_RAMIS_LEVY_NOBF,low,1368,-162.85100919283616,-10.646335848117433,0.1742892885654886,0.0961227017917242,48.21396225890156
E2_RAMIS_LEVY_NOBF,mid,1922,-6.15447390206862,-0.914359137567802,0.3852817951530991,0.2604359951072566,4.982726948890258
E2_RAMIS_LEVY_NOBF,high,1077,-0.4463404234017425,0.4088477446590884,0.6114530747633878,0.4989126480552535,-14.62182776026246
E3_RAMIS_LEVY_BF,low,1368,-184.1125418169712,-11.35225354325656,0.1852524920725675,0.0920493468602603,68.02961278628021
E3_RAMIS_LEVY_BF,mid,1922,-6.218593844310093,-0.9383909370942378,0.3870044334800824,0.2396332108971476,19.025645454609744
E3_RAMIS_LEVY_BF,high,1077,-0.4099266132518224,0.4114519834305418,0.6037068799213294,0.4911105095991897,-11.51882644973357
E4_ML_PURE_XGB,low,1368,-191.1747258052382,-8.753772780498208,0.1887531777323919,0.1239713715799234,239.64621908395776
E4_ML_PURE_XGB,mid,1922,-2.6842205773807,-0.2561784304171952,0.2764791902234826,0.1653755896396804,40.49094883614825
E4_ML_PURE_XGB,high,1077,-0.2395550118508707,0.3760995391085419,0.5660577931605073,0.4507681034127561,-25.01574246456537
E5_HYB_XGB_MEAN,low,1368,-154.24231286079444,-8.369875339171102,0.1696489550794554,0.096901008223559,187.3171196724915
E5_HYB_XGB_MEAN,mid,1922,-2.6138193470656206,-0.1993509612083048,0.2738248485697795,0.1717524040388096,45.41170827176288
E5_HYB_XGB_MEAN,high,1077,-0.0501332055683623,0.5123311296161104,0.5210148079052149,0.4214228023849509,-22.05991994926601
E6_HYB_XGB_QUANTILES,low,1368,-29.22442519355636,-2.420551671441954,0.0748557492951145,0.0471194931440789,90.2340822605086
E6_HYB_XGB_QUANTILES,mid,1922,-1.368890761226266,0.0497136089426166,0.2216981933232863,0.1335961414045237,30.909796146980952
E6_HYB_XGB_QUANTILES,high,1077,0.177652862817409,0.5721983470691147,0.4610578953783853,0.3573715351302034,-13.618719979953555
E7_HYB_GRU_MEAN,low,1368,-104.26349161142292,-7.050063133433056,0.139696338303382,0.0742788246086856,129.77639806790816
E7_HYB_GRU_MEAN,mid,1922,-2.1496271429347438,-0.1193314930956406,0.2556343493171752,0.1673145654182536,35.13368497685233
E7_HYB_GRU_MEAN,high,1077,0.1134649155789726,0.5878543992854603,0.4787136760758033,0.3855496408820137,-16.12788422359374
E8_HYB_GRU_QUANTILES,low,1368,-10.694359702084371,-0.854566060475247,0.0465622931562704,0.0323675138068657,56.99673890908074
E8_HYB_GRU_QUANTILES,mid,1922,-0.4053481111011086,0.3174873364166684,0.1707582101598029,0.1081690059186078,16.134121411776878
E8_HYB_GRU_QUANTILES,high,1077,0.522027438223039,0.7392319382157331,0.3515032576319902,0.2685282113344281,-6.129959341368462
```

## Resumen backend ML

Archivo: outputs/comparison/ml_backend_summary.csv

```text
experiment,model_type,ml_model,backend,backend_label,feature_source,postprocess_method,regime_threshold_source
E0_RAMIS_DET_NOBF,physical,,physical,physical,physical_only,none,train_Qobs_only
E1_RAMIS_DET_BF,physical,,physical,physical,physical_only,none,train_Qobs_only
E2_RAMIS_LEVY_NOBF,stochastic_physical,,physical,physical,physical_only,none,train_Qobs_only
E3_RAMIS_LEVY_BF,stochastic_physical,,physical,physical,physical_only,none,train_Qobs_only
E4_ML_PURE_XGB,machine_learning,xgboost,sklearn_hgb,sklearn_hgb,observed_forcing,clip_negative_to_zero,train_Qobs_only
E5_HYB_XGB_MEAN,hybrid,xgboost,sklearn_hgb,sklearn_hgb,stochastic_ensemble,clip_negative_to_zero,train_Qobs_only
E6_HYB_XGB_QUANTILES,hybrid,xgboost,sklearn_hgb,sklearn_hgb,stochastic_ensemble,clip_negative_to_zero,train_Qobs_only
E7_HYB_GRU_MEAN,hybrid_sequence,gru,sklearn_mlp,fallback_mlp_on_flattened_sequences,stochastic_ensemble,clip_negative_to_zero,train_Qobs_only
E8_HYB_GRU_QUANTILES,hybrid_sequence,gru,sklearn_mlp,fallback_mlp_on_flattened_sequences,stochastic_ensemble,clip_negative_to_zero,train_Qobs_only
```

## Issues auditoria leakage

Archivo: outputs/comparison/leakage_audit_issues.csv

```text
severity,experiment,check,message,evidence
PASS,GLOBAL,ramis_state_mode,RAMIS mantiene estado continuo train+validation.,continuous_train_validation
PASS,E0_RAMIS_DET_NOBF,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E0_RAMIS_DET_NOBF,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
WARN,E0_RAMIS_DET_NOBF,predictions_train_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 177, ""n_rows"": 6325}"
WARN,E0_RAMIS_DET_NOBF,predictions_validation_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 9, ""n_rows"": 4376}"
PASS,E0_RAMIS_DET_NOBF,temporal_split_order,Train termina antes de validation.,
PASS,E0_RAMIS_DET_NOBF,evaluation_window_exists,evaluation_window.json presente.,
PASS,E0_RAMIS_DET_NOBF,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4376, ""start_date"": ""2009-01-08""}"
PASS,E0_RAMIS_DET_NOBF,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6325, ""start_date"": ""1991-09-08""}"
PASS,E0_RAMIS_DET_NOBF,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""none"", ""model_type"": ""physical"", ""validation_negatives"": 0}"
PASS,E0_RAMIS_DET_NOBF,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E1_RAMIS_DET_BF,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E1_RAMIS_DET_BF,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
WARN,E1_RAMIS_DET_BF,predictions_train_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 177, ""n_rows"": 6325}"
WARN,E1_RAMIS_DET_BF,predictions_validation_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 9, ""n_rows"": 4376}"
PASS,E1_RAMIS_DET_BF,temporal_split_order,Train termina antes de validation.,
PASS,E1_RAMIS_DET_BF,evaluation_window_exists,evaluation_window.json presente.,
PASS,E1_RAMIS_DET_BF,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4376, ""start_date"": ""2009-01-08""}"
PASS,E1_RAMIS_DET_BF,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6325, ""start_date"": ""1991-09-08""}"
PASS,E1_RAMIS_DET_BF,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""none"", ""model_type"": ""physical"", ""validation_negatives"": 0}"
PASS,E1_RAMIS_DET_BF,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E2_RAMIS_LEVY_NOBF,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E2_RAMIS_LEVY_NOBF,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
WARN,E2_RAMIS_LEVY_NOBF,predictions_train_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 177, ""n_rows"": 6325}"
WARN,E2_RAMIS_LEVY_NOBF,predictions_validation_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 9, ""n_rows"": 4376}"
PASS,E2_RAMIS_LEVY_NOBF,temporal_split_order,Train termina antes de validation.,
PASS,E2_RAMIS_LEVY_NOBF,evaluation_window_exists,evaluation_window.json presente.,
PASS,E2_RAMIS_LEVY_NOBF,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4376, ""start_date"": ""2009-01-08""}"
PASS,E2_RAMIS_LEVY_NOBF,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6325, ""start_date"": ""1991-09-08""}"
PASS,E2_RAMIS_LEVY_NOBF,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""none"", ""model_type"": ""stochastic_physical"", ""validation_negatives"": 0}"
PASS,E2_RAMIS_LEVY_NOBF,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E3_RAMIS_LEVY_BF,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E3_RAMIS_LEVY_BF,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
WARN,E3_RAMIS_LEVY_BF,predictions_train_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 177, ""n_rows"": 6325}"
WARN,E3_RAMIS_LEVY_BF,predictions_validation_Qobs_finite,Qobs contiene valores faltantes/no numericos; se conservan como brechas observacionales y deben excluirse de metricas.,"{""n_missing"": 9, ""n_rows"": 4376}"
PASS,E3_RAMIS_LEVY_BF,temporal_split_order,Train termina antes de validation.,
PASS,E3_RAMIS_LEVY_BF,evaluation_window_exists,evaluation_window.json presente.,
PASS,E3_RAMIS_LEVY_BF,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4376, ""start_date"": ""2009-01-08""}"
PASS,E3_RAMIS_LEVY_BF,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6325, ""start_date"": ""1991-09-08""}"
PASS,E3_RAMIS_LEVY_BF,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""none"", ""model_type"": ""stochastic_physical"", ""validation_negatives"": 0}"
PASS,E3_RAMIS_LEVY_BF,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E4_ML_PURE_XGB,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E4_ML_PURE_XGB,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
PASS,E4_ML_PURE_XGB,lags_no_future,Lags no negativos; no se usan predictores futuros.,"[0, 1, 2]"
PASS,E4_ML_PURE_XGB,feature_source_allowed,Fuente de features permitida.,observed_forcing
PASS,E4_ML_PURE_XGB,features_no_observed_target,Las variables de entrada no incluyen Qobs/target observado.,"[""P"", ""PET"", ""Tmin"", ""Tmax""]"
PASS,E4_ML_PURE_XGB,temporal_split_order,Train termina antes de validation.,
PASS,E4_ML_PURE_XGB,evaluation_window_exists,evaluation_window.json presente.,
PASS,E4_ML_PURE_XGB,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4367, ""start_date"": ""2009-01-08""}"
PASS,E4_ML_PURE_XGB,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6148, ""start_date"": ""1991-09-08""}"
PASS,E4_ML_PURE_XGB,ml_backend_recorded,Backend ML registrado.,"{""backend"": ""sklearn_hgb"", ""label"": ""sklearn_hgb""}"
PASS,E4_ML_PURE_XGB,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""clip_negative_to_zero"", ""model_type"": ""machine_learning"", ""validation_negatives"": 0}"
PASS,E4_ML_PURE_XGB,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E5_HYB_XGB_MEAN,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E5_HYB_XGB_MEAN,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
PASS,E5_HYB_XGB_MEAN,lags_no_future,Lags no negativos; no se usan predictores futuros.,"[0, 1, 2]"
PASS,E5_HYB_XGB_MEAN,feature_source_allowed,Fuente de features permitida.,stochastic_ensemble
PASS,E5_HYB_XGB_MEAN,features_no_observed_target,Las variables de entrada no incluyen Qobs/target observado.,"[""Qmean""]"
PASS,E5_HYB_XGB_MEAN,temporal_split_order,Train termina antes de validation.,
PASS,E5_HYB_XGB_MEAN,evaluation_window_exists,evaluation_window.json presente.,
PASS,E5_HYB_XGB_MEAN,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4367, ""start_date"": ""2009-01-08""}"
PASS,E5_HYB_XGB_MEAN,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6148, ""start_date"": ""1991-09-08""}"
PASS,E5_HYB_XGB_MEAN,ml_backend_recorded,Backend ML registrado.,"{""backend"": ""sklearn_hgb"", ""label"": ""sklearn_hgb""}"
PASS,E5_HYB_XGB_MEAN,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""clip_negative_to_zero"", ""model_type"": ""hybrid"", ""validation_negatives"": 0}"
PASS,E5_HYB_XGB_MEAN,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E6_HYB_XGB_QUANTILES,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E6_HYB_XGB_QUANTILES,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
PASS,E6_HYB_XGB_QUANTILES,lags_no_future,Lags no negativos; no se usan predictores futuros.,"[0, 1, 2]"
PASS,E6_HYB_XGB_QUANTILES,feature_source_allowed,Fuente de features permitida.,stochastic_ensemble
PASS,E6_HYB_XGB_QUANTILES,features_no_observed_target,Las variables de entrada no incluyen Qobs/target observado.,"[""Qmean"", ""q05"", ""q25"", ""q50"", ""q75"", ""q95"", ""width_q95_q05"", ""width_q75_q25""]"
PASS,E6_HYB_XGB_QUANTILES,temporal_split_order,Train termina antes de validation.,
PASS,E6_HYB_XGB_QUANTILES,evaluation_window_exists,evaluation_window.json presente.,
PASS,E6_HYB_XGB_QUANTILES,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4367, ""start_date"": ""2009-01-08""}"
PASS,E6_HYB_XGB_QUANTILES,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6148, ""start_date"": ""1991-09-08""}"
PASS,E6_HYB_XGB_QUANTILES,ml_backend_recorded,Backend ML registrado.,"{""backend"": ""sklearn_hgb"", ""label"": ""sklearn_hgb""}"
PASS,E6_HYB_XGB_QUANTILES,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""clip_negative_to_zero"", ""model_type"": ""hybrid"", ""validation_negatives"": 0}"
PASS,E6_HYB_XGB_QUANTILES,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E7_HYB_GRU_MEAN,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E7_HYB_GRU_MEAN,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
PASS,E7_HYB_GRU_MEAN,sequence_no_future,La secuencia usa ventanas pasadas hasta t y predice t+horizon.,"{""horizon"": 1, ""sequence_length"": 7}"
PASS,E7_HYB_GRU_MEAN,feature_source_allowed,Fuente de features permitida.,stochastic_ensemble
PASS,E7_HYB_GRU_MEAN,features_no_observed_target,Las variables de entrada no incluyen Qobs/target observado.,"[""Qmean""]"
PASS,E7_HYB_GRU_MEAN,temporal_split_order,Train termina antes de validation.,
PASS,E7_HYB_GRU_MEAN,evaluation_window_exists,evaluation_window.json presente.,
PASS,E7_HYB_GRU_MEAN,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4367, ""start_date"": ""2009-01-08""}"
PASS,E7_HYB_GRU_MEAN,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6148, ""start_date"": ""1991-09-08""}"
PASS,E7_HYB_GRU_MEAN,ml_backend_recorded,Backend ML registrado.,"{""backend"": ""sklearn_mlp"", ""label"": ""fallback_mlp_on_flattened_sequences""}"
WARN,E7_HYB_GRU_MEAN,gru_backend_real,El experimento declarado GRU no debe redactarse como GRU real salvo que backend sea tensorflow/torch.,"{""backend"": ""sklearn_mlp"", ""label"": ""fallback_mlp_on_flattened_sequences""}"
PASS,E7_HYB_GRU_MEAN,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""clip_negative_to_zero"", ""model_type"": ""hybrid_sequence"", ""validation_negatives"": 0}"
PASS,E7_HYB_GRU_MEAN,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,E8_HYB_GRU_QUANTILES,experiment_output_dir_exists,Directorio de outputs presente.,
PASS,E8_HYB_GRU_QUANTILES,forecast_horizon_nonnegative,forecast_horizon es no negativo.,1
PASS,E8_HYB_GRU_QUANTILES,sequence_no_future,La secuencia usa ventanas pasadas hasta t y predice t+horizon.,"{""horizon"": 1, ""sequence_length"": 7}"
PASS,E8_HYB_GRU_QUANTILES,feature_source_allowed,Fuente de features permitida.,stochastic_ensemble
PASS,E8_HYB_GRU_QUANTILES,features_no_observed_target,Las variables de entrada no incluyen Qobs/target observado.,"[""Qmean"", ""q05"", ""q25"", ""q50"", ""q75"", ""q95"", ""width_q95_q05"", ""width_q75_q25"", ""Qbase""]"
PASS,E8_HYB_GRU_QUANTILES,temporal_split_order,Train termina antes de validation.,
PASS,E8_HYB_GRU_QUANTILES,evaluation_window_exists,evaluation_window.json presente.,
PASS,E8_HYB_GRU_QUANTILES,evaluation_window_validation_matches_predictions,Ventana validation coincide con predicciones.,"{""end_date"": ""2020-12-31"", ""n_after"": 4367, ""start_date"": ""2009-01-08""}"
PASS,E8_HYB_GRU_QUANTILES,evaluation_window_train_matches_predictions,Ventana train coincide con predicciones.,"{""end_date"": ""2008-12-31"", ""n_after"": 6148, ""start_date"": ""1991-09-08""}"
PASS,E8_HYB_GRU_QUANTILES,ml_backend_recorded,Backend ML registrado.,"{""backend"": ""sklearn_mlp"", ""label"": ""fallback_mlp_on_flattened_sequences""}"
WARN,E8_HYB_GRU_QUANTILES,gru_backend_real,El experimento declarado GRU no debe redactarse como GRU real salvo que backend sea tensorflow/torch.,"{""backend"": ""sklearn_mlp"", ""label"": ""fallback_mlp_on_flattened_sequences""}"
PASS,E8_HYB_GRU_QUANTILES,postprocessing_expected_scope,Postproceso aplicado solo donde corresponde.,"{""method"": ""clip_negative_to_zero"", ""model_type"": ""hybrid_sequence"", ""validation_negatives"": 10}"
PASS,E8_HYB_GRU_QUANTILES,regime_threshold_source,Umbrales de regimen calculados solo con Qobs de calibracion.,"{""n_train_finite_qobs"": 6155, ""note"": ""Los regimenes de validation usan umbrales estimados solo con Qobs de calibracion."", ""p25"": 0.074, ""p75"": 0.576, ""source"": ""train_Qobs_only""}"
PASS,GLOBAL,common_window_consistency,Todos los experimentos comparten inicio/fin de validation.,"{""end"": ""2020-12-31"", ""n_experiments"": 9, ""start"": ""2009-01-08""}"
PASS,GLOBAL,common_intersection_experiment_coverage,La interseccion comun cubre todos los experimentos esperados.,
PASS,GLOBAL,common_intersection_n_eval_equal,n_eval es identico para todos los experimentos en la interseccion exacta.,4367
```

## pytest -q

```text
..........................                                               [100%]
=============================== warnings summary ===============================
tests/test_smoke.py::test_imports
  /usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.4
    warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"

tests/test_smoke.py::test_e1_runs_end_to_end
  /home/jose/.local/lib/python3.10/site-packages/matplotlib/projections/__init__.py:63: UserWarning: Unable to import Axes3D. This may be due to multiple versions of Matplotlib being installed (e.g. as a system package and as a pip package). As a result, the 3D projection is not available.
    warnings.warn("Unable to import Axes3D. This may be due to multiple versions of "

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
26 passed, 2 warnings in 10.21s
```
## Checklist para el siguiente chat

Solicitar Fase 3.4B con estos entregables:

- Nuevo módulo o script para generación de figuras paper-ready.
- Funciones reutilizables para cargar predicciones por experimento.
- Uso consistente de la ventana común.
- Figuras guardadas en PNG y, preferentemente, SVG/PDF.
- Manifest de figuras.
- Tests smoke que validen que el script corre con datos mínimos.
- README actualizado con rutas de figuras y cautelas metodológicas.

## Archivos importantes esperados

- configs/experiments.yaml
- scripts/run_all_experiments.py
- scripts/summarize_results.py
- scripts/audit_leakage_phase34.py
- src/stohymolap/experiments/runner.py
- src/stohymolap/experiments/comparison.py
- src/stohymolap/diagnostics/leakage_audit.py
- outputs/comparison/leaderboard_common_intersection.csv
- outputs/comparison/regime_metrics_comparison.csv
- outputs/comparison/ml_backend_summary.csv
- outputs/experiments/*/predictions_validation.csv
- outputs/experiments/*/evaluation_window.json
- outputs/experiments/*/ml_backend.json
- outputs/experiments/*/postprocessing_report.json
- outputs/experiments/*/regime_thresholds.json
