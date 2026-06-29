# Cambios realizados — Fase 3: Materials and methods

## Alcance
Se revisó y reescribió la sección `Materials and methods` de `manuscript_fase02.tex`, generando `manuscript_fase03.tex` y el archivo independiente `new_methods_fase03.tex`.

## Cambios editoriales principales

1. **Reestructuración metodológica completa**
   - Se reorganizó la sección para seguir una progresión lógica:
     1. Study area
     2. Hydro-meteorological data and temporal protocol
     3. Lumped rainfall--runoff core
     4. Linear baseflow reservoir
     5. Lévy perturbations and Monte Carlo physical ensembles
     6. Machine-learning emulators and feature sets
     7. Calibration objective
     8. Experimental matrix and ablation design
     9. Relation to prior stochastic physical--ML hybridization
     10. Evaluation metrics and audit protocol

2. **Clarificación del tipo de modelo físico**
   - Se corrigió el lenguaje para evitar presentar StoHyMoLAP como un modelo hidrodinámico espacial.
   - El manuscrito ahora declara explícitamente que el núcleo físico es un simulador lumped rainfall--runoff estocástico, no un solver hidráulico/hidrodinámico 2D.

3. **Mejora del área de estudio**
   - Se reformuló la descripción del Ramis dentro del sistema Titicaca.
   - Se integraron referencias regionales con claves reales del `ref.bib`.
   - Se eliminó el placeholder visible `[FILL: ...]` del cuerpo del texto.
   - Se dejó una nota LaTeX oculta `%% [AUTHOR ACTION BEFORE SUBMISSION]` para insertar área de drenaje, coordenadas, elevación y periodo oficial de registro cuando se disponga de metadatos SENAMHI/ANA.

4. **Fortalecimiento del protocolo temporal y anti-leakage**
   - Se hizo explícito que el leaderboard usa una intersección común de validación de 4367 muestras diarias.
   - Se detallaron cuatro controles de leakage:
     - construcción de lags solo con información presente/pasada;
     - generación de secuencias dentro de cada partición temporal;
     - ajuste de escaladores solo en entrenamiento;
     - estimación de umbrales de régimen solo con observaciones de entrenamiento.

5. **Revisión del núcleo RAMIS**
   - Se mantuvieron las ecuaciones principales del estado rápido y descarga rápida.
   - Se explicó mejor el significado de `mu`, `lambda`, `Q0`, `alpha_A`, `sigma` y `Delta L_t`.
   - Se incorporó una interpretación más defensible de la no linealidad agregada y del parámetro `mu`.

6. **Reformulación del reservorio de baseflow**
   - Se eliminó lenguaje demasiado enfático y citas literales largas.
   - Se describió el reservorio lineal como componente opcional de memoria hidrológica.
   - Se mantuvieron las ecuaciones de recarga, almacenamiento, baseflow y caudal total.

7. **Revisión de perturbaciones Lévy y ensemble Monte Carlo**
   - Se reformuló la justificación de ruido Lévy/alpha-estable con un tono más prudente.
   - Se enfatizó que el ensemble tiene dos roles: predictor físico directo y fuente de features para ML.
   - Se mejoró la explicación de cuantiles y anchuras intercuantílicas como descriptores robustos.

8. **Clarificación de los emuladores ML**
   - Se definieron tres familias de features: ML puro, híbridos basados en media del ensemble e híbridos basados en cuantiles del ensemble.
   - Se corrigió el reclamo sobre XGB: los resultados deben interpretarse como gradient boosting con backend `sklearn_hgb`, no como evidencia específica de `xgboost`.
   - Se corrigió el reclamo sobre GRU: E7/E8 son experimentos etiquetados como GRU pero ejecutados como fallback MLP sobre secuencias aplanadas.

9. **Objetivo de calibración**
   - Se mantuvo la función objetivo compuesta `J`.
   - Se aclaró que el término de cobertura se omite y renormaliza cuando el modelo no produce intervalos.
   - Se justificó la función multi-métrica de forma más sobria y alineada con la literatura.

10. **Matriz experimental y diseño de ablación**
    - Se reformuló la tabla `tab:experiment-matrix` para reflejar el backend real usado.
    - Se explicó explícitamente qué contraste de ablación responde a qué componente metodológico.

11. **Relación con Houenafa et al. (2025)**
    - Se reposicionó el trabajo como extensión de ablación y auditoría, no como duplicación.
    - Se enfatizó que el estudio descompone memoria física, ruido estocástico, corrección ML y elección de descriptor.

12. **Métricas y auditoría**
    - Se reformuló la descripción de NSE, KGE, RMSE, MAE, PBIAS, R², PICP, PINAW, MPIW y Winkler.
    - Se reforzó la interpretación cautelosa del NSE en bajos caudales.
    - Se conectó el protocolo de auditoría con la reproducibilidad del manuscrito.

## Control de citas
- Todas las citas dentro de la sección `Materials and methods` fueron normalizadas a claves existentes en `ref.bib`.
- Resultado: **0 claves BibTeX indefinidas dentro de Métodos**.
- Persisten claves indefinidas en Results/Discussion/Limitations, que deben corregirse en fases posteriores.

## Control LaTeX
- Se compiló `manuscript_fase03.tex` con `pdflatex`.
- Resultado: **compilación sin error fatal**.
- Persisten advertencias de referencias/citas indefinidas fuera de Métodos, esperables porque esas secciones aún no han sido corregidas.
