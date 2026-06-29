# Decisiones editoriales - Fase 7

## 1. Captions como evidencia, no como interpretación excesiva
Se decidió que los captions describan qué se muestra, qué unidades se usan y qué precaución interpretativa aplica. Las interpretaciones más amplias se mantienen en Results/Discussion, no en captions.

## 2. E8 y GRU-labelled runs
Se mantuvo E8 como mejor configuración numérica, pero toda mención en captions/tablas lo separa de una afirmación de superioridad GRU. La redacción final sostiene que E8 fue ejecutado como fallback MLP sobre secuencias aplanadas.

## 3. Unidades explícitas en tablas
Las tablas ahora reportan explícitamente:

- RMSE, MAE, MPIW y Winkler score en `m^3 s^-1`.
- PBIAS en `%`.
- NSE, KGE, R2, PICP y PINAW como métricas adimensionales.

Esta decisión reduce ambigüedad y anticipa observaciones de reviewer sobre trazabilidad de unidades.

## 4. Identificadores largos
Se incorporó `\modelid{}` para permitir cortes tipográficos en nombres de experimentos. En tablas se prefirió conservar identificadores completos en lugar de renombrarlos a E0--E8 solamente, porque el manuscrito depende de la auditabilidad de cada experimento.

## 5. Título
Se ajustó levemente el título para reducir problemas tipográficos en `cas-sc`, manteniendo su carácter declarativo y el hallazgo principal:

`Stochastic Physical-Ensemble Quantiles Improve Hybrid Daily Streamflow Simulation in the High-Andean Ramis Basin`

## 6. Advertencias de compilación
No se intentó modificar internamente la clase Elsevier `cas-sc` para eliminar advertencias de `\maketitle`/hyperref, porque el PDF renderizado no muestra fallas visuales y cambiar la clase podría introducir problemas de compatibilidad editorial.
