# Decisiones editoriales — Fase 4 (Results)

## 1. Separar evidencia numérica de interpretación fuerte
Se decidió que Results debe presentar hallazgos con mínima discusión bibliográfica. Por ello, las comparaciones extensas con literatura previa fueron retiradas de esta sección y se reservarán para Discussion.

## 2. Doble lectura del ranking
Se mantuvo una lectura en dos niveles:

- **Mejor resultado numérico:** E8_HYB_GRU_QUANTILES.
- **Mejor resultado robusto y defendible:** E6_HYB_XGB_QUANTILES.

Esta distinción es indispensable porque E8 fue etiquetado como GRU, pero el backend ejecutado fue un fallback MLP sobre secuencias aplanadas.

## 3. Evitar reclamar superioridad recurrente
Se decidió no interpretar E7/E8 como evidencia de superioridad de GRU o redes recurrentes. La formulación correcta es: experimentos secuenciales etiquetados como GRU bajo ejecución fallback MLP.

## 4. Presentar los cuantiles como hallazgo principal
El resultado central de la fase 4 es que los descriptores cuantílicos del ensemble físico estocástico superan a la media del ensemble. Esto respalda la hipótesis de que la distribución física simulada contiene información útil que no se preserva en una trayectoria media.

## 5. No vender los intervalos estocásticos como incertidumbre calibrada
Los valores de PICP son demasiado bajos para sostener un reclamo de incertidumbre predictiva calibrada. La decisión editorial es presentar las bandas como referencias estocásticas subdispersivas y como fuente de atributos físicos para ML, no como intervalos confiables de pronóstico.

## 6. Mantener sobriedad sobre bajos caudales
Se decidió describir mejoras de RMSE en bajos caudales, pero sin sobredimensionarlas. El texto reconoce que altos caudales y picos de eventos siguen siendo difíciles y que la mejora de los híbridos no equivale a una solución completa del problema hidrológico.

## 7. Corrección LaTeX de tablas
La tabla de ablation effects fue corregida de `llllrrrr` a `lllrrrr` porque el número de columnas declaradas no coincidía con el contenido real. La tabla de backend fue reformateada con columnas `p{}` para mejorar legibilidad y evitar desbordes.
