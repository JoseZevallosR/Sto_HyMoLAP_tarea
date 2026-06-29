# Tareas pendientes después de Fase 3

## Antes del envío
1. Insertar metadatos oficiales de Puente Ramis:
   - área de drenaje;
   - coordenadas del gauge;
   - elevación;
   - institución/fuente oficial;
   - periodo completo de registro;
   - detalles de control de calidad de caudales.

2. Confirmar el origen de las variables meteorológicas:
   - precipitación;
   - temperatura mínima y máxima;
   - PET;
   - resolución espacial/temporal;
   - método de extracción al punto o cuenca;
   - unidades.

3. Confirmar si el salto entre 1 January 2009 y 7 January 2009 corresponde a burn-in, lags, secuencias o datos faltantes.

4. Añadir, si el autor lo considera necesario, referencias clásicas faltantes para:
   - Nash--Sutcliffe efficiency;
   - Kling--Gupta efficiency;
   - Chambers--Mallows--Stuck;
   - XGBoost;
   - GRU.

   En esta fase no se añadieron nuevas entradas al `ref.bib` para evitar inventar bibliografía fuera del paquete proporcionado.

## Para Fase 4 — Results
1. Corregir claves BibTeX abreviadas que todavía aparecen en Results.
2. Revisar coherencia entre resultados numéricos, tablas y figuras.
3. Verificar si los textos sobre mejoras porcentuales y diferencias NSE/KGE coinciden exactamente con las tablas.
4. Asegurar que E8 se describa siempre con la advertencia de fallback MLP.
5. Revisar la interpretación de bajos caudales y evitar sobreafirmaciones basadas en NSE negativo.
6. Comprobar consistencia entre:
   - `tab:ranking`;
   - `tab:ablation`;
   - `tab:regime-rmse`;
   - `tab:uncertainty`;
   - figuras de ranking, FDC, hydrograph, ablation, regime RMSE y uncertainty.

## Para fases posteriores
1. Reescribir Discussion para que no repita Results.
2. Separar claramente hallazgos, implicancias, limitaciones y recomendaciones.
3. Corregir claves BibTeX indefinidas remanentes en Discussion y Limitations.
4. Revisar si el título debe conservar “Machine-Learning” o pasar a “Machine Learning” sin guion en estilo Elsevier.
5. Verificar formato final de autoría, CRediT, conflictos de interés, data/code availability y acknowledgements.
