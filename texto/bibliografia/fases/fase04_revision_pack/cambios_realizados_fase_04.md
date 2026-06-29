# Cambios realizados — Fase 4 (Results)

## Alcance
Se revisó exclusivamente la sección `Results` de `manuscript_fase03.tex`, tomando como base el manuscrito ya corregido en las fases 1–3. No se modificaron `Discussion`, `Limitations`, `Conclusions` ni secciones finales.

## Cambios principales

1. **Reescritura completa de la sección Results**
   - Se reemplazó la sección original por una versión más directa, sobria y orientada a evidencia.
   - Se redujo el tono interpretativo excesivo para reservar parte de la discusión mecanística para la fase 5.
   - Se reforzó la distinción entre resultados numéricos, interpretación robusta y advertencias de auditoría.

2. **Common-window ranking**
   - Se explicó con mayor claridad la separación entre modelos físicos, ML puro e híbridos.
   - Se enfatizó que E6_HYB_XGB_QUANTILES es la configuración robusta más defendible porque el backend ejecutado coincide con el reclamo metodológico.
   - Se mantuvo E8_HYB_GRU_QUANTILES como el mejor resultado numérico, pero con una advertencia explícita: fue ejecutado como fallback MLP, no como GRU real.

3. **Hydrograph and flow-duration behaviour**
   - Se mejoró la conexión entre el hidrograma, las curvas de duración de caudal y las métricas.
   - Se describió de manera más precisa el colapso de E0 en la cola de alto excedencia.
   - Se evitó afirmar una corrección completa de los picos; el texto ahora reconoce que los errores de eventos extremos permanecen parcialmente corregidos.

4. **Ablation effects**
   - Se clarificó la contribución incremental de baseflow, perturbaciones Lévy, ML puro, híbridos con media y cuantiles.
   - Se destacó que el mayor efecto robusto proviene de E6 frente a E3: ΔNSE = 0.311 y ΔRMSE = -0.133 m³ s⁻¹.
   - Se hizo explícito que los cuantiles aportan información predictiva que se pierde al comprimir el ensemble a su media.
   - Se corrigió la estructura LaTeX de la tabla `tab:ablation`, que tenía una columna declarada de más.

5. **Performance by flow regime**
   - Se reescribió la sección para enfocarla en RMSE por régimen, evitando sobreinterpretar NSE de bajos caudales en Results.
   - Se mantuvo el hallazgo clave: E6 y E8 reducen sustancialmente los errores en bajos caudales, aunque los altos caudales siguen siendo los más difíciles.

6. **Uncertainty diagnostics and backend audit**
   - Se fortaleció la interpretación de PICP/PINAW/Winkler.
   - Se declaró explícitamente que los intervalos estocásticos son subdispersivos y no deben presentarse como incertidumbre predictiva calibrada.
   - Se reforzó que el principal valor del ensemble físico estocástico es como fuente de features para ML.
   - Se mejoró la tabla de auditoría de backend usando columnas tipo `p{}` para evitar problemas de ancho con strings largos.

7. **Citas bibliográficas en Results**
   - Se eliminaron citas bibliográficas de la sección Results para mantenerla enfocada en evidencia propia y evitar claves BibTeX indefinidas dentro de la fase 4.
   - La discusión comparativa con la literatura queda reservada para la fase 5.

## Archivos generados

- `manuscript_fase04.tex`: manuscrito completo actualizado hasta la fase 4.
- `new_results_fase04.tex`: sección Results revisada, lista para reemplazar en el manuscrito.
- `cambios_realizados_fase_04.md`: este archivo.
- `decisiones_editoriales_fase04.md`: decisiones editoriales tomadas durante la fase.
- `tareas_pendientes_fase04.md`: tareas pendientes para las siguientes fases.
- `fase04_revision_pack.zip`: paquete de continuidad con manuscrito, Results revisado, referencias, notas, figuras y archivos de trazabilidad.

## Verificación técnica

- `pdflatex` compiló sin error fatal.
- La herramienta `bibtex` no está disponible en el entorno actual, por lo que la resolución bibliográfica completa no pudo verificarse automáticamente.
- La sección Results revisada no contiene citas bibliográficas, por lo que no introduce nuevas claves indefinidas.
- Persisten claves BibTeX indefinidas en secciones posteriores aún no revisadas, principalmente Discussion y Limitations; deberán normalizarse en la fase 5.
