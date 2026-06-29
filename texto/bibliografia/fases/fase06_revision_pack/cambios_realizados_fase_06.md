# Cambios realizados - Fase 6: Limitations and Conclusions

## Secciones intervenidas

- `\section{Limitations}`
- `\section{Conclusions}`
- Ajuste menor en `Code and data availability` para retirar comentarios `FILL` visibles en el código fuente.
- Ajuste menor en comentarios de `Acknowledgements` para dejar una instrucción editorial no visible en el PDF.

## Cambios editoriales principales

1. Se reescribió completamente la sección de limitaciones para que funcione como una delimitación científica del alcance del estudio, no como una lista informal de advertencias.
2. Se organizaron cinco limitaciones principales:
   - ejecución fallback de los experimentos etiquetados como GRU;
   - ausencia de calibración probabilística de las bandas estocásticas;
   - persistencia del problema de bajos caudales;
   - alcance restringido a un modelo lumped y una cuenca alto-andina;
   - estado de auditoría WARN por observaciones faltantes, fallbacks y advertencias físicas.
3. Se reescribieron las conclusiones para evitar repetición de la discusión.
4. La conclusión principal quedó centrada en que el mayor aporte robusto proviene de los descriptores cuantílicos del ensemble físico estocástico.
5. Se mantuvo la distinción entre:
   - mejor resultado numérico: `E8_HYB_GRU_QUANTILES`;
   - mejor evidencia defendible: `E6_HYB_XGB_QUANTILES`.
6. Se reforzó que E8 no prueba superioridad de redes recurrentes porque fue ejecutado como fallback MLP.
7. Se incorporó una conclusión explícita sobre incertidumbre: las bandas estocásticas son subdispersivas y no deben presentarse como intervalos predictivos calibrados.
8. Se eliminaron todas las marcas `FILL` del archivo LaTeX.

## Resultado de compilación

- `pdflatex` ejecutado correctamente.
- `bibtex8` ejecutado correctamente.
- No se detectaron citas indefinidas ni referencias cruzadas indefinidas tras la compilación final.
- Persisten advertencias menores de `Overfull hbox`, principalmente asociadas a tablas y cadenas técnicas largas. Estas deben atenderse en la fase final de limpieza LaTeX.

## Archivos generados

- `manuscript_fase06.tex`
- `new_limitations_conclusions_fase06.tex`
- `manuscript_fase06.pdf`
- `cambios_realizados_fase_06.md`
- `decisiones_editoriales_fase06.md`
- `tareas_pendientes_fase06.md`
- `fase06_revision_pack.zip`
