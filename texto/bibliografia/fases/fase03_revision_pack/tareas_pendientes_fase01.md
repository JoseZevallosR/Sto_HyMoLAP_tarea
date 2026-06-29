# Tareas pendientes después de la Fase 1

## Prioridad alta antes de envío

1. **Confirmar autoría final**
   - Confirmar si Jose Zevallos será autor único o si habrá coautores.
   - Si se agregan coautores, actualizar `\shortauthors`, afiliaciones, correos, ORCID, CRediT y declaración de intereses.

2. **Revisar consistencia completa de referencias**
   - El manuscrito contiene numerosas claves `\citep{}` / `\citet{}` que no coinciden exactamente con las claves disponibles en `ref.bib`.
   - Algunas son mapeables directamente, por ejemplo:
     - `mekonnen2015` → `mekonnen2015hybrid`
     - `okkan2021` → `okkan2021embedding`
     - `papacharalampous2022` → `papacharalampous2022review`
     - `swiderski2016` → `swiderski2016aggregation`
     - `hall2021` → `hall2021hydrologist`
     - `houenafa2025` → `houenafa2025hybridization`
   - Otras requieren añadir entradas faltantes a `ref.bib` o reemplazar la cita por una referencia disponible.

3. **Completar placeholders editoriales**
   - Completar `Code and data availability`: URL/DOI del repositorio, disponibilidad de datos, restricciones SENAMHI/ANA si aplica.
   - Completar `Acknowledgements`: SENAMHI, ANA, financiamiento, infraestructura computacional o asesoría si corresponde.
   - Completar datos de procedencia/unidades de la serie observada Ramis si el paquete actual aún es demostrativo o reducido.

4. **Resolver advertencias de compilación**
   - La compilación preliminar con `pdflatex` fue exitosa, pero persisten advertencias de referencias/citas indefinidas.
   - También aparecieron advertencias menores por identificadores duplicados de figuras, que deben revisarse durante la fase de consistencia LaTeX/figuras.

## Próxima fase recomendada

5. **Fase 2 — Introduction**
   - Reescribir la introducción con lógica tipo Journal of Hydrology:
     1. problema hidrológico general;
     2. limitaciones de modelos físicos, ML puro e híbridos deterministas;
     3. brecha específica: descriptores cuantílicos de ensembles físicos estocásticos;
     4. necesidad de ablation auditada;
     5. contribuciones explícitas del manuscrito.
   - Normalizar citas de la introducción con claves reales de `ref.bib`.
   - Reducir afirmaciones demasiado fuertes y reemplazarlas por formulaciones defendibles.

## Para continuar en otro chat

6. **Archivos que deben adjuntarse al nuevo chat**
   - `manuscript_fase01.tex`
   - `ref.bib`
   - carpeta `figures/`
   - `notas.txt`
   - `cambios_realizados_fase_01.md`
   - `decisiones_editoriales.md`
   - `tareas_pendientes.md`
