# Cambios realizados — Fase 1

## Alcance
Fase 1 centrada en portada editorial, título, autoría, abstract, highlights, keywords y consistencia básica de metadatos de autor.

## Cambios aplicados en `manuscript_fase01.tex`

1. **Título**
   - Se reemplazó el título interrogativo original por un título declarativo, más directo y alineado con el hallazgo principal:
     `Stochastic Physical-Ensemble Quantiles Improve Hybrid Machine-Learning Simulation of Daily Streamflow in the High-Andean Ramis Basin`.
   - Se actualizó `\shorttitle` para reflejar el eje del manuscrito: cuantiles de ensemble físico estocástico e hibridación.

2. **Autoría y afiliación**
   - Se reemplazaron los metadatos genéricos del autor por el bloque proporcionado para Jose Zevallos.
   - Se incorporó ORCID, correo y contribución CRediT en el preámbulo.
   - Se eliminó el autor placeholder `Second Author` del front matter.
   - Se actualizó `\shortauthors{Zevallos}`.
   - Se mantuvo la afiliación CITA-UTEC con ciudad, código postal, estado y país.

3. **Corrección de sintaxis de afiliación**
   - El bloque proporcionado tenía una inconsistencia de índice (`\author[1]` frente a `\affiliation[2]`).
   - Para asegurar compilación correcta, se usó `\affiliation[1]`, manteniendo intacto el contenido institucional de la afiliación.

4. **Nota de título**
   - Se eliminó la nota de título que indicaba que el manuscrito era un borrador generado desde un paquete reproducible. Esa información no debe aparecer como nota editorial de título en una versión destinada a envío.

5. **Abstract**
   - Se reescribió el abstract para hacerlo más declarativo, compacto, autocontenido y orientado al hallazgo.
   - Se eliminó la dependencia de citas dentro del abstract.
   - Se preservaron los resultados clave: NSE/KGE del híbrido cuantílico, mejora de 0.311 NSE sobre el modelo físico estocástico, mejora de 0.165 sobre ML puro, ganancia de 0.114 NSE por cuantiles frente a media, subdispersión de bandas y caveat del backend GRU-labelled/fallback MLP.
   - Se sustituyó la formulación `XGB-labelled` por `gradient-boosting` para no sobreafirmar el uso de XGBoost cuando el backend real es `sklearn_hgb`.

6. **Highlights**
   - Se reescribieron los cinco highlights para que sean más breves, editoriales y consistentes con las recomendaciones Elsevier.
   - Se reforzó el mensaje principal: los cuantiles del ensemble físico son el principal driver del skill híbrido.

7. **Keywords**
   - Se actualizaron las palabras clave para reflejar con mayor precisión el alcance del manuscrito:
     `Daily streamflow simulation`, `Stochastic hydrological modelling`, `Physical--machine learning hybridization`, `Ensemble quantiles`, `Uncertainty diagnostics`, `High-Andean basin`.

8. **CRediT authorship contribution statement**
   - Se eliminó el texto placeholder de `First Author` y `Second Author`.
   - Se dejó una declaración CRediT consistente con el autor informado: `Jose Zevallos: Conceptualization of this study, Methodology, Software, Writing -- original draft.`

## Validación técnica realizada

- Se ejecutó una compilación preliminar con `pdflatex` usando `cas-sc.cls`.
- La compilación LaTeX básica fue exitosa y generó PDF sin error fatal.
- Persisten advertencias de referencias/citas indefinidas y advertencias menores de figuras, que se documentan como tareas pendientes porque exceden el alcance estricto de la Fase 1.
