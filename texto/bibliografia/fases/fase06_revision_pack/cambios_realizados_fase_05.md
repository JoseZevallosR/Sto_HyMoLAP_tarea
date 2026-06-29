# Cambios realizados — Fase 5: Discussion

## Alcance
Se revisó y reescribió la sección `Discussion` de `manuscript_fase04.tex`, generando `manuscript_fase05.tex` y `new_discussion_fase05.tex`.

## Cambios principales

1. **Reestructuración completa de la discusión**
   - Se reorganizó la sección en seis bloques argumentales:
     1. `Main inference from the ablation`.
     2. `Hydrological memory, baseflow and low-flow behaviour`.
     3. `Lévy perturbations: useful ensemble diversity, not calibrated uncertainty`.
     4. `Why ensemble quantiles improve the hybrid model`.
     5. `Backend-aware interpretation of the sequential experiments`.
     6. `Reproducibility, leakage control and submission readiness`.

2. **Clarificación del hallazgo central**
   - La discusión ahora afirma de forma explícita que el principal motor de mejora es la transferencia de información distribucional desde el ensemble físico estocástico hacia el emulador ML.
   - Se evita interpretar la mejora como producto directo del ruido Lévy o de una mayor complejidad nominal del backend.

3. **Corrección de la interpretación del baseflow**
   - Se matizó la lectura del reservorio lineal de baseflow.
   - El texto ahora reconoce que E1 mejora el NSE global frente a E0, pero que el RMSE de bajo caudal no mejora en `Table~\ref{tab:regime-rmse}`.
   - Se evita sobreafirmar que el reservorio corrige por sí solo el régimen seco.
   - Se propone evaluar futuros modelos con métricas enfocadas en bajo caudal, como inverse-flow NSE o funciones de pérdida ponderadas por régimen.

4. **Reinterpretación del componente Lévy**
   - Se dejó claro que las perturbaciones Lévy no mejoran de manera robusta el punto de predicción cuando se aplican directamente al modelo físico.
   - Se reforzó que las bandas estocásticas son subdispersivas y no deben presentarse como incertidumbre calibrada.
   - Se redefinió su valor científico como generador de diversidad de ensemble y de descriptores cuantílicos útiles para ML.

5. **Discusión más conservadora de los híbridos cuantílicos**
   - Se mantuvo el claim fuerte sobre E6 como mejor configuración backend-consistente.
   - Se explicó por qué los cuantiles contienen información que se pierde al usar solo la media del ensemble.
   - Se agregó una advertencia metodológica: E6--E4 combina diferencias de estructura física y de ingeniería de features; por tanto, no separa completamente ambos efectos.

6. **Corrección de la interpretación de E8/E7**
   - E8 se mantiene como el experimento numéricamente mejor, pero no como evidencia de superioridad GRU.
   - E7 y E8 se describen como experimentos GRU-labelled bajo fallback MLP sobre secuencias aplanadas.
   - La discusión propone repetir E7/E8 con un backend recurrente real en TensorFlow o PyTorch.

7. **Normalización bibliográfica**
   - Se corrigieron las claves BibTeX abreviadas dentro de la discusión.
   - También se aplicó una corrección mínima de higiene bibliográfica en `Limitations` y `Conclusions` para reemplazar claves abreviadas por claves reales de `ref.bib`.
   - Tras la fase 5, no quedan claves BibTeX inexistentes en el manuscrito.

8. **Verificación LaTeX**
   - Se compiló el manuscrito con `pdflatex`.
   - `bibtex` no está instalado en el entorno, pero `bibtex8` sí funcionó correctamente.
   - Tras ejecutar `bibtex8` y recompilar, no quedan citas indefinidas ni referencias cruzadas indefinidas.
   - Persisten advertencias menores de LaTeX: `No positions in optional float specifier`, asociadas a floats y pendientes para la fase final de limpieza LaTeX.

## Archivos generados

- `manuscript_fase05.tex`
- `new_discussion_fase05.tex`
- `manuscript_fase05.pdf`
- `cambios_realizados_fase_05.md`
- `decisiones_editoriales_fase05.md`
- `tareas_pendientes_fase05.md`
- `fase05_revision_pack.zip`
