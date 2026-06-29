# Tareas pendientes después de la Fase 2

## Prioridad alta antes de envío

1. **Completar y verificar datos del área de estudio**
   - En `Materials and methods` siguen existiendo placeholders sobre área de drenaje, coordenadas de estación, periodo de registro y procedencia/unidades de la serie observada Ramis.
   - Estos datos deben completarse con fuente trazable SENAMHI/ANA o la fuente oficial que corresponda.

2. **Normalizar citas en el resto del manuscrito**
   - La introducción ya usa claves válidas de `ref.bib`.
   - Persisten claves indefinidas en métodos, resultados, discusión, limitaciones y conclusiones.
   - Ejemplos de alias que aún deben corregirse:
     - `houenafa2025` → `houenafa2025hybridization`
     - `hall2021` → `hall2021hydrologist`
     - `nauditt2017` → `nauditt2017conceptual`
     - `van2013` → `van2013influence`
     - `gabellani2007` → `gabellani2007propagation`
     - `hong2006` → `hong2006uncertainty`
     - `papacharalampous2022` → `papacharalampous2022review`
   - Otras claves parecen no estar disponibles en `ref.bib` y requerirán añadir entradas o reemplazarlas.

3. **Revisar Methods en profundidad**
   - La próxima fase debe revisar:
     - coherencia del área de estudio;
     - definición de datos y protocolo temporal;
     - formulación del modelo RAMIS;
     - reservorio baseflow;
     - ruido L\'{e}vy / $\alpha$-estable;
     - generación Monte Carlo;
     - emuladores ML;
     - matriz experimental;
     - métricas y auditorías.

4. **Verificar consistencia entre Methods y figuras**
   - Revisar que Fig. `method_workflow.png` y Table `experiment-matrix` coincidan exactamente con la narrativa metodológica.
   - Revisar si las figuras duplican identificadores o generan advertencias de hyperref.

5. **Resolver advertencias de compilación**
   - La compilación con `pdflatex` fue exitosa, pero quedan advertencias de referencias/citas indefinidas en el resto del manuscrito.
   - `bibtex` no estuvo disponible en el entorno, por lo que debe probarse localmente con una cadena completa `pdflatex → bibtex → pdflatex → pdflatex`.

6. **Completar secciones finales**
   - `Code and data availability` requiere URL/DOI.
   - `Acknowledgements` requiere créditos institucionales/fuente de datos.
   - Confirmar declaración de IA asistida según política de la revista.

## Próxima fase recomendada

7. **Fase 3 — Materials and methods**
   - Revisar la sección metodológica completa para asegurar trazabilidad, reproducibilidad, claridad matemática y coherencia entre texto, tabla experimental y workflow.
   - Se recomienda trabajar esta fase en el mismo chat si el contexto todavía se mantiene; no es obligatorio cambiar de chat todavía.

## Para continuar en otro chat

Adjuntar:

- `fase02_revision_pack.zip`
- o, alternativamente:
  - `manuscript_fase02.tex`
  - `ref.bib`
  - carpeta `figures/`
  - `notas.txt`
  - `cambios_realizados_fase_01.md`
  - `cambios_realizados_fase_02.md`
  - `decisiones_editoriales_fase02.md`
  - `tareas_pendientes_fase02.md`
