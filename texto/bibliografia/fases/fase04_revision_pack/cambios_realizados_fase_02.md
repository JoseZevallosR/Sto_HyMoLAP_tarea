# Cambios realizados — Fase 2: Introduction

## Alcance de la fase

Se revisó exclusivamente la sección `\section{Introduction}` del manuscrito `manuscript_fase01.tex`, manteniendo intactas las demás secciones salvo por la sustitución del bloque introductorio completo. El resultado principal es `manuscript_fase02.tex`.

## Cambios editoriales principales

1. **Reestructuración lógica de la introducción**
   - La introducción fue reorganizada siguiendo una progresión más adecuada para *Journal of Hydrology*:
     1. problema general de simulación diaria de caudales en cuencas escasamente instrumentadas;
     2. especificidad del contexto altoandino/Altiplano;
     3. limitaciones complementarias de modelos conceptuales y ML puro;
     4. motivación de modelos híbridos físico--ML;
     5. brecha específica: híbridos deterministas vs descriptores distribucionales de ensembles estocásticos;
     6. necesidad de ablation auditada y reproducibilidad;
     7. contribuciones explícitas del manuscrito.

2. **Reducción de tono promocional**
   - Se eliminaron formulaciones demasiado enfáticas o de tipo “best-model narrative”.
   - El enfoque fue reformulado como una descomposición controlada y auditada de un mecanismo híbrido, no como una afirmación de superioridad universal.

3. **Normalización de citas en la introducción**
   - Todas las citas usadas en la nueva introducción fueron verificadas contra las claves disponibles en `ref.bib`.
   - Se reemplazaron alias no existentes como `razavi2013`, `nauditt2017`, `arsenault2022`, `frame2021`, `zhong2023`, `mekonnen2015`, `okkan2021`, `mohammadi2022`, `gabellani2007`, `hong2006`, `papacharalampous2022`, `houenafa2025`, `hall2021` y `kratzert2018` por sus claves reales.
   - La introducción ya no contiene claves BibTeX indefinidas.

4. **Eliminación de citas textuales largas**
   - Se reemplazaron citas literales por paráfrasis científicas con soporte bibliográfico.
   - Esto mejora fluidez, reduce dependencia de citas extensas y evita que la introducción parezca una revisión anotada.

5. **Fortalecimiento de la brecha científica**
   - La brecha quedó formulada en tres puntos:
     - no basta con mostrar un mejor modelo híbrido; hay que identificar qué componente produce la ganancia;
     - un ensemble estocástico útil como feature no implica automáticamente incertidumbre calibrada;
     - la reproducibilidad requiere que las etiquetas arquitectónicas coincidan con el backend realmente ejecutado.

6. **Clarificación de contribuciones**
   - Se formularon cuatro contribuciones explícitas:
     - transferencia de la hibridación estocástica físico--ML al contexto altoandino del Ramis;
     - comparación entre media y cuantiles del ensemble físico;
     - evaluación dual del ensemble como feature y como banda de incertidumbre;
     - auditoría de leakage temporal y backend computacional.

7. **Mantenimiento de la transparencia sobre GRU-labelled runs**
   - Se conservó, de forma más compacta, la advertencia de que los experimentos etiquetados como GRU fueron ejecutados con un MLP fallback sobre secuencias aplanadas.
   - La interpretación queda limitada a “GRU-labelled sequential hybrids under fallback execution”, no a superioridad de redes recurrentes.

## Verificación técnica

- Se verificó la sintaxis LaTeX con `pdflatex -interaction=nonstopmode`.
- La compilación produjo PDF sin error fatal.
- La introducción no presenta claves bibliográficas indefinidas respecto a `ref.bib`.
- Persisten citas indefinidas en secciones posteriores del manuscrito; se dejan para las fases de Methods, Results, Discussion y revisión LaTeX final.
- `bibtex` no estuvo disponible en el entorno de ejecución, por lo que no se regeneró `.bbl`.

## Archivos generados

- `manuscript_fase02.tex`
- `new_intro_fase02.tex`
- `cambios_realizados_fase_02.md`
- `decisiones_editoriales_fase02.md`
- `tareas_pendientes_fase02.md`
- `fase02_revision_pack.zip`
