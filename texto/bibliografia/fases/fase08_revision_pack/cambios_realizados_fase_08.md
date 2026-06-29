# Cambios realizados - Fase 8

## Alcance
Corrección metodológica focalizada en datos, área de estudio y exclusión de la narrativa del software.

## Cambios principales

1. **Exclusión del software de la narrativa**
   - Se retiraron las referencias al software desarrollado como producto del artículo.
   - La narrativa ahora se centra en el diseño de modelación, la matriz experimental, los datos hidro-meteorológicos y la evidencia de ablation.
   - La sección `Code and data availability` fue reemplazada por `Data availability`.
   - Se eliminaron referencias a scripts, paquetes de resultados, rutas internas y repositorios aún no asignados.

2. **Descripción formal de datos**
   - Se incorporó PISCOp v3.0 como fuente de precipitación diaria.
   - Se aclaró que PISCOp v3.0 es una actualización 2026 y que la cita a Aybar et al. (2020) documenta la familia PISCO/PISCOp y la metodología publicada para v2.1, no la versión exacta v3.0.
   - Se incorporó PISCOt v1.2 como fuente de temperatura mínima y máxima diaria.
   - Se añadió la referencia Huerta et al. (2023) para PISCOt v1.2.
   - Se especificó que precipitación y temperatura se agregan espacialmente sobre la cuenca delineada para obtener series diarias medias de cuenca.

3. **Evapotranspiración**
   - Se incorporó la ecuación de Hargreaves--Samani para estimar PET a partir de Tmin y Tmax.
   - Se definieron explícitamente `Tmean`, `Ra` y las unidades de PET.
   - Se añadió la referencia Hargreaves and Samani (1985).

4. **Área de estudio**
   - Se reescribió la descripción del área de estudio con mayor conexión entre geografía, forzantes PISCO/PISCOt y unidad hidrológica simulada.
   - Se insertó un bloque de figura para `figures/study_area_ramis.png` con placeholder compilable.
   - Se redactó un caption profesional estilo Journal of Hydrology para el mapa del área de estudio.

5. **Figura metodológica**
   - Se actualizó `method_workflow.dot` y `method_workflow.png` para indicar PISCOp v3.0, PISCOt v1.2 y PET por Hargreaves--Samani.
   - Se reemplazó la frase “Paper-ready outputs” por “Manuscript outputs”.

6. **Referencias**
   - Se añadieron a `ref.bib` las entradas BibTeX de Aybar et al. (2020), Huerta et al. (2023) y Hargreaves and Samani (1985).

## Compilación
- Compilación verificada con `pdflatex + bibtex8 + pdflatex + pdflatex`.
- No quedaron citas indefinidas ni referencias cruzadas indefinidas.
- Persisten advertencias menores propias de la clase `cas-sc` y duplicación interna de anchors de figuras, sin error fatal.
