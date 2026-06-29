# Cambios realizados - Fase 7

## Alcance
Fase centrada en limpieza editorial final de captions, claims y unidades, sin reabrir la estructura científica ya revisada en las fases 1--6.

## Cambios aplicados

1. **Captions de figuras y tablas**
   - Se reescribieron los captions para que sean más informativos y menos promocionales.
   - Se añadieron advertencias explícitas en captions asociados a experimentos GRU-labelled, indicando que deben interpretarse bajo auditoría de backend porque se ejecutaron como fallback MLP.
   - Se ajustaron captions de incertidumbre para declarar que PICP/PINAW son adimensionales y que MPIW/Winkler conservan unidades de caudal.
   - Se ajustaron captions de ranking, ablation, hydrograph, FDC y regime-RMSE para indicar explícitamente unidades y ventana común de validación cuando correspondía.

2. **Unidades**
   - Se añadió la macro LaTeX `\cms` para expresar `m$^3$\,s$^{-1}$` de forma consistente.
   - En tablas se explicitaron las unidades de RMSE, MAE, MPIW y Winkler score.
   - Se mantuvieron NSE, KGE, R2, PICP y PINAW como métricas adimensionales.
   - PBIAS se etiquetó explícitamente como porcentaje.

3. **Tablas**
   - Se reformateó la matriz experimental, el ranking común, la tabla de ablation, la tabla por regímenes, la tabla de incertidumbre, la auditoría de backend y el checklist de reproducibilidad.
   - Se reemplazaron etiquetas largas con identificadores LaTeX rompibles mediante `\modelid{}` para reducir problemas de ancho.
   - Se abreviaron backends en tablas como HGB y fallback MLP, con notas explicativas.

4. **Claims y tono editorial**
   - Se suavizaron expresiones como “dominant driver”, “clear separation”, “confirms” y “substantially reduced”.
   - Se mantuvo el hallazgo principal, pero expresado como evidencia específica del experimento y no como generalización universal.
   - Se conservó la distinción central: E8 es el mejor resultado numérico, pero E6 es el resultado backend-consistente más defendible.

5. **Compilación y verificación visual**
   - Se compiló con `pdflatex + bibtex8 + pdflatex + pdflatex + pdflatex`.
   - No quedan citas indefinidas ni referencias cruzadas indefinidas.
   - Se renderizó el PDF final y se inspeccionaron visualmente las páginas críticas con tablas y captions.
   - Persisten únicamente advertencias internas menores de `cas-sc` durante `\maketitle`/hyperref; no se observó impacto visual en el PDF renderizado.

## Archivos producidos

- `manuscript_fase07.tex`
- `manuscript_fase07.pdf`
- `manuscript_fase07.log`
- `fase07_revision_pack.zip`
