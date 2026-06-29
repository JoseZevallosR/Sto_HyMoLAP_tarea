# Decisiones editoriales acumuladas — Fases 1 y 2

## Fase 1 — Portada, título, abstract, highlights y keywords

1. **Título declarativo**
   - Se evitó el formato interrogativo porque un título de *Journal of Hydrology* debe comunicar con precisión el aporte central.
   - El título se centró en el hallazgo robusto: los cuantiles del ensemble físico estocástico mejoran la simulación híbrida de caudal diario.

2. **Énfasis en el hallazgo robusto, no en el mejor número absoluto**
   - Aunque E8 obtuvo el mejor desempeño numérico, no se presenta como evidencia de superioridad GRU porque el backend real fue un MLP fallback.
   - El abstract prioriza E6 como evidencia robusta de hibridación cuantílica con gradient boosting.

3. **Uso de `gradient-boosting` en lugar de `XGB-labelled` en el abstract**
   - Dado que el backend real reportado para E4--E6 es `sklearn_hgb`, se evitó afirmar directamente que el resultado corresponde a XGBoost.

4. **Abstract sin citas**
   - Se mantuvo el abstract autocontenido y sin citas, como es habitual en manuscritos Elsevier.

5. **Transparencia sobre backend**
   - Se mantuvo explícitamente la aclaración de que el mejor experimento numérico fue GRU-labelled pero ejecutado como fallback MLP.

6. **Afiliación con índice corregido**
   - Aunque el bloque solicitado incluía `\affiliation[2]`, se usó `\affiliation[1]` para mantener coherencia con `\author[1]` y asegurar compilación correcta.

7. **Eliminación de placeholders visibles de portada**
   - Se eliminaron placeholders de autoría y nota de título.

8. **Keywords orientadas a indexación**
   - Las palabras clave se seleccionaron para capturar hibridación, estocasticidad, incertidumbre y contexto altoandino.

## Fase 2 — Introduction

9. **Introducción reestructurada como argumento de brecha científica**
   - La sección ya no funciona como una lista de antecedentes, sino como una progresión problema--brecha--contribución.

10. **Contexto altoandino incorporado desde el inicio**
   - El Ramis/Altiplano se introduce como un caso hidrológico exigente por estacionalidad, escasez de observaciones, bajos caudales y variabilidad hidroclimática.

11. **Equilibrio entre modelos físicos y ML**
   - Se evitó presentar el ML puro como superior por defecto.
   - Se reconocen sus ventajas predictivas, pero también sus riesgos ante extrapolación, extremos y cambio de distribución.

12. **Hibridación presentada como mecanismo, no como etiqueta**
   - La introducción diferencia claramente entre híbridos deterministas y el aporte específico del manuscrito: usar descriptores distribucionales de un ensemble físico estocástico.

13. **Brecha formulada en tres preguntas defendibles**
   - ¿Qué componente produce la mejora?
   - ¿El ensemble estocástico está calibrado como incertidumbre o solo es informativo como feature?
   - ¿Las etiquetas arquitectónicas reflejan el backend realmente ejecutado?

14. **Contribuciones explícitas y auditables**
   - Las contribuciones se expresan como cuatro aportes concretos y verificables, no como afirmaciones generales de novedad.

15. **Citas de la introducción normalizadas**
   - Todas las citas de la introducción usan claves existentes en `ref.bib`.
   - La normalización completa del resto del manuscrito queda pendiente para fases posteriores.
