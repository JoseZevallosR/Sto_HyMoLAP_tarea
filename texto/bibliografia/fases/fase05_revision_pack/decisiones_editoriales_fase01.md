# Decisiones editoriales — Fase 1

1. **Título declarativo**
   - Se evitó el formato interrogativo porque un título de Journal of Hydrology debe comunicar con precisión el aporte central, no plantear una pregunta abierta.
   - La decisión editorial fue centrar el título en el hallazgo robusto: los cuantiles del ensemble físico estocástico mejoran la simulación híbrida de caudal diario.

2. **Énfasis en el hallazgo robusto, no en el mejor número absoluto**
   - Aunque E8 obtuvo el mejor desempeño numérico, el manuscrito no debe venderlo como evidencia de superioridad GRU porque el backend real fue un MLP fallback.
   - Por ello, el abstract prioriza E6 como evidencia robusta de hibridación cuantílica con gradient boosting.

3. **Uso de `gradient-boosting` en lugar de `XGB-labelled` en el abstract**
   - Dado que el backend real reportado para E4--E6 es `sklearn_hgb`, se decidió no afirmar directamente que el resultado corresponde a XGBoost.
   - Esto fortalece la transparencia metodológica y evita una posible observación crítica de reviewer.

4. **Abstract sin citas**
   - Se eliminó la carga de citas en el abstract para hacerlo autocontenido, más fluido y menos dependiente de claves bibliográficas que actualmente presentan inconsistencias.
   - Las citas deben sostenerse en la Introducción, Métodos y Discusión, no en el resumen.

5. **Transparencia sobre backend**
   - Se mantuvo explícitamente la aclaración de que el mejor experimento numérico fue GRU-labelled pero ejecutado como fallback MLP.
   - Esta decisión protege el manuscrito frente a críticas de reproducibilidad y sobreafirmación arquitectónica.

6. **Afiliación con índice corregido**
   - Aunque el bloque solicitado incluía `\affiliation[2]`, se usó `\affiliation[1]` para mantener coherencia con `\author[1]` y asegurar compilación correcta.
   - Si luego se agregan coautores o una afiliación institucional previa, este índice puede ajustarse.

7. **Eliminación de placeholders visibles**
   - Se eliminaron placeholders de autoría y nota de título porque proyectan un manuscrito incompleto y no deben aparecer en una versión cercana a envío.

8. **Keywords orientadas a indexación**
   - Las palabras clave se seleccionaron para mejorar descubribilidad en bases como Scopus/Web of Science y para capturar los tres ejes metodológicos: hibridación, estocasticidad e incertidumbre.
