# Decisiones editoriales — Fase 5: Discussion

## 1. Claim principal deliberadamente estrecho
La discusión se redactó para sostener un claim fuerte pero defendible: los cuantiles del ensemble físico estocástico mejoran la simulación híbrida backend-consistente de caudal diario. Se evitó afirmar que el método ya produce incertidumbre calibrada o que demuestra superioridad de redes recurrentes.

## 2. Separación entre desempeño numérico y evidencia robusta
Se mantiene la diferencia entre:

- `E8_HYB_GRU_QUANTILES`: mejor resultado numérico, pero ejecutado como fallback MLP.
- `E6_HYB_XGB_QUANTILES`: mejor resultado robusto y defendible porque el label y el backend ejecutado son consistentes.

Esta separación es crucial para una revisión tipo Journal of Hydrology.

## 3. Baseflow como memoria mínima, no solución completa
La discusión no presenta el reservorio lineal como una solución suficiente para bajo caudal. Se reconoce su contribución global, pero se aclara que su efecto no reduce el RMSE de bajo caudal en la tabla de regímenes.

## 4. Lévy como generador de features, no como producto probabilístico calibrado
El componente Lévy se interpreta como fuente de diversidad estocástica útil para construir descriptores cuantílicos. No se presenta como banda de incertidumbre operacional debido al bajo PICP.

## 5. Lenguaje más conservador y menos promocional
Se reemplazaron formulaciones demasiado fuertes por expresiones condicionales y auditables. La discusión enfatiza mecanismos plausibles, evidencia directa del ablation y limitaciones pendientes.

## 6. Citas normalizadas
Se evitó mantener citas textuales largas y se usaron referencias en modo argumentativo. Las claves abreviadas se reemplazaron por claves reales existentes en `ref.bib`.

## 7. Advertencias mantenidas visibles
Se decidió mantener el estado WARN como parte de la narrativa científica. Ocultarlo debilitaría la trazabilidad del estudio; reportarlo fortalece la credibilidad del manuscrito.
