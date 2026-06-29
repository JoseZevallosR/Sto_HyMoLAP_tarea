Continuemos con StoHyMoLAP desde Fase 3.4B.

Estado actual:

- Fase 3.4A cerrada y commiteada.
- Tests reportados: 26 passed, 2 warnings.
- Auditoría anti-leakage: WARN, no FAIL.
- No hay evidencia de leakage temporal: train y validation están separados; la ventana común es consistente.
- Se regeneró la matriz E0-E8 y el leaderboard común.
- E8_HYB_GRU_QUANTILES sigue como mejor modelo: NSE aproximado 0.8491, KGE aproximado 0.8986.
- E7/E8 están declarados como GRU, pero en este entorno corren con backend fallback_mlp_on_flattened_sequences por ausencia de TensorFlow/PyTorch. No deben redactarse como GRU real.

Quiero implementar la Fase 3.4B: figuras paper-ready.

Objetivo:

1. Crear un script, por ejemplo scripts/generate_phase34_figures.py.
2. Guardar figuras en outputs/figures/phase34/.
3. Generar un manifest de figuras en CSV/JSON.
4. Actualizar README con la sección de figuras.
5. Agregar tests smoke para validar carga de datos y generación mínima.

Figuras requeridas:

- Hidrograma de validación para E0_RAMIS_DET_NOBF, E1_RAMIS_DET_BF, E3_RAMIS_LEVY_BF, E4_ML_PURE_XGB, E6_HYB_XGB_QUANTILES y E8_HYB_GRU_QUANTILES.
- FDC comparando Qobs y Qsim para modelos seleccionados.
- Scatter Qobs-Qsim para modelos seleccionados.
- Residuos por régimen hidrológico.
- Componentes físicos Qfast, Qbase y Qtotal si están disponibles.
- Bandas E2/E3 como referencia estocástica, con advertencia de subdispersión.

Restricciones:

- Usar la ventana común/intersección exacta ya definida por outputs/comparison.
- Excluir Qobs faltantes en cálculos que lo requieran.
- No introducir dependencia pesada ni seaborn.
- Usar matplotlib/pandas/numpy.
- No asumir que E7/E8 son GRU real; documentar backend.
- Si una columna no existe, la figura debe degradar con advertencia controlada y registrarlo en el manifest.

Primero revisa el contexto adjunto y dime qué columnas existen en predictions_validation.csv por experimento, qué figuras son directamente posibles y cuáles requieren fallback. Luego genera el parche de Fase 3.4B.
