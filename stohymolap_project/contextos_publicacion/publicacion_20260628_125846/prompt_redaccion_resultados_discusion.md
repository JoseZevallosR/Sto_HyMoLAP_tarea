# Prompt para redacción de Resultados y Discusión — StoHyMoLAP

Quiero redactar las secciones **Resultados** y **Discusión** de un manuscrito científico sobre StoHyMoLAP/RAMIS.

Usa el paquete de contexto adjunto. Prioriza:

1. `contexto_publicacion_stohymolap.md`.
2. `outputs/reports/phase34/phase34_results_report.md`.
3. `table_final_ranking_common_window.csv`.
4. `table_ablation_effects_common_window.csv`.
5. `table_regime_metrics_selected.csv`.
6. `table_backend_manuscript_notes.csv`.
7. `phase34_figure_manifest.csv`.
8. El paper base `houenafa2025hybridization.pdf` solo como referencia de estructura y estilo científico, no para copiar texto.

Instrucciones de redacción:

- Redacta en estilo artículo científico, sobrio y publicable.
- No sobreafirmes causalidad; distingue evidencia empírica, interpretación hidrológica y limitaciones.
- No afirmar que E7/E8 son GRU reales si el backend dice `fallback_mlp_on_flattened_sequences`. Redactar como `GRU-labelled hybrid with fallback MLP backend` o equivalente, salvo que exista evidencia nueva de backend recurrente real.
- Usar la ventana común de validación para comparar E0-E8.
- Explicar por qué baseflow/memoria hidrológica mejora frente al modelo sin reservorio.
- Explicar el aporte y limitación de las perturbaciones Lévy.
- Explicar por qué los híbridos con cuantiles pueden mejorar frente a modelos físicos/ML puros.
- Incluir un párrafo crítico sobre leakage: auditoría WARN pero sin FAIL; no hay evidencia de leakage temporal según los artefactos.
- Incluir la limitación de subdispersión/fallback para bandas estocásticas si el manifest lo reporta.
- Proponer subtítulos de Resultados y Discusión compatibles con la estructura LaTeX incluida.

Entrega solicitada:

1. Redacción en inglés académico de `Results`.
2. Redacción en inglés académico de `Discussion`.
3. Una tabla de correspondencia figura/resultado/mensaje.
4. Una lista de claims permitidos, claims que deben matizarse y claims prohibidos.
5. Un texto breve para `Limitations`.
6. Un texto breve para `Conclusions`.
