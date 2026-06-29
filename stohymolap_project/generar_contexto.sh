#!/usr/bin/env bash
set -Eeuo pipefail

# Generador de contexto — StoHyMoLAP Fase 3.4B
# Uso:
#   bash generar_contexto.sh
#   bash generar_contexto.sh --with-tests
#   bash generar_contexto.sh --with-tests --with-audit
#
# Objetivo: preparar un paquete pequeño y útil para continuar en otro chat
# desde Fase 3.4A cerrada hacia Fase 3.4B figuras paper-ready.

PROJECT_ROOT="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PHASE="fase3_4B"
CONTEXT_DIR="contextos"
OUT_MD="contexto_stohymolap_${PHASE}.md"
PROMPT_MD="prompt_siguiente_chat_${PHASE}.md"
SNAPSHOT="${CONTEXT_DIR}/stohymolap_contexto_${PHASE}_${TIMESTAMP}.tar.gz"
FILELIST="${CONTEXT_DIR}/stohymolap_contexto_${PHASE}_${TIMESTAMP}_filelist.txt"

WITH_TESTS=0
WITH_AUDIT=0
WITH_STATUS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-tests)
      WITH_TESTS=1
      shift
      ;;
    --with-audit)
      WITH_AUDIT=1
      shift
      ;;
    --no-status)
      WITH_STATUS=0
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Genera contexto para StoHyMoLAP Fase 3.4B.

Opciones:
  --with-tests     Ejecuta pytest -q y adjunta el resumen.
  --with-audit     Ejecuta scripts/audit_leakage_phase34.py si existe.
  --no-status      Omite algunos diagnósticos de estado.
HELP
      exit 0
      ;;
    *)
      echo "Opcion no reconocida: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$CONTEXT_DIR"

run_cmd() {
  local title="$1"
  shift
  {
    echo
    echo "## ${title}"
    echo
    echo '```text'
    "$@" 2>&1 || true
    echo '```'
  } >> "$OUT_MD"
}

append_file_if_exists() {
  local title="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    {
      echo
      echo "## ${title}"
      echo
      echo "Archivo: ${path}"
      echo
      echo '```text'
      cat "$path" || true
      echo '```'
    } >> "$OUT_MD"
  fi
}

# -----------------------------------------------------------------------------
# 1) Contexto markdown principal
# -----------------------------------------------------------------------------
{
  echo "# Contexto StoHyMoLAP — Fase 3.4B"
  echo
  echo "Generado: ${TIMESTAMP}"
  echo "Proyecto: ${PROJECT_ROOT}"
  echo
} > "$OUT_MD"

cat >> "$OUT_MD" <<'MD'
## Estado de avance

Fase 3.4A cerrada funcionalmente:

- Auditor anti-leakage y QA implementado.
- Compatibilidad con outputs legacy corregida.
- Tests reportados por el usuario: 26 passed, 2 warnings.
- Commit principal: d369114 — feat: add phase 3.4A leakage audit and backend QA.
- Commit correctivo: e75439c — fix: make leakage audit compatible with legacy outputs.
- Auditoría actual: WARN, no FAIL.

Interpretación del WARN:

- No se observó leakage temporal: train termina antes de validation.
- Las ventanas de evaluación son consistentes.
- La intersección común cubre los 9 experimentos.
- Persisten brechas observacionales de Qobs en físicos E0-E3.
- E7/E8 no deben redactarse como GRU real en este entorno; el backend registrado es fallback_mlp_on_flattened_sequences.
- Algunos outputs legacy pueden requerir rerun para completar trazabilidad de postprocessing_report.json y regime_thresholds.json.

## Objetivo de la Fase 3.4B

Generar figuras paper-ready para StoHyMoLAP:

1. Hidrograma de validación para E0, E1, E3, E4, E6 y E8.
2. Curva de duración de caudales, FDC, comparando Qobs y Qsim.
3. Scatter Qobs-Qsim.
4. Residuos por régimen hidrológico.
5. Componentes físicos Qfast, Qbase y Qtotal si están disponibles.
6. Bandas estocásticas E2/E3 como referencia física, con advertencia explícita de subdispersión.
7. Manifest de figuras con rutas, modelos usados, columnas requeridas y notas metodológicas.

## Reglas metodológicas para 3.4B

- Usar la misma ventana común/intersección exacta usada por el leaderboard.
- No recalcular métricas con fechas distintas a las de comparación común.
- Excluir Qobs faltante de métricas y gráficos estadísticos que lo requieran.
- Documentar explícitamente que E7/E8 son fallback MLP si no hay TensorFlow/PyTorch.
- No vender las bandas E2/E3 como incertidumbre calibrada; tratarlas como referencia estocástica subdispersa.
- Guardar figuras en outputs/figures/phase34/.
- Guardar un manifest en outputs/figures/phase34/figure_manifest.csv y/o JSON.
MD

if [[ "$WITH_STATUS" -eq 1 ]]; then
  run_cmd "pwd" pwd
  run_cmd "git status --short" git status --short
  run_cmd "ultimos commits" git log --oneline -8
  run_cmd "archivos pyc versionados" bash -lc "git ls-files | grep -E '(__pycache__|\.py[co]$)' || true"
  run_cmd "estructura resumida" bash -lc "find . -maxdepth 3 -type f \
    ! -path './.git/*' \
    ! -path './contextos/*' \
    ! -path './outputs/experiments/*/model_artifact/*' \
    ! -path './outputs/figures/*' \
    | sort | sed 's#^./##' | head -300"
fi

append_file_if_exists "README" "README.md"
append_file_if_exists "Configuracion experimental" "configs/experiments.yaml"
append_file_if_exists "Auditoria leakage status" "outputs/comparison/leakage_audit_status.txt"
append_file_if_exists "Auditoria leakage report" "outputs/comparison/leakage_audit_report.md"
append_file_if_exists "Leaderboard comun" "outputs/comparison/leaderboard_common_intersection.csv"
append_file_if_exists "Metricas validation comun" "outputs/comparison/validation_metrics_common_intersection.csv"
append_file_if_exists "Efectos de ablacion" "outputs/comparison/ablation_effects_common_intersection.csv"
append_file_if_exists "Metricas por regimen" "outputs/comparison/regime_metrics_comparison.csv"
append_file_if_exists "Resumen backend ML" "outputs/comparison/ml_backend_summary.csv"
append_file_if_exists "Issues auditoria leakage" "outputs/comparison/leakage_audit_issues.csv"

if [[ "$WITH_AUDIT" -eq 1 && -f "scripts/audit_leakage_phase34.py" && -f "configs/experiments.yaml" ]]; then
  run_cmd "Ejecucion auditoria leakage" python scripts/audit_leakage_phase34.py --config configs/experiments.yaml
  append_file_if_exists "Auditoria leakage status posterior" "outputs/comparison/leakage_audit_status.txt"
  append_file_if_exists "Auditoria leakage report posterior" "outputs/comparison/leakage_audit_report.md"
fi

if [[ "$WITH_TESTS" -eq 1 ]]; then
  run_cmd "pytest -q" pytest -q
fi

cat >> "$OUT_MD" <<'MD'
## Checklist para el siguiente chat

Solicitar Fase 3.4B con estos entregables:

- Nuevo módulo o script para generación de figuras paper-ready.
- Funciones reutilizables para cargar predicciones por experimento.
- Uso consistente de la ventana común.
- Figuras guardadas en PNG y, preferentemente, SVG/PDF.
- Manifest de figuras.
- Tests smoke que validen que el script corre con datos mínimos.
- README actualizado con rutas de figuras y cautelas metodológicas.

## Archivos importantes esperados

- configs/experiments.yaml
- scripts/run_all_experiments.py
- scripts/summarize_results.py
- scripts/audit_leakage_phase34.py
- src/stohymolap/experiments/runner.py
- src/stohymolap/experiments/comparison.py
- src/stohymolap/diagnostics/leakage_audit.py
- outputs/comparison/leaderboard_common_intersection.csv
- outputs/comparison/regime_metrics_comparison.csv
- outputs/comparison/ml_backend_summary.csv
- outputs/experiments/*/predictions_validation.csv
- outputs/experiments/*/evaluation_window.json
- outputs/experiments/*/ml_backend.json
- outputs/experiments/*/postprocessing_report.json
- outputs/experiments/*/regime_thresholds.json
MD

# -----------------------------------------------------------------------------
# 2) Prompt para siguiente chat
# -----------------------------------------------------------------------------
cat > "$PROMPT_MD" <<'MD'
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
MD

# -----------------------------------------------------------------------------
# 3) Lista de archivos para el tar.gz
# -----------------------------------------------------------------------------
: > "$FILELIST"

add_if_exists() {
  local p="$1"
  if [[ -e "$p" ]]; then
    printf '%s\n' "$p" >> "$FILELIST"
  fi
  return 0
}

add_if_exists "$OUT_MD"
add_if_exists "$PROMPT_MD"
add_if_exists "README.md"
add_if_exists "pyproject.toml"
add_if_exists "requirements.txt"
add_if_exists ".gitignore"
add_if_exists "configs/experiments.yaml"

for d in scripts src tests; do
  if [[ -d "$d" ]]; then
    find "$d" -type f \
      ! -path '*/__pycache__/*' \
      ! -name '*.pyc' \
      ! -name '*.pyo' \
      | sort >> "$FILELIST"
  fi
done

if [[ -d "outputs/comparison" ]]; then
  find outputs/comparison -maxdepth 1 -type f \
    \( -name '*.csv' -o -name '*.md' -o -name '*.txt' -o -name '*.json' \) \
    | sort >> "$FILELIST"
fi

if [[ -d "outputs/experiments" ]]; then
  find outputs/experiments -mindepth 2 -maxdepth 2 -type f \
    \( \
      -name 'predictions_validation.csv' -o \
      -name 'predictions_train.csv' -o \
      -name 'evaluation_window.json' -o \
      -name 'ml_backend.json' -o \
      -name 'postprocessing_report.json' -o \
      -name 'regime_thresholds.json' -o \
      -name 'best_parameters.csv' \
    \) \
    | sort >> "$FILELIST"
fi

# Evitar duplicados y entradas inexistentes.
sort -u "$FILELIST" -o "$FILELIST"

if [[ ! -s "$FILELIST" ]]; then
  echo "No se encontraron archivos para empaquetar." >&2
  exit 1
fi

tar -czf "$SNAPSHOT" -T "$FILELIST"

cat <<EOF
OK: contexto generado.

Markdown principal:
  ${OUT_MD}

Prompt siguiente chat:
  ${PROMPT_MD}

Paquete para adjuntar:
  ${SNAPSHOT}

Lista de archivos:
  ${FILELIST}

Uso sugerido:
  1) Adjunta ${SNAPSHOT} en el nuevo chat.
  2) Pega el contenido de ${PROMPT_MD}.
EOF