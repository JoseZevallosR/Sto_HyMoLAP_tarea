#!/usr/bin/env bash
set -Eeuo pipefail

# generar_contexto.sh — StoHyMoLAP cierre Fase 2 / entrada Fase 3
#
# Uso:
#   bash generar_contexto.sh
#   bash generar_contexto.sh --with-tests
#   bash generar_contexto.sh --with-audit
#   bash generar_contexto.sh --with-tests --with-audit
#
# Salidas:
#   contextos/stohymolap_contexto_fase3_YYYYMMDD_HHMMSS.md
#   contextos/prompt_siguiente_chat_fase3_YYYYMMDD_HHMMSS.md
#   contextos/stohymolap_contexto_fase3_YYYYMMDD_HHMMSS.tar.gz
#
# Objetivo:
#   Preparar un paquete compacto para continuar en otro chat con Fase 3:
#   ablaciones, matriz experimental publicable, outputs comparables y figuras/tablas.

WITH_TESTS=0
WITH_AUDIT=0

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
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Argumento no reconocido: $1" >&2
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="contextos"
WORK_DIR="${OUT_DIR}/stohymolap_contexto_fase3_${TIMESTAMP}"
CONTEXT_MD="${WORK_DIR}/contexto_stohymolap_fase3.md"
PROMPT_MD="${WORK_DIR}/prompt_siguiente_chat_fase3.md"
MANIFEST="${WORK_DIR}/manifest.txt"
ARCHIVE="${OUT_DIR}/stohymolap_contexto_fase3_${TIMESTAMP}.tar.gz"

mkdir -p "$WORK_DIR"

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
  } >> "$CONTEXT_MD"
}

append_file_head() {
  local file="$1"
  local lines="${2:-120}"
  local title="${3:-$file}"
  {
    echo
    echo "## ${title}"
    if [[ -f "$file" ]]; then
      echo
      echo "\`${file}\`"
      echo
      echo '```text'
      sed -n "1,${lines}p" "$file" || true
      echo '```'
    else
      echo
      echo "No existe: \`${file}\`"
    fi
  } >> "$CONTEXT_MD"
}

copy_if_exists() {
  local src="$1"
  local dst_dir="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$dst_dir"
    cp -a "$src" "$dst_dir/"
    echo "$src" >> "$MANIFEST"
  fi
}

copy_selected_csv() {
  local src="$1"
  local dst_dir="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$dst_dir"
    cp "$src" "$dst_dir/"
    echo "$src" >> "$MANIFEST"
  fi
}

# Evitar arrastrar basura temporal de Python al paquete.
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

cat > "$CONTEXT_MD" <<EOF
# Contexto StoHyMoLAP para Fase 3

Generado: ${TIMESTAMP}

Directorio raíz:

\`\`\`text
${PROJECT_ROOT}
\`\`\`

## Estado conceptual

Fase 0/1 cerrada:
- Importación e instalación editable funcionando.
- Datos cargados sin relleno silencioso de Qobs con Qsim.
- Trazabilidad de origen Qobs: observed/missing/filled_from_qsim.
- Reproducibilidad agregada a config_used.

Fase 2 cerrada:
- Baseflow escalado con alpha_area.
- E1 determinístico calibra sin ruido Lévy.
- RAMIS/baseflow corre de forma continua entre train y validation.
- Qfast/Qbase consistente en t0.
- Parámetros RAMIS validados para estabilidad.
- Selección de parámetros por J multiobjetivo con strategy=best_j.
- Auditoría física E1/E2 implementada.

Resultado auditoría real Fase 2.6:
- Estado: WARN aceptable/metodológicamente justificable.
- Incidencia: PBIAS absoluto >100% solo en régimen bajo de E1.
- Métricas globales de validación aceptables:
  - E1 validation NSE ≈ 0.437, KGE ≈ 0.714, PBIAS ≈ 1.09%, Qsim/Qobs ≈ 1.009.
  - E2 validation NSE ≈ 0.426, KGE ≈ 0.724, PBIAS ≈ -0.92%, Qsim/Qobs ≈ 0.989.

Objetivo del siguiente chat:
- Entrar a Fase 3: matriz experimental de ablaciones publicable.
- Comparar RAMIS determinístico, RAMIS+Lévy, baseflow/memoria hidrológica, y componentes híbridos/ML sin crear repositorios separados.
EOF

run_cmd "Git branch" git branch --show-current
run_cmd "Git last commits" git --no-pager log --oneline -n 12
run_cmd "Git status short" git status --short
run_cmd "Git diff stat" git diff --stat
run_cmd "Git tracked files summary" git ls-files

run_cmd "Árbol compacto del proyecto" bash -lc \
  'find . -maxdepth 4 \
    -path "./.git" -prune -o \
    -path "./outputs/experiments/*/predictions*" -prune -o \
    -path "./outputs/experiments/*/physical_components*" -prune -o \
    -path "./outputs/experiments/*/ensemble*" -prune -o \
    -path "./contextos" -prune -o \
    -path "./data" -prune -o \
    -path "./__pycache__" -prune -o \
    -type f -print | sort | sed "s#^\./##"'

append_file_head "pyproject.toml" 160 "pyproject.toml"
append_file_head "requirements.txt" 180 "requirements.txt"
append_file_head ".gitignore" 160 ".gitignore"
append_file_head "README.md" 220 "README.md"
append_file_head "configs/base.yaml" 220 "configs/base.yaml"
append_file_head "configs/experiments.yaml" 300 "configs/experiments.yaml"
append_file_head "outputs/comparison/physical_audit_status.txt" 60 "physical_audit_status"
append_file_head "outputs/comparison/physical_audit_report.md" 220 "physical_audit_report.md"
append_file_head "outputs/comparison/physical_audit_issues.csv" 120 "physical_audit_issues.csv"
append_file_head "outputs/comparison/metrics_summary.csv" 120 "metrics_summary.csv si existe"
append_file_head "outputs/comparison/experiment_comparison.csv" 120 "experiment_comparison.csv si existe"

run_cmd "Resumen de outputs E1/E2" bash -lc \
  'for e in E1_RAMIS_DET_BF E2_RAMIS_LEVY_BF; do
      d="outputs/experiments/$e"
      echo "### $e"
      if [[ -d "$d" ]]; then
        find "$d" -maxdepth 1 -type f | sort | sed "s#^#- #"
        for f in best_parameters.csv top_k_parameters.csv metrics_train.csv metrics_validation.csv metrics_by_regime.csv config_used.yaml; do
          if [[ -f "$d/$f" ]]; then
            echo
            echo "--- $d/$f"
            sed -n "1,40p" "$d/$f"
          fi
        done
      else
        echo "No existe $d"
      fi
      echo
    done'

if [[ "$WITH_TESTS" -eq 1 ]]; then
  run_cmd "pytest completo" bash -lc 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q'
else
  {
    echo
    echo "## Tests"
    echo
    echo "No se ejecutaron tests. Para incluirlos:"
    echo
    echo "\`\`\`bash"
    echo "bash generar_contexto.sh --with-tests"
    echo "\`\`\`"
  } >> "$CONTEXT_MD"
fi

if [[ "$WITH_AUDIT" -eq 1 ]]; then
  run_cmd "Auditoría física Fase 2.6 E1/E2" bash -lc \
    'python scripts/audit_physical_phase26.py --config configs/experiments.yaml --experiments E1_RAMIS_DET_BF E2_RAMIS_LEVY_BF --run'
else
  {
    echo
    echo "## Auditoría física"
    echo
    echo "No se re-ejecutó auditoría. Para incluirla:"
    echo
    echo "\`\`\`bash"
    echo "bash generar_contexto.sh --with-audit"
    echo "\`\`\`"
  } >> "$CONTEXT_MD"
fi

cat > "$PROMPT_MD" <<'EOF'
# Prompt para siguiente chat — StoHyMoLAP Fase 3

Continuemos con StoHyMoLAP desde Fase 3.

Fase 0/1 y Fase 2 ya están cerradas. Fase 2 cerró con auditoría física `WARN` justificable por PBIAS >100% solo en régimen bajo de E1; las métricas globales de validación son aceptables. La implementación física RAMIS/baseflow quedó corregida:
- `Qobs` no se rellena silenciosamente con `Qsim`.
- `baseflow` se escala con `alpha_area`.
- E1 determinístico calibra sin ruido Lévy.
- La validación no reinicia con `Qobs` observado; se simula train+validation de forma continua.
- `Q0 = Qfast0 + Qbase0`, sin duplicar caudal inicial.
- Los parámetros RAMIS tienen validación de estabilidad.
- La calibración selecciona la mejor fila por `J` multiobjetivo (`parameter_selection: best_j`).
- Se generan componentes físicos `Qfast`, `Qbase`, `Qtotal` y auditoría reproducible.

Necesito implementar la Fase 3: matriz experimental de ablaciones publicable, comparando RAMIS determinístico, RAMIS+Lévy, baseflow/memoria hidrológica y componentes híbridos/ML sin crear repositorios separados.

Objetivos de Fase 3:
1. Definir matriz experimental clara y defendible.
2. Revisar/ajustar `configs/experiments.yaml` para experimentos de ablación.
3. Garantizar que cada experimento tenga outputs comparables.
4. Generar tablas de comparación global y por régimen.
5. Preparar figuras para paper: hidrogramas, FDC, dispersión Qobs-Qsim, residuos por régimen, incertidumbre/ensemble si aplica.
6. Evitar fuga de información entre train y validation.
7. Mantener reproducibilidad por seed, config_used, git commit y auditoría.
8. Dejar scripts de corrida y validación para todos los experimentos.

Primero revisa el contexto adjunto y dime:
- Si Fase 3 puede empezar sin modificar más la física.
- Qué experimentos mínimos y extendidos deben formar la matriz.
- Qué archivos conviene modificar primero.
- Qué criterios de aceptación debe tener Fase 3.
Después genera los parches necesarios por subfase.
EOF

# Copia de archivos clave al paquete.
mkdir -p "${WORK_DIR}/repo"

for f in \
  "pyproject.toml" \
  "requirements.txt" \
  ".gitignore" \
  "README.md" \
  "generar_contexto.sh" \
  "configs/base.yaml" \
  "configs/experiments.yaml" \
  "scripts/run_all_experiments.py" \
  "scripts/audit_physical_phase26.py" \
  "tests/test_smoke.py"
do
  if [[ -f "$f" ]]; then
    mkdir -p "${WORK_DIR}/repo/$(dirname "$f")"
    cp "$f" "${WORK_DIR}/repo/$f"
    echo "$f" >> "$MANIFEST"
  fi
done

for d in \
  "src/stohymolap/data" \
  "src/stohymolap/hydro" \
  "src/stohymolap/stochastic" \
  "src/stohymolap/calibration" \
  "src/stohymolap/experiments" \
  "src/stohymolap/metrics" \
  "src/stohymolap/diagnostics" \
  "src/stohymolap/features" \
  "src/stohymolap/ml" \
  "src/stohymolap/plotting" \
  "src/stohymolap/utils"
do
  if [[ -d "$d" ]]; then
    mkdir -p "${WORK_DIR}/repo/$(dirname "$d")"
    rsync -a \
      --exclude="__pycache__" \
      --exclude="*.pyc" \
      "$d" "${WORK_DIR}/repo/$(dirname "$d")/"
    echo "$d" >> "$MANIFEST"
  fi
done

# Reportes y CSV pequeños de auditoría/comparación.
mkdir -p "${WORK_DIR}/repo/outputs/comparison"
for f in \
  "outputs/comparison/physical_audit_status.txt" \
  "outputs/comparison/physical_audit_report.md" \
  "outputs/comparison/physical_audit_issues.csv" \
  "outputs/comparison/physical_audit_summary.csv" \
  "outputs/comparison/metrics_summary.csv" \
  "outputs/comparison/experiment_comparison.csv" \
  "outputs/comparison/run_all.log" \
  "outputs/comparison/phase26_audit.log"
do
  copy_selected_csv "$f" "${WORK_DIR}/repo/outputs/comparison"
done

# Parámetros y métricas por experimento. Evita series pesadas.
for e in E1_RAMIS_DET_BF E2_RAMIS_LEVY_BF; do
  srcd="outputs/experiments/${e}"
  dstd="${WORK_DIR}/repo/outputs/experiments/${e}"
  if [[ -d "$srcd" ]]; then
    mkdir -p "$dstd"
    for f in \
      "best_parameters.csv" \
      "top_k_parameters.csv" \
      "metrics_train.csv" \
      "metrics_validation.csv" \
      "metrics_by_regime.csv" \
      "config_used.yaml" \
      "run.log"
    do
      copy_selected_csv "${srcd}/${f}" "$dstd"
    done

    # Solo primeras filas de series pesadas para contexto.
    for f in \
      "predictions_train.csv" \
      "predictions_validation.csv" \
      "physical_components_train.csv" \
      "physical_components_validation.csv"
    do
      if [[ -f "${srcd}/${f}" ]]; then
        head -n 80 "${srcd}/${f}" > "${dstd}/${f%.csv}_HEAD.csv"
        echo "${srcd}/${f} -> HEAD" >> "$MANIFEST"
      fi
    done
  fi
done

# Agrega manifest final.
{
  echo "# Manifest"
  echo
  sort -u "$MANIFEST" 2>/dev/null || true
} > "${WORK_DIR}/manifest.md"

tar -czf "$ARCHIVE" -C "$OUT_DIR" "$(basename "$WORK_DIR")"

echo
echo "Contexto generado:"
echo "  ${CONTEXT_MD}"
echo "Prompt siguiente chat:"
echo "  ${PROMPT_MD}"
echo "Paquete tar.gz:"
echo "  ${ARCHIVE}"
echo
echo "Adjunta en el siguiente chat el .tar.gz y pega el contenido de:"
echo "  ${PROMPT_MD}"
echo