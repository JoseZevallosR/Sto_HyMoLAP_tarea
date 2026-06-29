#!/usr/bin/env bash
set -Eeuo pipefail

# StoHyMoLAP publication-context generator
# Genera un paquete reproducible para redactar Resultados y Discusión del paper.
# Uso recomendado desde la raíz del repo:
#   bash generate_context_publicacion.sh
#   bash generate_context_publicacion.sh --refresh --with-figures --with-pdf
#
# Salidas principales:
#   contextos_publicacion/<timestamp>/contexto_publicacion_stohymolap.md
#   contextos_publicacion/<timestamp>/paper_stohymolap_template.tex
#   contextos_publicacion/<timestamp>/prompt_redaccion_resultados_discusion.md
#   contextos_publicacion/stohymolap_publicacion_context_<timestamp>.tar.gz

PROJECT_ROOT="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT_DIR="contextos_publicacion"
OUT_DIR="${BASE_OUT_DIR}/publicacion_${TIMESTAMP}"
FILES_DIR="${OUT_DIR}/files"
CONTEXT_MD="${OUT_DIR}/contexto_publicacion_stohymolap.md"
PROMPT_MD="${OUT_DIR}/prompt_redaccion_resultados_discusion.md"
TEX_FILE="${OUT_DIR}/paper_stohymolap_template.tex"
MANIFEST="${OUT_DIR}/manifest_publicacion.txt"
TAR_FILE="${BASE_OUT_DIR}/stohymolap_publicacion_context_${TIMESTAMP}.tar.gz"

INCLUDE_FIGURES=1
INCLUDE_PDF=1
REFRESH=0
MAKE_TAR=1
MAX_CSV_LINES=250
MAX_TEXT_LINES=350
REFERENCE_PAPER="${REFERENCE_PAPER:-houenafa2025hybridization.pdf}"

usage() {
  cat <<'HELP'
Uso:
  bash generate_context_publicacion.sh [opciones]

Opciones:
  --refresh             Regenera figuras/reporte si existen los scripts correspondientes.
  --with-figures        Incluye PNG de outputs/figures/phase34/ en el paquete. Default: sí.
  --no-figures          No copia PNG, solo manifests.
  --with-pdf            Incluye houenafa2025hybridization.pdf si existe. Default: sí.
  --no-pdf              No copia el PDF base.
  --no-tar              No comprime el paquete.
  --max-csv-lines N     Máximo de líneas por CSV en el contexto Markdown. Default: 250.
  --max-text-lines N    Máximo de líneas por archivo de texto en el contexto Markdown. Default: 350.
  -h, --help            Muestra esta ayuda.

Variable opcional:
  REFERENCE_PAPER=/ruta/al/paper.pdf bash generate_context_publicacion.sh --with-pdf
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh) REFRESH=1; shift ;;
    --with-figures) INCLUDE_FIGURES=1; shift ;;
    --no-figures) INCLUDE_FIGURES=0; shift ;;
    --with-pdf) INCLUDE_PDF=1; shift ;;
    --no-pdf) INCLUDE_PDF=0; shift ;;
    --no-tar) MAKE_TAR=0; shift ;;
    --max-csv-lines) MAX_CSV_LINES="${2:?Falta valor para --max-csv-lines}"; shift 2 ;;
    --max-text-lines) MAX_TEXT_LINES="${2:?Falta valor para --max-text-lines}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opción no reconocida: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$FILES_DIR"
: > "$MANIFEST"

log() { printf '[publicacion] %s\n' "$*"; }
warn() { printf '[publicacion][WARN] %s\n' "$*" >&2; }

copy_if_exists() {
  local src="$1"
  local dst_base="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "${dst_base}/$(dirname "$src")"
    cp -a "$src" "${dst_base}/${src}"
    echo "$src" >> "$MANIFEST"
  fi
}

copy_glob_if_exists() {
  local pattern="$1"
  local dst_base="$2"
  shopt -s nullglob globstar
  local matches=( $pattern )
  shopt -u nullglob globstar
  if (( ${#matches[@]} > 0 )); then
    local f
    for f in "${matches[@]}"; do
      [[ -e "$f" ]] && copy_if_exists "$f" "$dst_base"
    done
  fi
}

append_cmd() {
  local title="$1"; shift
  {
    echo
    echo "### ${title}"
    echo
    echo '```bash'
    printf '$ %q' "$1"
    shift || true
    local arg
    for arg in "$@"; do printf ' %q' "$arg"; done
    echo
    echo '```'
    echo
    echo '```text'
    ( "$@" ) 2>&1 || true
    echo '```'
  } >> "$CONTEXT_MD"
}

append_cmd_raw() {
  local title="$1"; shift
  {
    echo
    echo "### ${title}"
    echo
    echo '```text'
    ( "$@" ) 2>&1 || true
    echo '```'
  } >> "$CONTEXT_MD"
}

append_file_preview() {
  local title="$1"
  local file="$2"
  local max_lines="${3:-$MAX_TEXT_LINES}"
  if [[ -f "$file" ]]; then
    {
      echo
      echo "### ${title}"
      echo
      echo "Archivo: \`${file}\`"
      echo
      echo '```text'
      sed -n "1,${max_lines}p" "$file" || true
      local total
      total="$(wc -l < "$file" 2>/dev/null || echo 0)"
      if [[ "$total" =~ ^[0-9]+$ ]] && (( total > max_lines )); then
        echo
        echo "[TRUNCADO: se muestran ${max_lines} de ${total} líneas]"
      fi
      echo '```'
    } >> "$CONTEXT_MD"
  fi
}

append_csv_preview() {
  local title="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    {
      echo
      echo "### ${title}"
      echo
      echo "Archivo: \`${file}\`"
      echo
      echo '```csv'
      sed -n "1,${MAX_CSV_LINES}p" "$file" || true
      local total
      total="$(wc -l < "$file" 2>/dev/null || echo 0)"
      if [[ "$total" =~ ^[0-9]+$ ]] && (( total > MAX_CSV_LINES )); then
        echo
        echo "# [TRUNCADO: se muestran ${MAX_CSV_LINES} de ${total} líneas]"
      fi
      echo '```'
    } >> "$CONTEXT_MD"
  fi
}

safe_run_refresh() {
  if (( REFRESH == 1 )); then
    if [[ -f scripts/generate_phase34_figures.py ]]; then
      log "Regenerando figuras Fase 3.4B..."
      python scripts/generate_phase34_figures.py --output-root outputs || warn "No se pudo regenerar figuras. Continúo con archivos existentes."
    fi
    if [[ -f scripts/generate_phase34_report.py ]]; then
      log "Regenerando reporte Fase 3.4C/3.5..."
      python scripts/generate_phase34_report.py --output-root outputs || warn "No se pudo regenerar reporte. Continúo con archivos existentes."
    fi
  fi
}

safe_run_refresh

# -----------------------------------------------------------------------------
# Copia de archivos relevantes
# -----------------------------------------------------------------------------
copy_if_exists README.md "$FILES_DIR"
copy_if_exists pyproject.toml "$FILES_DIR"
copy_if_exists setup.cfg "$FILES_DIR"
copy_if_exists requirements.txt "$FILES_DIR"
copy_if_exists requirements-dev.txt "$FILES_DIR"
copy_if_exists .gitignore "$FILES_DIR"

copy_if_exists scripts/generate_phase34_figures.py "$FILES_DIR"
copy_if_exists scripts/generate_phase34_report.py "$FILES_DIR"
copy_if_exists tests/test_phase34_figures.py "$FILES_DIR"
copy_if_exists tests/test_phase34_report.py "$FILES_DIR"

copy_glob_if_exists "outputs/reports/phase34/*" "$FILES_DIR"
copy_glob_if_exists "outputs/figures/phase34/*.csv" "$FILES_DIR"
copy_glob_if_exists "outputs/figures/phase34/*.json" "$FILES_DIR"
if (( INCLUDE_FIGURES == 1 )); then
  copy_glob_if_exists "outputs/figures/phase34/*.png" "$FILES_DIR"
  copy_glob_if_exists "outputs/figures/phase34/*.pdf" "$FILES_DIR"
fi

copy_glob_if_exists "outputs/comparison/*.csv" "$FILES_DIR"
copy_glob_if_exists "outputs/comparison/*.json" "$FILES_DIR"
copy_glob_if_exists "outputs/comparison/*.md" "$FILES_DIR"
copy_glob_if_exists "outputs/comparison/*.txt" "$FILES_DIR"
copy_glob_if_exists "outputs/comparison/*.yaml" "$FILES_DIR"
copy_glob_if_exists "outputs/comparison/*.yml" "$FILES_DIR"

copy_glob_if_exists "outputs/**/ml_backend.json" "$FILES_DIR"
copy_glob_if_exists "outputs/**/metrics*.json" "$FILES_DIR"
copy_glob_if_exists "outputs/**/metadata*.json" "$FILES_DIR"
copy_glob_if_exists "outputs/**/predictions_validation.csv" "$FILES_DIR"

# Código fuente mínimo útil para trazabilidad metodológica.
copy_glob_if_exists "src/stohymolap/**/*.py" "$FILES_DIR"
copy_glob_if_exists "tests/*.py" "$FILES_DIR"

# Paper base, si existe en el proyecto.
if (( INCLUDE_PDF == 1 )); then
  if [[ -f "$REFERENCE_PAPER" ]]; then
    mkdir -p "$FILES_DIR/reference"
    cp -a "$REFERENCE_PAPER" "$FILES_DIR/reference/$(basename "$REFERENCE_PAPER")"
    echo "$REFERENCE_PAPER" >> "$MANIFEST"
  else
    warn "No encontré ${REFERENCE_PAPER}. Si lo quieres incluir, copia el PDF al repo o usa REFERENCE_PAPER=/ruta/paper.pdf."
  fi
fi

# -----------------------------------------------------------------------------
# Contexto Markdown principal
# -----------------------------------------------------------------------------
{
  echo "# Contexto de publicación — StoHyMoLAP"
  echo
  echo "Generado: ${TIMESTAMP}"
  echo "Proyecto: ${PROJECT_ROOT}"
  echo
  echo "## Objetivo del paquete"
  echo
  echo "Preparar un contexto autosuficiente para redactar las secciones de Resultados y Discusión de un manuscrito basado en StoHyMoLAP/RAMIS, usando los artefactos reproducibles de Fase 3.4B y Fase 3.4C/3.5."
  echo
  echo "## Regla crítica de redacción"
  echo
  echo "No afirmar que E7/E8 son GRU reales si el backend indica \`fallback_mlp_on_flattened_sequences\`. Deben describirse como configuraciones GRU-like/declaradas como GRU, ejecutadas con backend fallback por ausencia de TensorFlow/PyTorch, salvo que se reinstale un backend recurrente real y se regenere la evidencia."
  echo
  echo "## Relación con el paper base"
  echo
  echo "El artículo de Houénafa et al. (2025) propone la hibridación entre un modelo hidrológico estocástico y ML mediante propiedades estadísticas simuladas de la distribución diaria de caudal, incluyendo media y cuantiles. Su estructura útil para esta redacción combina: comparación de modelos físicos/ML/híbridos, tabla de métricas, hidrogramas y scatter, análisis por clases de caudal, sensibilidad a rangos de incertidumbre y discusión de la dependencia de desempeño respecto de las propiedades de distribución usadas como entrada."
  echo
  echo "## Pregunta científica sugerida"
  echo
  echo "¿En qué medida la incorporación de memoria hidrológica/baseflow, perturbaciones estocásticas tipo Lévy y predictores híbridos basados en salidas físicas mejora la simulación diaria de caudales frente a modelos físicos y ML puros bajo una ventana de validación común sin evidencia de leakage temporal?"
  echo
  echo "## Mensaje central preliminar"
  echo
  echo "El desempeño mejora de forma acumulativa cuando el modelo conserva estructura física e incorpora información adicional útil para ML. El mejor resultado común corresponde a E8_HYB_GRU_QUANTILES; sin embargo, la discusión debe matizarse porque el backend ejecutado es fallback MLP sobre secuencias aplanadas y no una GRU recurrente real."
  echo
  echo "## Artefactos clave esperados"
  echo
  echo "- \`outputs/reports/phase34/phase34_results_report.md\`: reporte narrativo y tablas base."
  echo "- \`outputs/reports/phase34/table_final_ranking_common_window.csv\`: ranking común E0-E8."
  echo "- \`outputs/reports/phase34/table_ablation_effects_common_window.csv\`: efectos de ablación."
  echo "- \`outputs/reports/phase34/table_regime_metrics_selected.csv\`: métricas por régimen."
  echo "- \`outputs/reports/phase34/table_backend_manuscript_notes.csv\`: notas de backend para redacción."
  echo "- \`outputs/figures/phase34/phase34_figure_manifest.csv\`: disponibilidad de figuras."
  echo "- \`outputs/figures/phase34/*.png\`: figuras paper-ready."
  echo
} > "$CONTEXT_MD"

append_cmd_raw "Git — rama actual" git branch --show-current
append_cmd_raw "Git — últimos commits" git log --oneline -8
append_cmd_raw "Git — estado corto" git status --short
append_cmd_raw "Árbol relevante del proyecto" bash -lc "find . -maxdepth 3 -type f | sed 's#^./##' | grep -Ev '(^.git/|__pycache__|\.pyc$|^outputs/.*/predictions_(train|validation)\.csv$)' | sort | head -300"

append_file_preview "README" README.md 250
append_file_preview "Reporte paper-ready Fase 3.4C/3.5" outputs/reports/phase34/phase34_results_report.md 500

append_csv_preview "Ranking final en ventana común" outputs/reports/phase34/table_final_ranking_common_window.csv
append_csv_preview "Modelos seleccionados para figuras" outputs/reports/phase34/table_selected_models_common_window.csv
append_csv_preview "Efectos de ablación" outputs/reports/phase34/table_ablation_effects_common_window.csv
append_csv_preview "Métricas por régimen" outputs/reports/phase34/table_regime_metrics_selected.csv
append_csv_preview "Referencias estocásticas/incertidumbre" outputs/reports/phase34/table_uncertainty_stochastic_references.csv
append_csv_preview "Resumen de estado de figuras" outputs/reports/phase34/table_figure_status_summary.csv
append_csv_preview "Notas de backend para manuscrito" outputs/reports/phase34/table_backend_manuscript_notes.csv
append_csv_preview "Checklist de reproducibilidad" outputs/reports/phase34/table_reproducibility_checklist.csv
append_csv_preview "Manifest de figuras" outputs/figures/phase34/phase34_figure_manifest.csv
append_file_preview "Manifest JSON de reporte" outputs/reports/phase34/phase34_report_manifest.json 250
append_file_preview "Manifest JSON de figuras" outputs/figures/phase34/phase34_figure_manifest.json 250

append_csv_preview "Leaderboard común original" outputs/comparison/leaderboard_common_intersection.csv
append_csv_preview "Leaderboard general" outputs/comparison/leaderboard.csv
append_file_preview "Auditoría anti-leakage" outputs/comparison/leakage_audit_report.md 300
append_file_preview "Estado auditoría anti-leakage" outputs/comparison/leakage_audit_status.txt 50
append_file_preview "Auditoría física" outputs/comparison/physical_audit_report.md 300
append_file_preview "Estado auditoría física" outputs/comparison/physical_audit_status.txt 50

{
  echo
  echo "## Columnas disponibles en predictions_validation.csv"
  echo
  echo '```text'
  python - <<'PY' 2>/dev/null || true
from pathlib import Path
import csv
paths = sorted(Path('outputs').glob('**/predictions_validation.csv'))
if not paths:
    print('No se encontraron predictions_validation.csv bajo outputs/.')
for p in paths:
    try:
        with p.open(newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
        print(f'{p}: {", ".join(header)}')
    except Exception as exc:
        print(f'{p}: ERROR leyendo cabecera: {exc}')
PY
  echo '```'
} >> "$CONTEXT_MD"

{
  echo
  echo "## Figuras disponibles"
  echo
  echo '```text'
  find outputs/figures/phase34 -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.csv' -o -name '*.json' \) 2>/dev/null | sort || true
  echo '```'
  echo
  echo "## Archivos copiados al paquete"
  echo
  echo '```text'
  sort "$MANIFEST" | uniq || true
  echo '```'
} >> "$CONTEXT_MD"

# -----------------------------------------------------------------------------
# Prompt para el siguiente chat
# -----------------------------------------------------------------------------
cat > "$PROMPT_MD" <<'PROMPT_EOF'
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
PROMPT_EOF

echo "$PROMPT_MD" >> "$MANIFEST"

# -----------------------------------------------------------------------------
# Plantilla LaTeX del paper
# -----------------------------------------------------------------------------
cat > "$TEX_FILE" <<'LATEX_EOF'
% StoHyMoLAP / RAMIS paper template
% Suggested target style: Elsevier elsarticle / Results in Engineering-like structure.
% Compile with: latexmk -pdf paper_stohymolap_template.tex

\documentclass[preprint,12pt]{elsarticle}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{lineno}

\journal{Results in Engineering}

% -----------------------------------------------------------------------------
% Macros
% -----------------------------------------------------------------------------
\newcommand{\Qobs}{Q_{\mathrm{obs}}}
\newcommand{\Qsim}{Q_{\mathrm{sim}}}
\newcommand{\NSE}{\mathrm{NSE}}
\newcommand{\KGE}{\mathrm{KGE}}
\newcommand{\RMSE}{\mathrm{RMSE}}
\newcommand{\MAE}{\mathrm{MAE}}
\newcommand{\PBIAS}{\mathrm{PBIAS}}
\newcommand{\warningbackend}{\textbf{Backend note:} E7/E8 must not be described as real recurrent GRU models unless TensorFlow/PyTorch evidence is regenerated. In the current artefacts they use a fallback MLP on flattened sequences.}

\begin{document}

\begin{frontmatter}

\title{Hybrid stochastic hydrological modelling with hydrological memory and machine learning for daily rainfall--runoff simulation}

\author[inst1]{Author One}
\author[inst2]{Author Two}
\address[inst1]{Institution, City, Country}
\address[inst2]{Institution, City, Country}

\begin{abstract}
% Draft after Results/Discussion are stable.
% Suggested content:
% 1) Problem: daily discharge simulation remains difficult under stochastic rainfall--runoff variability.
% 2) Aim: evaluate RAMIS/StoHyMoLAP-inspired deterministic, stochastic, ML, and hybrid configurations.
% 3) Methods: E0--E8 ablation matrix, common validation window, anti-leakage audit, metrics NSE/KGE/RMSE/MAE/PBIAS/R2.
% 4) Results: best common-window model and main ablation effects.
% 5) Limitation: GRU-labelled models currently executed with fallback MLP backend; stochastic bands show limited dispersion if applicable.
% 6) Implication: hybrid physical--data-driven inputs can improve point discharge prediction when carefully audited.
\end{abstract}

\begin{keyword}
Stochastic hydrological model \sep HyMoLAP \sep rainfall--runoff modelling \sep hybrid models \sep hydrological memory \sep machine learning \sep uncertainty
\end{keyword}

\end{frontmatter}

\linenumbers

% -----------------------------------------------------------------------------
\section{Introduction}
% Suggested structure:
% - Importance of accurate daily streamflow simulation.
% - Limitations of purely deterministic hydrological models under random fluctuations.
% - Limitations of standalone ML when physical memory/process information is weak.
% - Hybridization context: using hydrological model outputs/statistical properties as ML inputs.
% - Gap: reproducible ablation of RAMIS/StoHyMoLAP-inspired physical memory, Lévy stochasticity, and ML/hybrid post-processing.
% - Contributions:
%   (i) common-window E0--E8 experimental matrix;
%   (ii) baseflow/memory and stochastic ablation;
%   (iii) hybrid XGB/GRU-labelled quantile configurations;
%   (iv) anti-leakage and backend audit;
%   (v) paper-ready figures and reproducibility package.

% -----------------------------------------------------------------------------
\section{Materials and methods}

\subsection{Study area and hydro-meteorological data}
% Describe basin, period, forcing variables, observed discharge, temporal resolution.
% Include train/validation split and missing-data treatment.

\subsection{RAMIS / HyMoLAP-inspired deterministic rainfall--runoff formulation}
% Present deterministic core and effective precipitation logic.
% Keep equations consistent with the implemented model.

\subsection{Hydrological memory and baseflow reservoir}
% Explain the added baseflow/memory component.
% Link to E0 vs E1 and E2 vs E3 comparisons.

\subsection{Lévy-driven stochastic extension}
% Explain stochastic perturbation and intended uncertainty representation.
% State how stochastic summaries enter the experimental matrix.
% Avoid claiming well-calibrated uncertainty bands if manifest reports subdispersion.

\subsection{Machine learning and hybrid configurations}
% Describe E4 pure ML, E5/E6 hybrid XGB, E7/E8 GRU-labelled hybrids.
% Include backend note explicitly.
\warningbackend

\subsection{Experimental matrix}
\begin{table}[htbp]
\centering
\caption{Experimental matrix used for the common-window comparison. Replace values with the final table exported from the reproducibility package.}
\label{tab:experimental_matrix}
\begin{tabular}{llll}
\toprule
Experiment & Family & Main input & Purpose \\
\midrule
E0 & Physical & deterministic, no baseflow & baseline physical response \\
E1 & Physical & deterministic with baseflow & hydrological memory effect \\
E2 & Stochastic & Lévy, no baseflow & stochastic effect without memory \\
E3 & Stochastic & Lévy with baseflow & stochastic + memory effect \\
E4 & ML & pure XGB & data-driven baseline \\
E5 & Hybrid & mean physical/stochastic features + XGB & hybrid mean-feature effect \\
E6 & Hybrid & quantile features + XGB & hybrid uncertainty-feature effect \\
E7 & Hybrid & mean features + GRU-labelled fallback & sequence-labelled hybrid mean effect \\
E8 & Hybrid & quantile features + GRU-labelled fallback & sequence-labelled hybrid quantile effect \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Evaluation metrics and common validation window}
% Define NSE, KGE, RMSE, MAE, PBIAS, R2 if used.
% State that all comparisons use exact common intersection and missing Qobs are excluded.

\subsection{Leakage and reproducibility checks}
% Report anti-leakage audit status and interpretation.
% Mention tests passed and generated manifests.

% -----------------------------------------------------------------------------
\section{Results}

\subsection{Common-window model ranking}
% Use table_final_ranking_common_window.csv.
% Main claim: identify best model by NSE/KGE over n_eval common samples.

\begin{table}[htbp]
\centering
\caption{Final ranking over the common validation window. Convert the CSV table to LaTeX before submission.}
\label{tab:final_ranking_common}
\begin{tabular}{lrrrrrrl}
\toprule
Model & NSE & KGE & RMSE & MAE & PBIAS & R$^2$ & Backend \\
\midrule
% Fill from outputs/reports/phase34/table_final_ranking_common_window.csv
\bottomrule
\end{tabular}
\end{table}

\subsection{Ablation effects of baseflow, stochasticity and hybridization}
% Use table_ablation_effects_common_window.csv.
% Discuss deltas in NSE/KGE/RMSE/MAE/PBIAS.

\subsection{Hydrograph behaviour in validation}
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{outputs/figures/phase34/hydrograph_validation_common.png}
\caption{Observed and simulated validation hydrographs for selected physical, ML and hybrid configurations over the common validation window.}
\label{fig:hydrograph_common}
\end{figure}

\subsection{Flow-duration and scatter diagnostics}
\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{outputs/figures/phase34/fdc_common.png}
\caption{Flow-duration curves comparing observed and simulated discharges for the selected configurations.}
\label{fig:fdc_common}
\end{figure}

% Add selected scatter figures or a composite if later generated.

\subsection{Error structure by hydrological regime}
% Use residuals_by_regime_*.png and table_regime_metrics_selected.csv.
% Interpret low/medium/high flow behaviour.

\subsection{Stochastic reference and uncertainty-band limitations}
% Use stochastic_reference_E2_E3.png and uncertainty table.
% State fallback/subdispersion if reported by manifest.

% -----------------------------------------------------------------------------
\section{Discussion}

\subsection{Why hydrological memory improves daily discharge simulation}
% Interpret baseflow/reservoir as slow-flow persistence and catchment memory.
% Connect to deterministic_baseflow and stochastic_baseflow ablations.

\subsection{Role of stochastic Lévy perturbations}
% Discuss whether stochasticity improves point metrics or mainly provides distributional features.
% Avoid claiming uncertainty calibration unless coverage/bands support it.

\subsection{Hybrid physical--machine learning gains}
% Discuss ML as correction/mapping from physical/stochastic features to Qsim.
% Compare pure ML vs hybrid models.

\subsection{Quantile-informed features and regime-dependent performance}
% Discuss why distributional summaries can help high/low flows.
% Compare E6 and E8 carefully.

\subsection{Comparison with previous stochastic HyMoLAP hybridization studies}
% Compare conceptually with Houenafa et al. (2025): stochastic model outputs, mean/quantile features, physical + ML hybridization.
% Emphasize differences: RAMIS/baseflow, common-window ablation, fallback backend caveat, current dataset.

\subsection{Limitations}
% Suggested points:
% - E7/E8 backend fallback is not a true recurrent GRU implementation in current environment.
% - Component plots skipped if Qfast/Qbase/Qtotal columns are absent.
% - Stochastic bands may show subdispersion or unavailable quantile columns.
% - Results depend on the common validation period and selected basin/data quality.
% - WARN audits require transparent reporting even if no FAIL is found.

\subsection{Implications and future work}
% Suggested points:
% - Re-run E7/E8 with TensorFlow/PyTorch GRU backend.
% - Export physical components Qfast/Qbase/Qtotal.
% - Improve stochastic coverage and calibration of quantile bands.
% - Test transferability across basins and hydroclimatic regimes.

% -----------------------------------------------------------------------------
\section{Conclusions}
% 4--6 concise conclusions:
% - Best common-window model and magnitude of gain.
% - Baseflow/memory contribution.
% - Quantile-informed hybrid contribution.
% - Backend caveat.
% - Reproducibility and future work.

\section*{Code and data availability}
% Provide repository URL, commit hash, scripts, outputs and manifests.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Acknowledgements}
% Optional.

\bibliographystyle{elsarticle-num}
\bibliography{references}

\end{document}
LATEX_EOF

echo "$TEX_FILE" >> "$MANIFEST"

# -----------------------------------------------------------------------------
# Resumen final del paquete
# -----------------------------------------------------------------------------
{
  echo
  echo "## Estructura LaTeX generada"
  echo
  echo "Archivo: \`${TEX_FILE}\`"
  echo
  echo "La plantilla contiene secciones para Introduction, Materials and methods, Results, Discussion, Conclusions, Code/Data availability y referencias. Incluye placeholders para tablas y figuras de Fase 3.4B/3.4C."
  echo
  echo "## Prompt generado"
  echo
  echo "Archivo: \`${PROMPT_MD}\`"
} >> "$CONTEXT_MD"

if (( MAKE_TAR == 1 )); then
  mkdir -p "$BASE_OUT_DIR"
  tar -czf "$TAR_FILE" -C "$BASE_OUT_DIR" "$(basename "$OUT_DIR")"
fi

log "Contexto Markdown: $CONTEXT_MD"
log "Prompt redacción:    $PROMPT_MD"
log "Plantilla LaTeX:     $TEX_FILE"
log "Manifest:            $MANIFEST"
if (( MAKE_TAR == 1 )); then
  log "Paquete tar.gz:      $TAR_FILE"
fi

cat <<EOF

Listo. Para usarlo en otro chat, adjunta preferentemente:
  1) ${CONTEXT_MD}
  2) ${PROMPT_MD}
  3) ${TEX_FILE}
  4) ${TAR_FILE}

Comando recomendado:
  bash generate_context_publicacion.sh --refresh --with-figures --with-pdf

EOF
