from graphviz import Digraph

dot = Digraph(
    name="StoHyMoLAP_Software_Structure",
    format="png"
)
dot.attr(rankdir="LR", fontsize="12")
dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="10")
dot.attr("edge", fontname="Arial", fontsize="9")

# =========================
# Nodo principal
# =========================
dot.node("main", "main.py\nFlujo principal del modelo", fillcolor="#D6EAF8")

# =========================
# Configuración y datos
# =========================
dot.node("config", "config.py\nParámetros + bounds de Lévy\nlatitud, train_fraction", fillcolor="#E8F8F5")
dot.node(
    "data_io",
    "data_io.py\nCarga ramis_hydro.csv\nrellena flow_obs con flow_sim\ndevuelve series + fechas",
    fillcolor="#E8F8F5"
)
dot.node(
    "evapo",
    "evapotranspiration.py\nPET por Hargreaves-Samani\n(tmin, tmax, fecha, latitud)",
    fillcolor="#D1F2EB"
)
dot.node("diagnostics", "diagnostics.py\nEstadísticas iniciales", fillcolor="#E8F8F5")

# Fuente de datos
dot.node(
    "ramis_csv",
    "ramis_hydro.csv\ndate, flow_obs, flow_sim,\nprecipitation_mean, tmin, tmax",
    shape="cylinder",
    fillcolor="#D5F5E3"
)

# =========================
# Núcleo del modelo
# =========================
dot.node("model", "model.py\nModelo hidrológico", fillcolor="#FCF3CF")
dot.node("levy", "levy.py\nRuido estocástico de Lévy", fillcolor="#FCF3CF")
dot.node(
    "calibration",
    "calibration.py\nCalibración Monte Carlo\nμ, λ, σ + α, β Lévy\nselección top-K por NSE",
    fillcolor="#FADBD8"
)
dot.node("validation", "validation.py\nValidación del modelo", fillcolor="#FADBD8")
dot.node("metrics", "metrics.py\nNSE, KGE, RMSE, PBIAS", fillcolor="#FDEBD0")

# =========================
# Resultados
# =========================
dot.node("outputs", "outputs.py\nExportación de resultados", fillcolor="#EAECEE")
dot.node("plots", "plots.py\nHidrogramas con eje de fechas", fillcolor="#EAECEE")
dot.node(
    "outdir",
    "outputs_stohymolap_paper/\nCSV + figuras generadas",
    shape="folder",
    fillcolor="#D5DBDB"
)
dot.node("params", "calibration_best_parameters.csv\n(μ, λ, σ, α, β, NSE)", shape="note", fillcolor="#F2F3F4")
dot.node("series", "calibration_series.csv", shape="note", fillcolor="#F2F3F4")
dot.node("hydrograph", "calibration_hydrograph.png", shape="note", fillcolor="#F2F3F4")

# =========================
# Relaciones principales
# =========================
dot.edge("main", "config", label="lee configuración")
dot.edge("main", "data_io", label="carga datos + fechas")
dot.edge("main", "diagnostics", label="resume datos")
dot.edge("main", "calibration", label="ejecuta calibración")
dot.edge("main", "validation", label="valida con α/β calibrados")
dot.edge("main", "outputs", label="guarda CSV")
dot.edge("main", "plots", label="genera figuras")

# Datos y preprocesamiento
dot.edge("ramis_csv", "data_io", label="lee CSV")
dot.edge("data_io", "evapo", label="estima PET")

# Núcleo
dot.edge("calibration", "model", label="simula caudales")
dot.edge("calibration", "levy", label="muestrea α, β\ny genera trayectorias")
dot.edge("calibration", "metrics", label="evalúa NSE")
dot.edge("validation", "model", label="simula validación")
dot.edge("validation", "levy", label="incertidumbre")
dot.edge("validation", "metrics", label="evalúa desempeño")

# Salidas
dot.edge("outputs", "outdir")
dot.edge("plots", "outdir")
dot.edge("outdir", "params")
dot.edge("outdir", "series")
dot.edge("outdir", "hydrograph")

# =========================
# Render
# =========================
dot.render(
    filename="stohymolap_software_structure",
    cleanup=True
)
print("Diagrama generado: stohymolap_software_structure.png")