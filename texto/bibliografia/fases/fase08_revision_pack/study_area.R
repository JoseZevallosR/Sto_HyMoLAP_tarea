# Exporta la cuenca delimitada como PNG 300 DPI
# Requiere: httr, sf, ggplot2, ggspatial, tidyverse

library(httr)
library(sf)
library(ggplot2)
library(ggspatial)  # annotation_map_tile() + scale bar + north arrow

# --------------------------------------------------
# 1. Llamada a la API
# --------------------------------------------------
pour_lat <- -15.2550
pour_lon <- -69.8730

resp <- POST(
  "https://fluviotech.com/api/watersheds/delineate/",
  body  = list(lat = pour_lat, lon = pour_lon),
  encode = "json"
)
stop_for_status(resp)
data <- content(resp, as = "parsed", type = "application/json")

# --------------------------------------------------
# 2. Convertir GeoJSON a objetos sf
# --------------------------------------------------
catchment_json    <- jsonlite::toJSON(data$catchment,    auto_unbox = TRUE)
river_json        <- jsonlite::toJSON(data$river_network, auto_unbox = TRUE)

sf_cuenca <- st_read(catchment_json,    quiet = TRUE)
sf_rios   <- st_read(river_json,        quiet = TRUE)
sf_pour   <- st_sfc(st_point(c(pour_lon, pour_lat)), crs = 4326) |>
  st_sf(label = paste0("(", pour_lat, "°, ", pour_lon, "°)"))

# --------------------------------------------------
# 3. Figura ggplot2 con basemap
# --------------------------------------------------
p <- ggplot() +
  # Basemap OpenStreetMap
  annotation_map_tile(type = "osm", zoom = 10, alpha = 0.7) +
  
  # Cuenca
  geom_sf(data  = sf_cuenca,
          fill  = "#2196F3", colour = "#0D47A1",
          linewidth = 0.4, alpha = 0.35) +
  
  # Red de drenaje
  geom_sf(data   = sf_rios,
          colour = "#00BCD4", linewidth = 0.5, alpha = 0.9) +
  
  # Punto de pour
  geom_sf(data   = sf_pour,
          shape  = 25,          # triángulo relleno invertido
          fill   = "red", colour = "darkred",
          size   = 3) +
  
  # Escala + norte
  annotation_scale(location = "bl", width_hint = 0.25, text_cex = 0.6) +
  annotation_north_arrow(location = "tr", which_north = "true",
                         style = north_arrow_fancy_orienteering(
                           fill = c("grey40", "white"), text_size = 8)) +
  
  # Etiquetas
  labs(
    title    = "Cuenca hidrográfica delimitada",
    subtitle = paste0("Punto de pour: (", pour_lat, "°, ", pour_lon, "°)"),
    x        = "Longitud (°O)",
    y        = "Latitud (°S)",
    #caption  = "Fuente: FluvioTech API · Basemap: OpenStreetMap"
  ) +
  
  # Leyenda manual
  scale_fill_identity(
    name   = NULL,
    guide  = guide_legend(),
    labels = "Cuenca delimitada"
  ) +
  
  theme_bw(base_size = 10) +
  theme(
    plot.title    = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 8, colour = "grey40"),
    axis.text     = element_text(size = 7),
    legend.position = "bottom"
  )

# --------------------------------------------------
# 4. Guardar a 300 DPI
# --------------------------------------------------
ggsave(
  filename = "cuenca_delimitada_300dpi.png",
  plot     = p,
  dpi      = 300,
  width    = 8,
  height   = 8,
  units    = "in",
  bg       = "white"
)
cat("✅ Imagen guardada: cuenca_delimitada_300dpi.png\n")