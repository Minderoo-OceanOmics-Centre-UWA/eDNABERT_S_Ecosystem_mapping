library(tidyverse)
library(patchwork)
library(akima)

meta <- readxl::read_excel(
  './asv_final_faire_metadata.xlsx',
  sheet = 'sampleMetadata',
  skip = 2
)
tsne <- read_csv('./tsne_12S.csv')
umap <- read_csv('./umap_12S.csv')
create_scatter_plot <- function(data, x_col, y_col, color_var, title) {
  data %>%
    ggplot(aes(
      x = .data[[x_col]],
      y = .data[[y_col]],
      color = .data[[color_var]]
    )) +
    geom_point(size = 2) +
    ggtitle(title) +
    theme_bw()
}
create_interpolated_plot <- function(data, x_col, y_col, z_var, title) {
  # Filter out NA values for interpolation
  plot_data <- data %>% filter(!is.na(.data[[z_var]]))

  # Interpolate using direct column access
  interp_result <- interp(
    plot_data[[x_col]],
    plot_data[[y_col]],
    plot_data[[z_var]],
    nx = 50,
    ny = 50
  )

  # Create interpolation dataframe
  interp_df <- expand.grid(x = interp_result$x, y = interp_result$y) %>%
    mutate(z = as.vector(interp_result$z))

  # Create plot
  ggplot() +
    geom_raster(data = interp_df, aes(x = x, y = y, fill = z), alpha = 0.7) +
    scale_fill_viridis_c(name = str_to_title(z_var), na.value = "transparent") +
    geom_point(
      data = plot_data,
      aes(x = .data[[x_col]], y = .data[[y_col]], color = .data[[z_var]]),
      size = 4
    ) +
    ggtitle(title) +
    theme_bw()
}


tsne_data <- tsne %>% left_join(meta, by = c('site_id' = 'samp_name'))
umap_data <- umap %>% left_join(meta, by = c('site_id' = 'samp_name'))

tsne_lat <- create_scatter_plot(
  tsne_data,
  "tsne_x",
  "tsne_y",
  "decimalLatitude",
  "tSNE - Latitude"
)
tsne_lon <- create_scatter_plot(
  tsne_data,
  "tsne_x",
  "tsne_y",
  "decimalLongitude",
  "tSNE - Longitude"
)

umap_clustered <- umap_data %>%
  mutate(
    cluster = as.factor(kmeans(cbind(umap_x, umap_y), centers = 5)$cluster)
  )

umap_lat <- umap_clustered %>%
  ggplot(aes(x = umap_x, y = umap_y, color = decimalLatitude)) +
  geom_point(size = 2) +
  stat_ellipse(
    aes(group = cluster),
    type = "norm",
    linetype = 2,
    color = "black",
    alpha = 0.7
  ) +
  ggtitle("UMAP - Latitude") +
  theme_bw()

umap_lon <- umap_clustered %>%
  ggplot(aes(x = umap_x, y = umap_y, color = decimalLongitude)) +
  geom_point(size = 2) +
  stat_ellipse(
    aes(group = cluster),
    type = "norm",
    linetype = 2,
    color = "black",
    alpha = 0.7
  ) +
  ggtitle("UMAP - Longitude") +
  theme_bw()

tsne_lat_interp <- create_interpolated_plot(
  tsne_data,
  "tsne_x",
  "tsne_y",
  "decimalLatitude",
  "tSNE - Latitude (Interpolated)"
)
tsne_lon_interp <- create_interpolated_plot(
  tsne_data,
  "tsne_x",
  "tsne_y",
  "decimalLongitude",
  "tSNE - Longitude (Interpolated)"
)
umap_lat_interp <- create_interpolated_plot(
  umap_data,
  "umap_x",
  "umap_y",
  "decimalLatitude",
  "UMAP - Latitude (Interpolated)"
)
umap_lon_interp <- create_interpolated_plot(
  umap_data,
  "umap_x",
  "umap_y",
  "decimalLongitude",
  "UMAP - Longitude (Interpolated)"
)

(tsne_lat / tsne_lon) | (umap_lat / umap_lon)

(tsne_lat_interp / tsne_lon_interp) | (umap_lat_interp / umap_lon_interp)


models <- list(
  "UMAP X ~ Lat + Lon" = lm(
    umap_x ~ decimalLatitude + decimalLongitude,
    data = umap_data
  ),
  "UMAP Y ~ Lat + Lon" = lm(
    umap_y ~ decimalLatitude + decimalLongitude,
    data = umap_data
  ),
  "tSNE X ~ Lat + Lon" = lm(
    tsne_x ~ decimalLatitude + decimalLongitude,
    data = tsne_data
  ),
  "tSNE Y ~ Lat + Lon" = lm(
    tsne_y ~ decimalLatitude + decimalLongitude,
    data = tsne_data
  )
)

iwalk(
  models,
  ~ {
    cat(.y, "\n")
    print(summary(.x))
    cat("\n", rep("-", 50), "\n\n")
  }
)

