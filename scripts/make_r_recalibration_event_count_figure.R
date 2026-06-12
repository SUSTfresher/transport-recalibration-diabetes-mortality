.libPaths(c("D:/NHANES_R/library", .libPaths()))

library(ggplot2)
library(patchwork)
library(dplyr)
library(readr)
library(svglite)
library(ragg)

project_root <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
out_dir <- file.path(project_root, "outputs", "r_nature_style_figure_preview")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

table4b_path <- file.path(project_root, "outputs", "manuscript_tables", "Table_4b_recalibration_uncertainty_intervals.csv")

ink <- "#222222"
muted <- "#6f6f6f"
grid_col <- "#E7E7E7"
raw_col <- "#C87944"
platt_col <- "#7657A8"
iso_col <- "#6F9B7B"
int_col <- "#8B8B8B"

method_cols <- c(
  "Intercept-only" = int_col,
  "Isotonic" = iso_col,
  "Platt" = platt_col,
  "Raw transport" = raw_col
)

base_theme <- theme_classic(base_family = "Arial", base_size = 7.2) +
  theme(
    axis.line = element_line(linewidth = 0.35, colour = ink),
    axis.ticks = element_line(linewidth = 0.35, colour = ink),
    axis.text = element_text(colour = ink, size = 6.4),
    axis.title = element_text(colour = ink, size = 7.2),
    plot.title = element_text(face = "bold", colour = ink, size = 8.0, hjust = 0),
    plot.subtitle = element_text(colour = muted, size = 6.1, hjust = 0, margin = margin(t = 1, b = 3)),
    plot.tag = element_text(face = "bold", colour = ink, size = 8.0),
    plot.tag.position = c(0.0, 1.0),
    legend.position = "none",
    panel.grid.major = element_line(colour = grid_col, linewidth = 0.28),
    panel.grid.minor = element_blank(),
    plot.margin = margin(4, 8, 4, 5)
  )

raw <- read_csv(table4b_path, show_col_types = FALSE) %>%
  filter(
    prediction_set == "NHANES_to_MIMIC_1y_base_logistic_regression",
    method %in% c("raw", "intercept_only", "isotonic", "platt"),
    event_target %in% c(0, 25, 50, 100, 200)
  ) %>%
  mutate(
    method_label = recode(
      method,
      raw = "Raw transport",
      intercept_only = "Intercept-only",
      isotonic = "Isotonic",
      platt = "Platt"
    ),
    method_label = factor(method_label, levels = c("Raw transport", "Intercept-only", "Isotonic", "Platt"))
  )

event_df <- raw %>%
  filter(method != "raw") %>%
  mutate(event_target = as.numeric(event_target))

raw_row <- raw %>% filter(method == "raw") %>% slice(1)

event_breaks <- c(25, 50, 100, 200)

p_ece <- ggplot(event_df, aes(event_target, ece_10bin_mean, colour = method_label, fill = method_label)) +
  geom_hline(yintercept = raw_row$ece_10bin_mean, linewidth = 0.36, linetype = "22", colour = raw_col) +
  geom_ribbon(aes(ymin = ece_10bin_ci_lower, ymax = ece_10bin_ci_upper), alpha = 0.13, linewidth = 0) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.1) +
  annotate("text", x = 27, y = raw_row$ece_10bin_mean + 0.012, label = "Raw ECE 0.286", hjust = 0, size = 2.1, colour = raw_col, family = "Arial") +
  annotate("text", x = 116, y = 0.034, label = "Platt 100\nECE 0.016", hjust = 0, size = 2.0, colour = platt_col, family = "Arial", lineheight = 0.88) +
  scale_colour_manual(values = method_cols) +
  scale_fill_manual(values = method_cols) +
  scale_x_continuous(breaks = event_breaks, limits = c(20, 210), expand = expansion(mult = c(0.02, 0.05))) +
  scale_y_continuous(limits = c(0, 0.31), breaks = c(0, 0.05, 0.10, 0.20, 0.30), expand = expansion(mult = c(0.02, 0.02))) +
  labs(
    title = "ECE decreases",
    subtitle = "Empirical intervals across 200 resamples.",
    x = NULL,
    y = "ECE",
    tag = "a"
  ) +
  base_theme

p_slope <- ggplot(event_df, aes(event_target, calibration_slope_mean, colour = method_label, fill = method_label)) +
  geom_hline(yintercept = 1, linewidth = 0.36, linetype = "22", colour = ink) +
  geom_hline(yintercept = raw_row$calibration_slope_mean, linewidth = 0.30, linetype = "22", colour = raw_col) +
  geom_ribbon(aes(ymin = calibration_slope_ci_lower, ymax = calibration_slope_ci_upper), alpha = 0.12, linewidth = 0) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.1) +
  annotate("text", x = 27, y = raw_row$calibration_slope_mean + 0.10, label = "Raw slope 0.45", hjust = 0, size = 2.1, colour = raw_col, family = "Arial") +
  annotate("text", x = 103, y = 1.22, label = "ideal slope", hjust = 0, size = 2.0, colour = ink, family = "Arial") +
  scale_colour_manual(values = method_cols) +
  scale_fill_manual(values = method_cols) +
  scale_x_continuous(breaks = event_breaks, limits = c(20, 210), expand = expansion(mult = c(0.02, 0.05))) +
  coord_cartesian(ylim = c(0.15, 1.70), clip = "on") +
  labs(
    title = "Slope recovers",
    subtitle = "Intercept-only leaves the slope unchanged.",
    x = NULL,
    y = "Calibration slope",
    tag = "b"
  ) +
  base_theme

p_intercept <- ggplot(event_df, aes(event_target, calibration_intercept_mean, colour = method_label, fill = method_label)) +
  geom_hline(yintercept = 0, linewidth = 0.36, linetype = "22", colour = ink) +
  geom_hline(yintercept = raw_row$calibration_intercept_mean, linewidth = 0.30, linetype = "22", colour = raw_col) +
  geom_ribbon(aes(ymin = calibration_intercept_ci_lower, ymax = calibration_intercept_ci_upper), alpha = 0.12, linewidth = 0) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.1) +
  annotate("text", x = 27, y = raw_row$calibration_intercept_mean + 0.14, label = "Raw intercept -1.51", hjust = 0, size = 2.1, colour = raw_col, family = "Arial") +
  annotate("text", x = 103, y = 0.18, label = "ideal intercept", hjust = 0, size = 2.0, colour = ink, family = "Arial") +
  scale_colour_manual(values = method_cols) +
  scale_fill_manual(values = method_cols) +
  scale_x_continuous(breaks = event_breaks, limits = c(20, 210), expand = expansion(mult = c(0.02, 0.05))) +
  coord_cartesian(ylim = c(-1.65, 0.70), clip = "on") +
  labs(
    title = "Intercept normalizes",
    subtitle = "Platt 100-200 intervals include the ideal intercept.",
    x = "Local outcome events used for recalibration",
    y = "Calibration intercept",
    tag = "c"
  ) +
  base_theme

legend_df <- tibble(
  method_label = factor(c("Intercept-only", "Isotonic", "Platt", "Raw transport"), levels = names(method_cols)),
  x = c(0.06, 0.29, 0.46, 0.61),
  y = 0.45
)

legend_plot <- ggplot(legend_df, aes(x, y, colour = method_label)) +
  geom_segment(aes(xend = x + 0.055, yend = y), linewidth = 1.2) +
  geom_point(aes(x = x + 0.027), size = 2.0) +
  geom_text(aes(x = x + 0.070, label = method_label), hjust = 0, size = 2.05, family = "Arial", colour = ink) +
  scale_colour_manual(values = method_cols) +
  xlim(0, 1) +
  ylim(0, 1) +
  theme_void() +
  theme(legend.position = "none", plot.margin = margin(0, 4, 0, 4))

title_block <- ggplot() +
  annotate(
    "text",
    x = 0,
    y = 0.72,
    hjust = 0,
    label = "Local outcome events required for transport recalibration",
    family = "Arial",
    fontface = "bold",
    size = 3.9,
    colour = ink
  ) +
  annotate(
    "text",
    x = 0,
    y = 0.30,
    hjust = 0,
    label = "Across 200 resamples, Platt recalibration with 100-200 events restores both slope and intercept;\nintercept-only remains a negative control and isotonic remains unstable at small sample sizes.",
    family = "Arial",
    size = 2.12,
    colour = muted
  ) +
  xlim(0, 1) +
  ylim(0, 1) +
  theme_void() +
  theme(plot.margin = margin(0, 5, 0, 5))

figure <- title_block / legend_plot / (p_ece | p_slope | p_intercept) +
  plot_layout(heights = c(0.18, 0.08, 1), widths = c(1, 1, 1)) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

png_path <- file.path(out_dir, "r_recalibration_event_count_preview.png")
svg_path <- file.path(out_dir, "r_recalibration_event_count_preview.svg")
pdf_path <- file.path(out_dir, "r_recalibration_event_count_preview.pdf")

ragg::agg_png(png_path, width = 7.60, height = 4.20, units = "in", res = 450, background = "white")
print(figure)
dev.off()

svglite::svglite(svg_path, width = 7.60, height = 4.20, bg = "white")
print(figure)
dev.off()

cairo_pdf(pdf_path, width = 7.60, height = 4.20, bg = "white")
print(figure)
dev.off()

message("Wrote R event-count figure preview to: ", out_dir)
