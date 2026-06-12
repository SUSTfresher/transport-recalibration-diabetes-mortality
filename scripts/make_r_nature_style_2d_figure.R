.libPaths(c("D:/NHANES_R/library", .libPaths()))

library(ggplot2)
library(patchwork)
library(dplyr)
library(readr)
library(ggrepel)
library(grid)
library(svglite)
library(ragg)

project_root <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
out_dir <- file.path(project_root, "outputs", "r_nature_style_figure_preview")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

table4b_path <- file.path(project_root, "outputs", "manuscript_tables", "Table_4b_recalibration_uncertainty_intervals.csv")
dca_path <- file.path(project_root, "outputs", "decision_curve", "decision_curve_transportability.csv")

ink <- "#222222"
muted <- "#6f6f6f"
grid_col <- "#E7E7E7"
raw_col <- "#C87944"
platt_col <- "#7657A8"
iso_col <- "#6F9B7B"
int_col <- "#8B8B8B"
mimic_col <- "#5B987A"

method_cols <- c(
  "Raw transport" = raw_col,
  "Intercept-only" = int_col,
  "Isotonic" = iso_col,
  "Platt" = platt_col,
  "Platt 100 events" = platt_col,
  "MIMIC internal" = mimic_col,
  "Treat all" = "#BDBDBD",
  "Treat none" = "#555555"
)

base_theme <- theme_classic(base_family = "Arial", base_size = 7.2) +
  theme(
    axis.line = element_line(linewidth = 0.35, colour = ink),
    axis.ticks = element_line(linewidth = 0.35, colour = ink),
    axis.text = element_text(colour = ink, size = 6.5),
    axis.title = element_text(colour = ink, size = 7.2),
    plot.title = element_text(face = "bold", colour = ink, size = 8.1, hjust = 0),
    plot.subtitle = element_text(colour = muted, size = 6.2, hjust = 0, margin = margin(t = 1, b = 3)),
    plot.tag = element_text(face = "bold", colour = ink, size = 8.2),
    plot.tag.position = c(0.0, 1.0),
    legend.position = "none",
    panel.grid.major = element_line(colour = grid_col, linewidth = 0.28),
    panel.grid.minor = element_blank(),
    plot.margin = margin(5, 8, 5, 5)
  )

state <- read_csv(table4b_path, show_col_types = FALSE) %>%
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

state_labels <- bind_rows(
  state %>%
    filter(method == "raw") %>%
    mutate(label = "Raw transport\nslope 0.45\nintercept -1.51", label_x = calibration_slope_mean + 0.06, label_y = calibration_intercept_mean + 0.18),
  state %>%
    filter(method == "intercept_only", event_target == 200) %>%
    mutate(label = "Intercept-only", label_x = calibration_slope_mean + 0.04, label_y = calibration_intercept_mean - 0.02),
  state %>%
    filter(method == "isotonic", event_target == 200) %>%
    mutate(label = "Isotonic 200", label_x = calibration_slope_mean + 0.04, label_y = calibration_intercept_mean + 0.02),
  state %>%
    filter(method == "platt", event_target == 100) %>%
    mutate(label = "Platt 100", label_x = calibration_slope_mean + 0.075, label_y = calibration_intercept_mean + 0.12),
  state %>%
    filter(method == "platt", event_target == 200) %>%
    mutate(label = "Platt 200", label_x = calibration_slope_mean + 0.075, label_y = calibration_intercept_mean - 0.09)
)

ideal <- tibble(
  calibration_slope_mean = 1,
  calibration_intercept_mean = 0,
  label = "Ideal",
  label_x = 0.82,
  label_y = 0.14
)

p_state <- ggplot() +
  geom_hline(yintercept = 0, linetype = "22", linewidth = 0.36, colour = "#444444") +
  geom_vline(xintercept = 1, linetype = "22", linewidth = 0.36, colour = "#444444") +
  geom_path(
    data = state %>% filter(method != "raw"),
    aes(calibration_slope_mean, calibration_intercept_mean, group = method_label, colour = method_label),
    linewidth = 1.0,
    alpha = 0.90
  ) +
  geom_point(
    data = state %>% filter(method != "raw"),
    aes(calibration_slope_mean, calibration_intercept_mean, colour = method_label),
    size = 2.2,
    stroke = 0.35
  ) +
  geom_point(
    data = state %>% filter(method == "raw"),
    aes(calibration_slope_mean, calibration_intercept_mean),
    size = 4.2,
    colour = raw_col,
    alpha = 0.96
  ) +
  geom_point(
    data = ideal,
    aes(calibration_slope_mean, calibration_intercept_mean),
    shape = 8,
    size = 4.4,
    colour = ink,
    stroke = 1.0
  ) +
  geom_text(
    data = state_labels,
    aes(label_x, label_y, label = label, colour = method_label),
    size = 2.12,
    lineheight = 0.86,
    hjust = 0,
    vjust = 0.5,
    family = "Arial"
  ) +
  geom_text(
    data = ideal,
    aes(label_x, label_y, label = label),
    size = 2.1,
    hjust = 0,
    colour = ink,
    family = "Arial"
  ) +
  scale_colour_manual(values = method_cols) +
  scale_x_continuous(limits = c(0.20, 1.42), breaks = c(0.25, 0.50, 0.75, 1.00, 1.25), expand = expansion(mult = c(0.02, 0.04))) +
  scale_y_continuous(limits = c(-1.62, 0.56), breaks = seq(-1.5, 0.5, 0.5), expand = expansion(mult = c(0.02, 0.03))) +
  labs(
    title = "Calibration state map",
    subtitle = "Raw transport sits far from the ideal slope/intercept point.",
    x = "Calibration slope",
    y = "Calibration intercept",
    tag = "a"
  ) +
  base_theme

interval <- state %>%
  filter(method %in% c("intercept_only", "platt"), event_target %in% c(25, 50, 100, 200)) %>%
  mutate(
    family = recode(method, intercept_only = "Intercept-only", platt = "Platt"),
    family = factor(family, levels = c("Intercept-only", "Platt")),
    y = case_when(
      method == "intercept_only" ~ 9 - match(event_target, c(25, 50, 100, 200)),
      method == "platt" ~ 4 - match(event_target, c(25, 50, 100, 200))
    ),
    event_label = as.character(event_target)
  )

family_labels <- tibble(
  x = c(0.33, 0.33),
  y = c(8.45, 3.45),
  label = c("Intercept-only", "Platt"),
  family = factor(c("Intercept-only", "Platt"), levels = c("Intercept-only", "Platt"))
)

p_interval <- ggplot(interval, aes(y = y, colour = family)) +
  geom_vline(xintercept = 1, linetype = "22", linewidth = 0.38, colour = ink) +
  geom_segment(aes(x = calibration_slope_ci_lower, xend = calibration_slope_ci_upper, yend = y), linewidth = 1.0, alpha = 0.88) +
  geom_point(aes(x = calibration_slope_mean), size = 2.3) +
  geom_text(data = family_labels, aes(x = x, y = y, label = label, colour = family), inherit.aes = FALSE, hjust = 0, size = 2.25, fontface = "bold", family = "Arial") +
  scale_colour_manual(values = c("Intercept-only" = int_col, "Platt" = platt_col)) +
  scale_x_continuous(breaks = c(0.5, 1.0, 1.5, 2.0), expand = expansion(mult = c(0.02, 0.03))) +
  scale_y_continuous(
    breaks = c(8, 7, 6, 5, 3, 2, 1, 0),
    labels = c("25", "50", "100", "200", "25", "50", "100", "200"),
    expand = c(0, 0)
  ) +
  coord_cartesian(xlim = c(0.30, 2.05), ylim = c(-0.65, 8.75), clip = "on") +
  labs(
    title = "Slope recovery",
    subtitle = "Platt intervals cross the ideal slope by 100-200 local events.",
    x = "Calibration slope",
    y = "Local events",
    tag = "b"
  ) +
  base_theme +
  theme(panel.grid.major.y = element_blank())

dca <- read_csv(dca_path, show_col_types = FALSE) %>%
  filter(model %in% c("Treat none", "Treat all", "NHANES raw logistic", "NHANES Platt 100 events", "MIMIC internal HGB")) %>%
  mutate(
    model_label = recode(
      model,
      `Treat none` = "Treat none",
      `Treat all` = "Treat all",
      `NHANES raw logistic` = "Raw transport",
      `NHANES Platt 100 events` = "Platt 100 events",
      `MIMIC internal HGB` = "MIMIC internal"
    ),
    model_label = factor(model_label, levels = c("Treat none", "Treat all", "Raw transport", "Platt 100 events", "MIMIC internal"))
  )

dca_labels <- bind_rows(
  dca %>% filter(model_label %in% c("Raw transport", "Platt 100 events", "MIMIC internal"), threshold >= 0.30) %>% group_by(model_label) %>% slice(1) %>% ungroup() %>% mutate(label_x = threshold + 0.006, label_y = net_benefit),
  tibble(model_label = factor(c("Treat all", "Treat none"), levels = levels(dca$model_label)), label_x = c(0.095, 0.205), label_y = c(0.065, 0.006), net_benefit = c(NA, NA), threshold = c(NA, NA))
) %>%
  mutate(label = as.character(model_label))

p_dca <- ggplot(dca, aes(threshold, net_benefit, colour = model_label, linetype = model_label)) +
  geom_hline(yintercept = 0, linewidth = 0.35, colour = ink) +
  geom_line(linewidth = 1.05, alpha = 0.95) +
  geom_text(
    data = dca_labels,
    aes(label_x, label_y, label = label, colour = model_label),
    inherit.aes = FALSE,
    size = 2.08,
    hjust = 0,
    family = "Arial"
  ) +
  scale_colour_manual(values = method_cols) +
  scale_linetype_manual(values = c("Treat none" = "dotted", "Treat all" = "dashed", "Raw transport" = "solid", "Platt 100 events" = "solid", "MIMIC internal" = "solid")) +
  scale_x_continuous(breaks = c(0.1, 0.2, 0.3), expand = expansion(mult = c(0.01, 0.08))) +
  scale_y_continuous(breaks = c(-0.05, 0, 0.05, 0.10), expand = expansion(mult = c(0.02, 0.02))) +
  coord_cartesian(xlim = c(0.05, 0.35), ylim = c(-0.07, 0.14), clip = "on") +
  labs(
    title = "Clinical utility",
    subtitle = "Recalibrated transport retains positive net benefit at clinical thresholds.",
    x = "Risk threshold",
    y = "Net benefit",
    tag = "c"
  ) +
  base_theme

main_title <- "Local recalibration restores transported mortality-risk calibration"
main_subtitle <- paste(strwrap(
  "Raw NHANES-to-MIMIC transport showed slope 0.45 and intercept -1.51; Platt recalibration with 100-200 local events restored slope/intercept and preserved positive net benefit.",
  width = 138
), collapse = "\n")

title_block <- ggplot() +
  annotate("text", x = 0, y = 0.70, hjust = 0, label = main_title, family = "Arial", fontface = "bold", size = 3.95, colour = ink) +
  annotate("text", x = 0, y = 0.28, hjust = 0, label = main_subtitle, family = "Arial", size = 2.18, colour = muted, lineheight = 0.95) +
  xlim(0, 1) +
  ylim(0, 1) +
  theme_void() +
  theme(plot.margin = margin(0, 5, 2, 5))

figure <- title_block / (p_state | (p_interval / p_dca)) +
  plot_layout(heights = c(0.17, 1), widths = c(1.05, 1.20)) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

png_path <- file.path(out_dir, "r_nature_style_2d_preview.png")
svg_path <- file.path(out_dir, "r_nature_style_2d_preview.svg")
pdf_path <- file.path(out_dir, "r_nature_style_2d_preview.pdf")

ragg::agg_png(png_path, width = 7.35, height = 5.45, units = "in", res = 450, background = "white")
print(figure)
dev.off()

svglite::svglite(svg_path, width = 7.35, height = 5.45, bg = "white")
print(figure)
dev.off()

cairo_pdf(pdf_path, width = 7.35, height = 5.45, bg = "white")
print(figure)
dev.off()

message("Wrote R figure preview to: ", out_dir)
