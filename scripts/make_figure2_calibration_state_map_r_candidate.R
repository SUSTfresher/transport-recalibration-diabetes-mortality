#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE)
  repo_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
} else {
  repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}
.libPaths(c("D:/NHANES_R/library", .libPaths()))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(svglite)
  library(ragg)
})

table_dir <- file.path(repo_root, "outputs", "current_submission", "tables")
figure_dir <- file.path(repo_root, "outputs", "current_submission", "figures", "figure2_r_candidate")
source_dir <- file.path(repo_root, "outputs", "current_submission", "source_data")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

direction_order <- c(
  "NHANES -> MIMIC-IV",
  "MIMIC-IV ICU -> eICU",
  "eICU -> MIMIC-IV ICU"
)

direction_labels <- c(
  "NHANES -> MIMIC-IV" = "NHANES\n-> MIMIC-IV",
  "MIMIC-IV ICU -> eICU" = "MIMIC-IV ICU\n-> eICU",
  "eICU -> MIMIC-IV ICU" = "eICU\n-> MIMIC-IV ICU"
)

failure_labels <- c(
  "NHANES -> MIMIC-IV" = "Slope\ndistortion",
  "MIMIC-IV ICU -> eICU" = "Mixed\nfailure",
  "eICU -> MIMIC-IV ICU" = "Baseline-risk\ndisplacement"
)

colors <- c(
  "NHANES -> MIMIC-IV" = "#4E79A7",
  "MIMIC-IV ICU -> eICU" = "#59A14F",
  "eICU -> MIMIC-IV ICU" = "#B07AA1"
)

raw <- read_csv(
  file.path(table_dir, "Table_3_cross_scenario_transport_performance_numeric.csv"),
  show_col_types = FALSE
) |>
  filter(direction %in% direction_order) |>
  mutate(
    direction = factor(direction, levels = direction_order),
    y_direction = factor(direction, levels = rev(direction_order)),
    x_direction = factor(direction, levels = direction_order),
    direction_label = direction_labels[as.character(direction)]
  ) |>
  arrange(direction)

updates <- tibble(
  direction = factor(direction_order, levels = direction_order),
  selected_method = c("Platt", "Platt", "Intercept-only"),
  event_count = 100,
  updated_slope = c(1.051, 1.095, 1.055),
  updated_intercept = c(0.071, 0.135, 0.081),
  updated_ece = c(0.015710001940162, 0.0105225807750483, 0.0088830645630017),
  updated_ece_ci_lower = c(0.0095805232444662, 0.0068308612168831, 0.0080830207730887),
  updated_ece_ci_upper = c(0.0320765514310111, 0.0182724998740923, 0.0102711741571959)
)

plot_data <- raw |>
  left_join(updates, by = "direction") |>
  mutate(
    direction_chr = as.character(direction),
    failure_label = failure_labels[direction_chr],
    state_label = sprintf(
      "%s\n%.2f, %.2f",
      failure_label,
      calibration_slope_point,
      calibration_intercept_point
    ),
    update_label = paste0(selected_method, "\n100 events"),
    ece_label = sprintf("ECE %.3f\n-> %.3f", ece_10bin_point, updated_ece),
    raw_x = as.numeric(direction) - 0.045,
    update_x = as.numeric(direction) + 0.045
  )

source_rows <- bind_rows(
  plot_data |>
    transmute(
      panel = "2a",
      direction = as.character(direction),
      metric = "Transport AUC",
      method = "Raw transport",
      event_count = 0,
      value = roc_auc_point,
      ci_lower = roc_auc_ci_lower,
      ci_upper = roc_auc_ci_upper,
      notes = "Point and 95% bootstrap CI from Table 3."
    ),
  plot_data |>
    transmute(
      panel = "2b",
      direction = as.character(direction),
      metric = "ECE",
      method = "Raw transport",
      event_count = 0,
      value = ece_10bin_point,
      ci_lower = ece_10bin_ci_lower,
      ci_upper = ece_10bin_ci_upper,
      notes = "Point and 95% bootstrap CI from Table 3."
    ),
  plot_data |>
    transmute(
      panel = "2b",
      direction = as.character(direction),
      metric = "Brier score",
      method = "Raw transport",
      event_count = 0,
      value = brier_score_point,
      ci_lower = brier_score_ci_lower,
      ci_upper = brier_score_ci_upper,
      notes = "Point and 95% bootstrap CI from Table 3."
    ),
  plot_data |>
    transmute(
      panel = "2c",
      direction = as.character(direction),
      metric = "Calibration slope",
      method = "Raw transport",
      event_count = 0,
      value = calibration_slope_point,
      ci_lower = calibration_slope_ci_lower,
      ci_upper = calibration_slope_ci_upper,
      notes = "Calibration-state map coordinate from raw transported model."
    ),
  plot_data |>
    transmute(
      panel = "2c",
      direction = as.character(direction),
      metric = "Calibration intercept",
      method = "Raw transport",
      event_count = 0,
      value = calibration_intercept_point,
      ci_lower = calibration_intercept_ci_lower,
      ci_upper = calibration_intercept_ci_upper,
      notes = "Calibration-state map coordinate from raw transported model."
    ),
  plot_data |>
    transmute(
      panel = "2d",
      direction = as.character(direction),
      metric = "Calibration slope",
      method = "Raw transport",
      event_count = 0,
      value = calibration_slope_point,
      ci_lower = calibration_slope_ci_lower,
      ci_upper = calibration_slope_ci_upper,
      notes = "Raw transported slope before local recalibration."
    ),
  plot_data |>
    transmute(
      panel = "2d",
      direction = as.character(direction),
      metric = "Calibration slope",
      method = paste0(selected_method, " 100 events"),
      event_count = 100,
      value = updated_slope,
      ci_lower = NA_real_,
      ci_upper = NA_real_,
      notes = "Locked manuscript value for selected failure-matched 100-event updating."
    ),
  plot_data |>
    transmute(
      panel = "2d",
      direction = as.character(direction),
      metric = "ECE",
      method = paste0(selected_method, " 100 events"),
      event_count = 100,
      value = updated_ece,
      ci_lower = updated_ece_ci_lower,
      ci_upper = updated_ece_ci_upper,
      notes = "Empirical interval across 200 local calibration samples from Table 4."
    )
)

write_csv(source_rows, file.path(source_dir, "Figure_2_source_data_r_candidate.csv"))

theme_set(
  theme_classic(base_size = 7, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.text = element_text(colour = "black", size = 6.3),
      axis.title = element_text(colour = "black", size = 7),
      plot.title = element_text(size = 7.5, face = "plain", hjust = 0),
      plot.tag = element_text(size = 9, face = "bold"),
      legend.position = "none",
      legend.title = element_blank(),
      legend.text = element_text(size = 6.2),
      legend.key.size = grid::unit(3.3, "mm"),
      panel.grid.major = element_line(colour = "#E7E7E7", linewidth = 0.28),
      panel.grid.minor = element_blank(),
      plot.margin = margin(5.5, 6, 5.5, 5.5)
    )
)

p_a <- ggplot(plot_data, aes(x = roc_auc_point, y = y_direction, colour = direction)) +
  geom_segment(aes(x = roc_auc_ci_lower, xend = roc_auc_ci_upper, yend = y_direction), linewidth = 1.1) +
  geom_point(size = 2.4, stroke = 0.4) +
  geom_text(aes(label = sprintf("%.3f", roc_auc_point)), nudge_x = 0.018, size = 2.25, colour = "black") +
  scale_colour_manual(values = colors, guide = "none") +
  scale_y_discrete(labels = direction_labels) +
  scale_x_continuous(limits = c(0.64, 0.78), breaks = seq(0.64, 0.78, 0.02), expand = expansion(mult = c(0, 0))) +
  labs(title = "Discrimination was partially retained", x = "Transport AUC", y = NULL) +
  theme(panel.grid.major.y = element_blank(), axis.ticks.y = element_blank())

metric_data <- bind_rows(
  plot_data |>
    transmute(direction, direction_label, metric = "ECE", value = ece_10bin_point),
  plot_data |>
    transmute(direction, direction_label, metric = "Brier score", value = brier_score_point)
) |>
  mutate(metric = factor(metric, levels = c("ECE", "Brier score")))

p_b <- ggplot(metric_data, aes(x = direction, y = value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.64), width = 0.31, colour = "white", linewidth = 0.25) +
  geom_text(
    aes(label = sprintf("%.3f", value)),
    position = position_dodge(width = 0.64),
    vjust = -0.35,
    size = 2.15,
    colour = "black"
  ) +
  scale_fill_manual(values = c("ECE" = "#6B6F76", "Brier score" = "#A7BBC7")) +
  scale_x_discrete(labels = direction_labels) +
  scale_y_continuous(limits = c(0, 0.42), breaks = seq(0, 0.40, 0.05), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Probability-scale error remained large", x = NULL, y = "Metric value") +
  theme(
    panel.grid.major.x = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.03, 0.98),
    legend.justification = c(0, 1),
    legend.direction = "horizontal",
    legend.background = element_blank(),
    legend.margin = margin(0, 0, 0, 0)
  )

state_offsets <- tibble(
  direction = factor(direction_order, levels = direction_order),
  dx = c(0.03, 0.03, -0.04),
  dy = c(0.16, -0.03, 0.18),
  hjust = c(0, 0, 1)
)

state_data <- plot_data |>
  left_join(state_offsets, by = "direction")

p_c <- ggplot(state_data, aes(x = calibration_slope_point, y = calibration_intercept_point, colour = direction)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.38, colour = "#9AA0A6") +
  geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.38, colour = "#9AA0A6") +
  geom_point(size = 2.8) +
  annotate("point", x = 1, y = 0, shape = 8, size = 2.4, colour = "#3C4043") +
  annotate("text", x = 1.012, y = 0.02, label = "Ideal", hjust = 0, vjust = 0, size = 2.2, colour = "#3C4043") +
  geom_text(
    aes(
      x = calibration_slope_point + dx,
      y = calibration_intercept_point + dy,
      label = state_label,
      hjust = hjust
    ),
    size = 2.15,
    lineheight = 0.9,
    colour = "black",
    show.legend = FALSE
  ) +
  scale_colour_manual(values = colors, guide = "none") +
  scale_x_continuous(limits = c(0.36, 1.18), breaks = seq(0.4, 1.1, 0.1), expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(limits = c(-2.65, 0.18), breaks = seq(-2.5, 0, 0.5), expand = expansion(mult = c(0, 0))) +
  labs(title = "Calibration state separated failure modes", x = "Calibration slope", y = "Calibration intercept")

segment_data <- plot_data |>
  transmute(
    direction,
    raw_x,
    update_x,
    raw_slope = calibration_slope_point,
    updated_slope,
    update_label,
    ece_label
  )

p_d <- ggplot(segment_data) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.38, colour = "#9AA0A6") +
  geom_segment(
    aes(x = raw_x, xend = update_x, y = raw_slope, yend = updated_slope),
    linewidth = 1.1,
    colour = "#C4C7C5"
  ) +
  geom_point(aes(x = raw_x, y = raw_slope), size = 2.25, colour = "#C4C7C5") +
  geom_point(
    aes(x = update_x, y = updated_slope, colour = direction),
    size = 2.65,
    show.legend = FALSE
  ) +
  geom_text(aes(x = as.numeric(direction), y = 1.275, label = update_label), size = 2.05, lineheight = 0.9) +
  geom_text(aes(x = as.numeric(direction), y = 0.355, label = ece_label), size = 2.05, lineheight = 0.9, colour = "#3C4043") +
  scale_colour_manual(values = colors, guide = "none") +
  scale_x_continuous(
    limits = c(0.55, 3.45),
    breaks = seq_along(direction_order),
    labels = direction_labels,
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(limits = c(0.28, 1.34), breaks = seq(0.4, 1.2, 0.2), expand = expansion(mult = c(0, 0))) +
  labs(title = "Failure-matched updating repaired the failed component", x = NULL, y = "Calibration slope")

fig <- (p_a + p_b) / (p_c + p_d) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag.position = c(0, 1), plot.tag = element_text(size = 9, face = "bold"))

base <- file.path(figure_dir, "Figure_2_calibration_state_map_r_candidate")
width_in <- 7.2
height_in <- 5.35

svglite(paste0(base, ".svg"), width = width_in, height = height_in)
print(fig)
dev.off()

cairo_pdf(paste0(base, ".pdf"), width = width_in, height = height_in, family = "Arial")
print(fig)
dev.off()

agg_png(paste0(base, ".png"), width = width_in, height = height_in, units = "in", res = 300, background = "white")
print(fig)
dev.off()

agg_tiff(paste0(base, ".tiff"), width = width_in, height = height_in, units = "in", res = 600, background = "white")
print(fig)
dev.off()

message("Wrote R candidate figure outputs to: ", figure_dir)
message("Wrote R candidate source data to: ", file.path(source_dir, "Figure_2_source_data_r_candidate.csv"))
