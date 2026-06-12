.libPaths(c("D:/NHANES_R/library", .libPaths()))

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(grid)
  library(svglite)
  library(ragg)
})

project_root <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
out_dir <- file.path(project_root, "outputs", "npj_main_figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

path <- function(...) file.path(project_root, ...)

palette <- c(
  ink = "#222222",
  muted = "#6E6E6E",
  grid = "#E6E6E6",
  nhanes = "#4C78A8",
  mimic = "#5B987A",
  raw = "#C87944",
  platt = "#7657A8",
  isotonic = "#6F9B7B",
  intercept = "#8B8B8B",
  internal = "#2F6F4E",
  accent = "#D24B40"
)

col <- function(name) unname(palette[name])

method_cols <- c(
  "Raw transport" = col("raw"),
  "Intercept-only" = col("intercept"),
  "Isotonic" = col("isotonic"),
  "Platt" = col("platt"),
  "Platt 100 events" = col("platt"),
  "MIMIC internal" = col("internal"),
  "Treat all" = "#BDBDBD",
  "Treat none" = "#555555"
)

source_cols <- c("NHANES" = col("nhanes"), "MIMIC-IV" = col("mimic"))

theme_npj <- function(base_size = 6.6, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = col("ink")),
      axis.ticks = element_line(linewidth = 0.32, colour = col("ink")),
      axis.text = element_text(colour = col("ink"), size = base_size - 0.45),
      axis.title = element_text(colour = col("ink"), size = base_size),
      plot.title = element_text(face = "bold", colour = col("ink"), size = base_size + 0.8, hjust = 0),
      plot.subtitle = element_text(colour = col("muted"), size = base_size - 0.55, hjust = 0, margin = margin(t = 1, b = 3)),
      plot.tag = element_text(face = "bold", colour = col("ink"), size = 8.2),
      plot.tag.position = c(0, 1),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.7, colour = col("ink")),
      legend.key.height = unit(3.5, "mm"),
      legend.key.width = unit(5, "mm"),
      panel.grid.major = element_line(colour = col("grid"), linewidth = 0.25),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 6, 4, 5)
    )
}

theme_set(theme_npj())

save_npj <- function(plot, stem, width_mm = 183, height_mm = 120, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4

  svglite::svglite(file.path(out_dir, paste0(stem, ".svg")), width = w, height = h, bg = "white")
  print(plot)
  dev.off()

  grDevices::cairo_pdf(file.path(out_dir, paste0(stem, ".pdf")), width = w, height = h, family = "Arial", bg = "white")
  print(plot)
  dev.off()

  ragg::agg_tiff(file.path(out_dir, paste0(stem, ".tiff")), width = w, height = h, units = "in", res = dpi, background = "white", compression = "lzw")
  print(plot)
  dev.off()

  ragg::agg_png(file.path(out_dir, paste0(stem, ".png")), width = w, height = h, units = "in", res = 300, background = "white")
  print(plot)
  dev.off()
}

extract_mid <- function(x) {
  as.numeric(str_match(x, "^(-?\\d+\\.?\\d*)")[, 2])
}

extract_low <- function(x) {
  as.numeric(str_match(x, "\\((-?\\d+\\.?\\d*)-")[, 2])
}

extract_high <- function(x) {
  as.numeric(str_match(x, "-(-?\\d+\\.?\\d*)\\)")[, 2])
}

percent_lab <- function(x, accuracy = 1) {
  paste0(round(x * 100 / accuracy) * accuracy, "%")
}

title_block <- function(title, subtitle) {
  ggplot() +
    annotate("text", x = 0, y = 0.70, label = title, hjust = 0, family = "Arial", fontface = "bold", size = 3.9, colour = col("ink")) +
    annotate("text", x = 0, y = 0.28, label = subtitle, hjust = 0, family = "Arial", size = 2.05, lineheight = 0.92, colour = col("muted")) +
    xlim(0, 1) +
    ylim(0, 1) +
    theme_void() +
    theme(plot.margin = margin(0, 4, 0, 4))
}

make_figure_1 <- function() {
  baseline <- read_csv(path("outputs", "manuscript_tables", "Table_1_baseline_characteristics.csv"), show_col_types = FALSE)
  smd <- read_csv(path("outputs", "manuscript_tables", "Table_2a_common_feature_smd.csv"), show_col_types = FALSE)

  cohort <- tibble(
    source = factor(c("NHANES", "MIMIC all", "MIMIC first"), levels = c("NHANES", "MIMIC all", "MIMIC first")),
    n = c(6564, 133450, 46312),
    events = c(166, 25099, 6740),
    event_rate = c(0.025, 0.188, 0.146)
  )

  p_cohort <- ggplot(cohort, aes(source, event_rate, fill = source)) +
    geom_col(width = 0.62, colour = "white", linewidth = 0.25) +
    geom_text(aes(label = paste0(round(event_rate * 100, 1), "%\n", format(events, big.mark = ","), " events")),
              vjust = -0.25, size = 2.05, lineheight = 0.88, family = "Arial", colour = col("ink")) +
    scale_fill_manual(values = c("NHANES" = col("nhanes"), "MIMIC all" = col("mimic"), "MIMIC first" = "#8DBA98"), guide = "none") +
    scale_y_continuous(labels = function(x) percent_lab(x, accuracy = 1), limits = c(0, 0.23), breaks = c(0, 0.05, 0.10, 0.15, 0.20), expand = c(0, 0)) +
    labs(title = "Outcome prevalence differs", subtitle = "One-year mortality is much higher in hospital admissions.", x = NULL, y = "Event rate", tag = "a") +
    theme_npj() +
    theme(legend.position = "none", axis.text.x = element_text(size = 5.8), panel.grid.major.x = element_blank())

  smd_plot <- smd %>%
    mutate(
      feature_label = factor(feature_label, levels = rev(feature_label[order(abs(smd_mimic_minus_nhanes))])),
      direction = if_else(smd_mimic_minus_nhanes >= 0, "Higher in MIMIC-IV", "Higher in NHANES")
    )

  p_smd <- ggplot(smd_plot, aes(smd_mimic_minus_nhanes, feature_label, fill = direction)) +
    geom_vline(xintercept = 0, colour = col("ink"), linewidth = 0.32) +
    geom_vline(xintercept = c(-0.1, 0.1), colour = col("muted"), linewidth = 0.25, linetype = "22") +
    geom_vline(xintercept = c(-0.2, 0.2), colour = col("muted"), linewidth = 0.25, linetype = "33") +
    geom_col(width = 0.64) +
    scale_fill_manual(values = c("Higher in MIMIC-IV" = col("mimic"), "Higher in NHANES" = col("nhanes")), guide = "none") +
    scale_x_continuous(limits = c(-0.36, 0.62), breaks = c(-0.2, 0, 0.2, 0.4, 0.6), expand = expansion(mult = c(0.02, 0.02))) +
    labs(title = "Common predictors shift", subtitle = "Standardized mean difference, MIMIC-IV minus NHANES.", x = "SMD", y = NULL, tag = "b") +
    theme_npj() +
    theme(legend.position = "bottom", panel.grid.major.y = element_blank())

  missing_rows <- baseline %>%
    filter(variable %in% c("BMI, kg/m2", "Systolic BP, mmHg", "Diastolic BP, mmHg", "HbA1c", "Glucose", "Creatinine")) %>%
    transmute(
      variable,
      nhanes_missing = 1 - (`NHANES participants nonmissing` / 6564),
      mimic_missing = 1 - (`MIMIC-IV admissions nonmissing` / 133450)
    ) %>%
    pivot_longer(c(nhanes_missing, mimic_missing), names_to = "source", values_to = "missing") %>%
    mutate(
      source = recode(source, nhanes_missing = "NHANES", mimic_missing = "MIMIC-IV"),
      source = factor(source, levels = c("NHANES", "MIMIC-IV")),
      variable = factor(variable, levels = rev(c("BMI, kg/m2", "Systolic BP, mmHg", "Diastolic BP, mmHg", "HbA1c", "Glucose", "Creatinine")))
    )

  p_missing <- ggplot(missing_rows, aes(source, variable, fill = missing)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    geom_text(aes(label = percent_lab(missing, accuracy = 1)), size = 2.0, family = "Arial", colour = col("ink")) +
    scale_fill_gradient(low = "#F7FBF7", high = "#3F7F5E", limits = c(0, 1), labels = function(x) percent_lab(x, accuracy = 1)) +
    labs(title = "Measurement availability shifts", subtitle = "Protocolized versus clinically selective measurement.", x = NULL, y = NULL, fill = "Missing", tag = "c") +
    theme_npj() +
    theme(panel.grid = element_blank(), axis.text.x = element_text(size = 6.2), legend.position = "right")

  fig <- title_block(
    "NHANES and MIMIC-IV define distinct diabetes mortality settings",
    "Population survey participants and hospital admissions differ in baseline mortality, common predictor distributions and measurement availability."
  ) / (p_cohort | p_smd | p_missing) +
    plot_layout(heights = c(0.16, 1), widths = c(0.78, 1.18, 1.12), guides = "keep") &
    theme(plot.background = element_rect(fill = "white", colour = NA))

  save_npj(fig, "figure_1_data_shift", width_mm = 183, height_mm = 112)
}

make_figure_2 <- function() {
  metrics <- read_csv(path("outputs", "manuscript_tables", "Table_3_transportability_metrics_ci.csv"), show_col_types = FALSE)
  cal <- read_csv(path("outputs", "nhanes_mimic_oneyear_mortality_transport", "transportability_calibration.csv"), show_col_types = FALSE)
  subgroup <- read_csv(path("outputs", "manuscript_tables", "Table_5_subgroup_transportability.csv"), show_col_types = FALSE)

  metric_rows <- metrics %>%
    filter(
      (train_source == "NHANES" & test_target == "MIMIC-IV" & feature_set == "base" & model == "logistic_regression") |
        (train_source == "MIMIC-IV" & test_target == "MIMIC-IV" & feature_set == "base" & model == "hist_gradient_boosting")
    ) %>%
    mutate(
      setting = if_else(train_source == "NHANES", "Raw transport", "MIMIC internal"),
      setting = factor(setting, levels = c("Raw transport", "MIMIC internal"))
    ) %>%
    select(setting, roc_auc, brier_score, ece_10bin) %>%
    pivot_longer(c(roc_auc, brier_score, ece_10bin), names_to = "metric", values_to = "value_ci") %>%
    mutate(
      value = extract_mid(value_ci),
      metric = recode(metric, roc_auc = "ROC AUC", brier_score = "Brier score", ece_10bin = "ECE"),
      metric = factor(metric, levels = c("ROC AUC", "Brier score", "ECE"))
    )

  p_metrics <- ggplot(metric_rows, aes(metric, value, colour = setting, group = setting)) +
    geom_line(linewidth = 0.85, alpha = 0.88) +
    geom_point(size = 2.8) +
    geom_text(aes(label = sprintf("%.3f", value)), nudge_y = 0.035, size = 2.05, family = "Arial", colour = col("ink"), show.legend = FALSE) +
    scale_colour_manual(values = c("Raw transport" = col("raw"), "MIMIC internal" = col("internal"))) +
    scale_y_continuous(limits = c(0, 0.84), breaks = c(0, 0.2, 0.4, 0.6, 0.8), expand = c(0, 0)) +
    labs(title = "Discrimination is retained, calibration is not", subtitle = "Raw transport keeps moderate AUC but has much larger ECE.", x = NULL, y = "Metric value", tag = "a") +
    theme_npj() +
    theme(legend.position = "bottom", panel.grid.major.x = element_blank())

  cal_focus <- cal %>%
    filter(analysis %in% c("NHANES_to_MIMIC-IV_base_logistic_regression", "MIMIC-IV_to_MIMIC-IV_base_hist_gradient_boosting")) %>%
    mutate(
      label = recode(
        analysis,
        `NHANES_to_MIMIC-IV_base_logistic_regression` = "Raw NHANES-to-MIMIC",
        `MIMIC-IV_to_MIMIC-IV_base_hist_gradient_boosting` = "MIMIC internal HGB"
      ),
      label = factor(label, levels = c("Raw NHANES-to-MIMIC", "MIMIC internal HGB"))
    )

  p_cal <- ggplot(cal_focus, aes(mean_predicted_probability, observed_probability, colour = label)) +
    geom_abline(slope = 1, intercept = 0, colour = col("ink"), linewidth = 0.35, linetype = "22") +
    geom_line(linewidth = 0.95) +
    geom_point(size = 1.7) +
    annotate("text", x = 0.60, y = 0.82, label = "perfect calibration", size = 1.95, colour = col("muted"), family = "Arial", hjust = 0) +
    scale_colour_manual(values = c("Raw NHANES-to-MIMIC" = col("raw"), "MIMIC internal HGB" = col("internal"))) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
    labs(title = "Calibration curve drifts upward", subtitle = "Predicted probabilities exceed observed mortality after transport.", x = "Mean predicted risk", y = "Observed event rate", tag = "b") +
    theme_npj() +
    theme(legend.position = "bottom")

  state <- metrics %>%
    filter(
      (train_source == "NHANES" & test_target == "MIMIC-IV" & feature_set == "base" & model == "logistic_regression") |
        (train_source == "MIMIC-IV" & test_target == "MIMIC-IV" & feature_set == "base" & model == "hist_gradient_boosting")
    ) %>%
    transmute(
      setting = if_else(train_source == "NHANES", "Raw transport", "MIMIC internal HGB"),
      slope = extract_mid(calibration_slope),
      intercept = extract_mid(calibration_intercept)
    )

  p_state <- ggplot(state, aes(slope, intercept, colour = setting)) +
    geom_hline(yintercept = 0, colour = col("ink"), linewidth = 0.32, linetype = "22") +
    geom_vline(xintercept = 1, colour = col("ink"), linewidth = 0.32, linetype = "22") +
    geom_point(size = 3.0) +
    geom_text(aes(label = setting), nudge_x = c(0.06, 0.05), nudge_y = c(-0.10, 0.10), hjust = 0, size = 2.05, family = "Arial") +
    annotate("point", x = 1, y = 0, shape = 8, size = 3.5, colour = col("ink")) +
    annotate("text", x = 1.05, y = 0.05, label = "ideal", hjust = 0, size = 1.95, family = "Arial") +
    scale_colour_manual(values = c("Raw transport" = col("raw"), "MIMIC internal HGB" = col("internal"))) +
    coord_cartesian(xlim = c(0.30, 1.30), ylim = c(-1.70, 0.55), clip = "on") +
    labs(title = "Slope and intercept reveal failure mode", subtitle = "Raw transport combines overprediction with slope compression.", x = "Calibration slope", y = "Calibration intercept", tag = "c") +
    theme_npj() +
    theme(legend.position = "none")

  subgroup_plot <- subgroup %>%
    filter(subgroup_type %in% c("Age", "CKD history", "CVD history"), subgroup != "Overall") %>%
    mutate(
      subgroup = factor(subgroup, levels = rev(c("Age <65", "Age >=65", "No CKD", "CKD", "No CVD", "CVD"))),
      subgroup_type = factor(subgroup_type, levels = c("Age", "CKD history", "CVD history"))
    )

  p_subgroup <- ggplot(subgroup_plot, aes(ece_10bin, subgroup, fill = subgroup_type)) +
    geom_col(width = 0.64, colour = "white", linewidth = 0.22) +
    geom_text(aes(label = sprintf("%.3f", ece_10bin)), hjust = -0.12, size = 1.95, family = "Arial", colour = col("ink")) +
    scale_fill_manual(values = c("Age" = "#8DA0CB", "CKD history" = "#66A61E", "CVD history" = "#E6AB02")) +
    scale_x_continuous(limits = c(0, 0.48), breaks = c(0, 0.2, 0.4), expand = c(0, 0)) +
    labs(title = "Average calibration hides subgroup heterogeneity", subtitle = "ECE is higher in older, CKD and CVD admissions.", x = "Subgroup ECE", y = NULL, tag = "d") +
    theme_npj() +
    theme(legend.position = "none", panel.grid.major.y = element_blank())

  fig <- title_block(
    "Transport failure is dominated by calibration drift",
    "The NHANES-trained model preserves moderate risk ranking in MIMIC-IV, but absolute mortality probabilities are compressed and overestimated."
  ) / ((p_metrics | p_cal) / (p_state | p_subgroup)) +
    plot_layout(heights = c(0.15, 1), widths = c(1, 1)) &
    theme(plot.background = element_rect(fill = "white", colour = NA))

  save_npj(fig, "figure_2_transport_calibration_failure", width_mm = 183, height_mm = 130)
}

make_figure_3 <- function() {
  rec <- read_csv(path("outputs", "manuscript_tables", "Table_4b_recalibration_uncertainty_intervals.csv"), show_col_types = FALSE) %>%
    filter(
      prediction_set == "NHANES_to_MIMIC_1y_base_logistic_regression",
      method %in% c("raw", "intercept_only", "isotonic", "platt"),
      event_target %in% c(0, 25, 50, 100, 200)
    ) %>%
    mutate(
      method_label = recode(method, raw = "Raw transport", intercept_only = "Intercept-only", isotonic = "Isotonic", platt = "Platt"),
      method_label = factor(method_label, levels = c("Raw transport", "Intercept-only", "Isotonic", "Platt"))
    )
  dca <- read_csv(path("outputs", "decision_curve", "decision_curve_transportability.csv"), show_col_types = FALSE)

  raw_row <- rec %>% filter(method == "raw") %>% slice(1)
  event_df <- rec %>% filter(method != "raw")

  p_ece <- ggplot(event_df, aes(event_target, ece_10bin_mean, colour = method_label, fill = method_label)) +
    geom_hline(yintercept = raw_row$ece_10bin_mean, colour = col("raw"), linewidth = 0.32, linetype = "22") +
    geom_ribbon(aes(ymin = ece_10bin_ci_lower, ymax = ece_10bin_ci_upper), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 0.95) +
    geom_point(size = 1.9) +
    annotate("text", x = 28, y = raw_row$ece_10bin_mean + 0.012, label = "raw ECE 0.286", hjust = 0, size = 1.95, colour = col("raw"), family = "Arial") +
    annotate("text", x = 108, y = 0.041, label = "Platt 100\nECE 0.016", hjust = 0, lineheight = 0.88, size = 1.92, colour = col("platt"), family = "Arial") +
    scale_colour_manual(values = method_cols) +
    scale_fill_manual(values = method_cols) +
    scale_x_continuous(breaks = c(25, 50, 100, 200), limits = c(20, 210), expand = expansion(mult = c(0.02, 0.04))) +
    scale_y_continuous(limits = c(0, 0.31), breaks = c(0, 0.05, 0.10, 0.20, 0.30), expand = expansion(mult = c(0.02, 0.02))) +
    labs(title = "ECE falls rapidly with local events", subtitle = "Empirical 2.5-97.5 percentile intervals across 200 resamples.", x = "Local outcome events", y = "Expected calibration error", tag = "a") +
    theme_npj() +
    theme(legend.position = "bottom")

  p_state <- ggplot() +
    geom_hline(yintercept = 0, colour = col("ink"), linewidth = 0.32, linetype = "22") +
    geom_vline(xintercept = 1, colour = col("ink"), linewidth = 0.32, linetype = "22") +
    geom_path(data = rec %>% filter(method != "raw"), aes(calibration_slope_mean, calibration_intercept_mean, colour = method_label, group = method_label), linewidth = 0.95) +
    geom_point(data = rec %>% filter(method != "raw"), aes(calibration_slope_mean, calibration_intercept_mean, colour = method_label), size = 1.9) +
    geom_point(data = raw_row, aes(calibration_slope_mean, calibration_intercept_mean), colour = col("raw"), size = 3.2) +
    annotate("point", x = 1, y = 0, shape = 8, colour = col("ink"), size = 3.3) +
    annotate("text", x = raw_row$calibration_slope_mean + 0.06, y = raw_row$calibration_intercept_mean + 0.17, label = "raw", colour = col("raw"), hjust = 0, size = 1.95, family = "Arial") +
    annotate("text", x = 1.08, y = 0.17, label = "Platt 100-200", colour = col("platt"), hjust = 0, size = 1.95, family = "Arial") +
    scale_colour_manual(values = method_cols) +
    coord_cartesian(xlim = c(0.18, 1.55), ylim = c(-1.62, 0.60), clip = "on") +
    labs(title = "Platt moves predictions toward ideal calibration", subtitle = "Intercept-only corrects location but not slope.", x = "Calibration slope", y = "Calibration intercept", tag = "b") +
    theme_npj() +
    theme(legend.position = "none")

  dca_plot <- dca %>%
    filter(model %in% c("Treat none", "Treat all", "NHANES raw logistic", "NHANES Platt 100 events", "MIMIC internal HGB")) %>%
    mutate(
      model_label = recode(model, `NHANES raw logistic` = "Raw transport", `NHANES Platt 100 events` = "Platt 100 events", `MIMIC internal HGB` = "MIMIC internal"),
      model_label = factor(model_label, levels = c("Treat none", "Treat all", "Raw transport", "Platt 100 events", "MIMIC internal"))
    ) %>%
    filter(threshold >= 0.05, threshold <= 0.37)

  dca_labels <- dca_plot %>%
    filter(threshold == 0.30, model_label %in% c("Raw transport", "Platt 100 events", "MIMIC internal")) %>%
    mutate(
      label_x = case_when(
        model_label == "Raw transport" ~ 0.278,
        TRUE ~ threshold + 0.006
      ),
      label_y = case_when(
        model_label == "MIMIC internal" ~ net_benefit + 0.012,
        model_label == "Platt 100 events" ~ net_benefit + 0.004,
        model_label == "Raw transport" ~ net_benefit - 0.008,
        TRUE ~ net_benefit
      ),
      label = as.character(model_label)
    )

  p_dca <- ggplot(dca_plot, aes(threshold, net_benefit, colour = model_label, linetype = model_label)) +
    geom_hline(yintercept = 0, colour = col("ink"), linewidth = 0.32) +
    geom_line(linewidth = 0.95) +
    geom_text(data = dca_labels, aes(label_x, label_y, label = label, colour = model_label), inherit.aes = FALSE, hjust = 0, size = 2.0, family = "Arial") +
    scale_colour_manual(values = method_cols) +
    scale_linetype_manual(values = c("Treat none" = "dotted", "Treat all" = "dashed", "Raw transport" = "solid", "Platt 100 events" = "solid", "MIMIC internal" = "solid")) +
    scale_x_continuous(breaks = c(0.1, 0.2, 0.3), limits = c(0.05, 0.37), expand = expansion(mult = c(0.01, 0.08))) +
    coord_cartesian(ylim = c(-0.07, 0.14), clip = "on") +
    labs(title = "Recalibration restores positive net benefit", subtitle = "Raw transport becomes unfavorable at higher thresholds.", x = "Risk threshold", y = "Net benefit", tag = "c") +
    theme_npj() +
    theme(legend.position = "none")

  fig <- ((p_ece | p_state) / p_dca) +
    plot_layout(heights = c(1, 1.05), widths = c(1, 1), guides = "keep") +
    plot_annotation(
      title = "Local Platt recalibration repairs calibration and utility",
      subtitle = "With 100-200 target-site outcome events, Platt recalibration restores calibration parameters toward their ideal values and improves threshold-based net benefit.",
      theme = theme(
        plot.title = element_text(family = "Arial", face = "bold", size = 13.5, colour = col("ink"), hjust = 0),
        plot.subtitle = element_text(family = "Arial", size = 7.2, colour = col("muted"), hjust = 0, margin = margin(t = 1, b = 6)),
        plot.background = element_rect(fill = "white", colour = NA),
        plot.margin = margin(6, 8, 6, 8)
      )
    )

  save_npj(fig, "figure_3_recalibration_repair", width_mm = 183, height_mm = 118)
}

make_supplementary_source_classifier <- function() {
  roc <- read_csv(path("outputs", "domain_shift", "source_classifier_roc_curve.csv"), show_col_types = FALSE)
  summary <- read_csv(path("outputs", "manuscript_tables", "Table_2b_source_classifier.csv"), show_col_types = FALSE) %>%
    mutate(
      feature_label = recode(feature_set, basic_features = "Basic features", lab_enhanced_features = "Lab-enhanced features"),
      model_label = recode(model, logistic_regression = "Logistic regression", random_forest = "Random forest"),
      label = paste(feature_label, model_label, sprintf("AUC %.3f", roc_auc), sep = "\n")
    )

  roc_plot <- roc %>%
    mutate(
      feature_label = recode(feature_set, basic_features = "Basic features", lab_enhanced_features = "Lab-enhanced features"),
      model_label = recode(model, logistic_regression = "Logistic regression", random_forest = "Random forest")
    ) %>%
    left_join(summary %>% select(feature_label, model_label, label), by = c("feature_label", "model_label"))

  p_roc <- ggplot(roc_plot, aes(fpr, tpr, colour = label)) +
    geom_abline(slope = 1, intercept = 0, colour = col("muted"), linewidth = 0.32, linetype = "22") +
    geom_line(linewidth = 0.9) +
    scale_colour_manual(values = c("#6F6F6F", "#4C78A8", "#7B5EA7", "#D27C3F")) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = "Supplementary source-classifier analysis", subtitle = "Laboratory-augmented random forest nearly separates NHANES from MIMIC-IV.", x = "False-positive rate", y = "True-positive rate", colour = NULL, tag = "a") +
    theme_npj() +
    theme(legend.position = "right")

  save_npj(p_roc, "supplementary_figure_source_classifier", width_mm = 130, height_mm = 95)
}

make_figure_1()
make_figure_2()
make_figure_3()
make_supplementary_source_classifier()

message("Wrote npj-style figures to: ", out_dir)
