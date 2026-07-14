# ============================================================
# Biomarker Quality Exploration Script
# Author: amk125
# Date: 2026-06-05
# Outputs: missingness tables, distribution plots, correlation heatmaps
# ============================================================

# ---- Libraries ----
library(tidyverse)
library(ggplot2)
library(reshape2)
library(corrplot)
library(gridExtra)
library(scales)

# ---- Output directory ----
out_dir <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/sources/biomarker_outputs"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load data ----
message("Loading data...")
ukb <- readRDS("/rds/general/project/hda_24-25/live/amk125_thesis/General/extraction_and_recoding/outputs/ukb_recoded.rds")
message(paste("Data loaded:", nrow(ukb), "rows,", ncol(ukb), "columns"))

# ============================================================
# Define biomarker groups
# ============================================================

blood_biochem_vars <- c(
    "alanine_aminotransferase", "albumin_bio", "alkaline_phosphatase",
      "apolipoprotein_a", "apolipoprotein_b", "aspartate_aminotransferase",
      "c_reactive_protein", "calcium", "cholesterol", "creatinine_bio",
      "cystatin_c", "direct_bilirubin", "gamma_glutamyltransferase",
      "glucose_bio", "glycated_haemoglobin_hba1c", "hdl_cholesterol_bio",
      "igf_1", "ldl_direct", "lipoprotein_a", "oestradiol",
      "phosphate", "rheumatoid_factor", "shbg", "testosterone",
      "total_bilirubin", "total_protein", "triglycerides", "urate",
      "urea", "vitamin_d"
)

blood_count_vars <- c(
    "basophill_count", "eosinophill_count", "haemoglobin_concentration",
      "immature_reticulocyte_fraction", "lymphocyte_count",
      "mean_corpuscular_haemoglobin", "mean_corpuscular_volume",
      "mean_platelet_thrombocyte_volume", "mean_reticulocyte_volume",
      "mean_sphered_cell_volume", "monocyte_count", "neutrophill_count",
      "nucleated_red_blood_cell_count", "platelet_count", "platelet_crit",
      "red_blood_cell_count", "reticulocyte_count", "white_blood_cell_count"
)

# Helper: resolve column names (try exact match, then .0.0 suffix)
resolve_vars <- function(vars, df) {
    found <- c()
      for (v in vars) {
            if (v %in% names(df)) {
                    found <- c(found, v)
            } else {
                    v2 <- paste0(v, ".0.0")
                          if (v2 %in% names(df)) found <- c(found, v2)
            }
      }
      found
}

biochem_cols <- resolve_vars(blood_biochem_vars, ukb)
count_cols   <- resolve_vars(blood_count_vars, ukb)
all_cols     <- unique(c(biochem_cols, count_cols))

message(paste("Blood biochemistry columns found:", length(biochem_cols)))
message(paste("Blood count columns found:", length(count_cols)))
message(paste("Total biomarker columns found:", length(all_cols)))

# ============================================================
# 1. MISSINGNESS TABLE
# ============================================================
message("\n--- Computing missingness ---")

compute_missingness <- function(df, cols, group_name) {
    miss_df <- data.frame(
          Group     = group_name,
              Variable  = cols,
              N_missing = sapply(cols, function(x) sum(is.na(df[[x]]))),
              N_total   = nrow(df),
              stringsAsFactors = FALSE
    )
      miss_df$Pct_missing <- round(miss_df$N_missing / miss_df$N_total * 100, 2)
        miss_df$N_present   <- miss_df$N_total - miss_df$N_missing
          miss_df <- miss_df[order(miss_df$Pct_missing, decreasing = TRUE), ]
            rownames(miss_df) <- NULL
              miss_df
}

miss_biochem <- compute_missingness(ukb, biochem_cols, "Blood Biochemistry")
miss_count   <- compute_missingness(ukb, count_cols,   "Blood Count")
miss_all     <- rbind(miss_biochem, miss_count)

write.csv(miss_biochem, file.path(out_dir, "missingness_blood_biochemistry.csv"), row.names = FALSE)
write.csv(miss_count,   file.path(out_dir, "missingness_blood_count.csv"),        row.names = FALSE)
write.csv(miss_all,     file.path(out_dir, "missingness_all_biomarkers.csv"),     row.names = FALSE)
message("Missingness tables saved.")

# Missingness bar plot - Blood Biochemistry
p_miss_biochem <- ggplot(miss_biochem, aes(x = reorder(Variable, Pct_missing), y = Pct_missing)) +
    geom_bar(stat = "identity", fill = "#E07B54") +
    coord_flip() +
    labs(title = "Missingness: Blood Biochemistry Biomarkers",
                x = "Biomarker", y = "% Missing") +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))
ggsave(file.path(out_dir, "missingness_bar_blood_biochemistry.png"),
              p_miss_biochem, width = 10, height = 7, dpi = 150)

# Missingness bar plot - Blood Count
p_miss_count <- ggplot(miss_count, aes(x = reorder(Variable, Pct_missing), y = Pct_missing)) +
    geom_bar(stat = "identity", fill = "#5B8DB8") +
    coord_flip() +
    labs(title = "Missingness: Blood Count Biomarkers",
                x = "Biomarker", y = "% Missing") +
    theme_bw(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))
ggsave(file.path(out_dir, "missingness_bar_blood_count.png"),
              p_miss_count, width = 10, height = 7, dpi = 150)

# Missingness bar plot - All biomarkers
p_miss_all <- ggplot(miss_all, aes(x = reorder(Variable, Pct_missing), y = Pct_missing, fill = Group)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_fill_manual(values = c("Blood Biochemistry" = "#E07B54", "Blood Count" = "#5B8DB8")) +
    labs(title = "Missingness: All Biomarkers", x = "Biomarker", y = "% Missing", fill = "Group") +
    theme_bw(base_size = 10) +
    theme(plot.title = element_text(face = "bold"))
ggsave(file.path(out_dir, "missingness_bar_all_biomarkers.png"),
              p_miss_all, width = 12, height = 12, dpi = 150)
message("Missingness plots saved.")

# ============================================================
# 2. DISTRIBUTION SUMMARIES
# ============================================================
message("\n--- Computing distribution summaries ---")

if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
library(e1071)

compute_summary <- function(df, cols, group_name) {
    do.call(rbind, lapply(cols, function(x) {
          vals <- df[[x]]
              data.frame(
                      Group    = group_name,
                            Variable = x,
                            N        = sum(!is.na(vals)),
                            Mean     = round(mean(vals, na.rm = TRUE), 4),
                            SD       = round(sd(vals,   na.rm = TRUE), 4),
                            Median   = round(median(vals, na.rm = TRUE), 4),
                            IQR      = round(IQR(vals, na.rm = TRUE), 4),
                            Min      = round(min(vals, na.rm = TRUE), 4),
                            Max      = round(max(vals, na.rm = TRUE), 4),
                            Skewness = round(e1071::skewness(vals, na.rm = TRUE), 4),
                            stringsAsFactors = FALSE
              )
    }))
}

summ_biochem <- compute_summary(ukb, biochem_cols, "Blood Biochemistry")
summ_count   <- compute_summary(ukb, count_cols,   "Blood Count")
summ_all     <- rbind(summ_biochem, summ_count)

write.csv(summ_biochem, file.path(out_dir, "distribution_summary_blood_biochemistry.csv"), row.names = FALSE)
write.csv(summ_count,   file.path(out_dir, "distribution_summary_blood_count.csv"),        row.names = FALSE)
write.csv(summ_all,     file.path(out_dir, "distribution_summary_all_biomarkers.csv"),     row.names = FALSE)
message("Distribution summary tables saved.")

# Distribution histograms - Blood Biochemistry
message("Plotting distribution histograms for Blood Biochemistry...")
biochem_data_long <- ukb[, biochem_cols, drop = FALSE] %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
    filter(!is.na(Value))
p_dist_biochem <- ggplot(biochem_data_long, aes(x = Value)) +
    geom_histogram(bins = 60, fill = "#E07B54", color = "white", alpha = 0.85) +
    facet_wrap(~Variable, scales = "free", ncol = 5) +
    labs(title = "Distribution of Blood Biochemistry Biomarkers", x = "Value", y = "Count") +
    theme_bw(base_size = 9) +
    theme(plot.title = element_text(face = "bold"), strip.text = element_text(size = 7))
ggsave(file.path(out_dir, "distributions_blood_biochemistry.png"),
              p_dist_biochem, width = 20, height = 14, dpi = 150)

# Distribution histograms - Blood Count
message("Plotting distribution histograms for Blood Count...")
count_data_long <- ukb[, count_cols, drop = FALSE] %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
    filter(!is.na(Value))
p_dist_count <- ggplot(count_data_long, aes(x = Value)) +
    geom_histogram(bins = 60, fill = "#5B8DB8", color = "white", alpha = 0.85) +
    facet_wrap(~Variable, scales = "free", ncol = 5) +
    labs(title = "Distribution of Blood Count Biomarkers", x = "Value", y = "Count") +
    theme_bw(base_size = 9) +
    theme(plot.title = element_text(face = "bold"), strip.text = element_text(size = 7))
ggsave(file.path(out_dir, "distributions_blood_count.png"),
              p_dist_count, width = 18, height = 12, dpi = 150)
message("Distribution plots saved.")

# ============================================================
# 3. CORRELATION HEATMAPS
# ============================================================
message("\n--- Computing correlation matrices ---")

compute_and_save_corr <- function(df, cols, group_name, file_prefix) {
    mat     <- df[, cols, drop = FALSE]
      cor_mat <- cor(mat, use = "pairwise.complete.obs", method = "pearson")
        short_names <- gsub("_", " ", gsub("\\.0\\.0$", "", cols))
          rownames(cor_mat) <- short_names
            colnames(cor_mat) <- short_names
              write.csv(as.data.frame(cor_mat),
                                    file.path(out_dir, paste0(file_prefix, "_correlation_matrix.csv")))
                cor_melt <- reshape2::melt(cor_mat, na.rm = TRUE)
                  p_heat <- ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
                        geom_tile(color = "white") +
                        scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#D6604D",
                                                                      midpoint = 0, limit = c(-1, 1), space = "Lab",
                                                                      name = "Pearson\nCorr") +
                        labs(title = paste("Correlation Heatmap:", group_name), x = "", y = "") +
                        theme_bw(base_size = 9) +
                        theme(
                                plot.title   = element_text(face = "bold", size = 11),
                                      axis.text.x  = element_text(angle = 45, hjust = 1, size = 7),
                                      axis.text.y  = element_text(size = 7),
                                      legend.title = element_text(size = 9),
                                      aspect.ratio = 1
                        )
                    w <- max(8, length(cols) * 0.45)
                      ggsave(file.path(out_dir, paste0(file_prefix, "_correlation_heatmap.png")),
                                      p_heat, width = w, height = w * 0.9, dpi = 150)
                        message(paste("Saved correlation heatmap:", file_prefix))
                        invisible(cor_mat)
}

compute_and_save_corr(ukb, biochem_cols, "Blood Biochemistry", "blood_biochemistry")
compute_and_save_corr(ukb, count_cols,   "Blood Count",        "blood_count")
compute_and_save_corr(ukb, all_cols,     "All Biomarkers",     "all_biomarkers")

message("\n============================================================")
message("All outputs saved to:")
message(out_dir)
message("============================================================")
message("Script complete!")