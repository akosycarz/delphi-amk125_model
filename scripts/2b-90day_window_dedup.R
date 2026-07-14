# ===========================================================================
# 90-Day Window Deduplication for clinical_history_icd
# ===========================================================================
# For each patient (eid), if the same diagnosis (eid + coding + code) repeats
# within 90 days of the previous kept occurrence, the duplicate is removed.
# Only the first occurrence (and any recurrence after 90+ days) is kept.
# ===========================================================================

library(data.table)

# --- Paths ------------------------------------------------------------------
input_path  <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/sources/clinical_history_icd.rds"
output_path <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/sources/clinical_history_icd.rds"

# --- Load data --------------------------------------------------------------
cat("Loading data...\n")
clinical_history_icd <- readRDS(input_path)
cat("Original rows:", nrow(clinical_history_icd), "\n")

# --- Convert to data.table and sort -----------------------------------------
dt <- as.data.table(clinical_history_icd)
setorder(dt, eid, coding, code, date)

# --- 90-day window function -------------------------------------------------
# Returns logical vector: TRUE = keep, FALSE = remove
keep_90d <- function(dates) {
  if (length(dates) == 1L) return(TRUE)
  keep      <- logical(length(dates))
  keep[1]   <- TRUE
  last_kept <- as.numeric(dates[1])
  for (i in seq_along(dates)[-1]) {
    d <- as.numeric(dates[i])
    if (d - last_kept > 90) {
      keep[i]   <- TRUE
      last_kept <- d
    }
  }
  keep
}

# --- Apply deduplication by group -------------------------------------------
cat("Applying 90-day window deduplication...\n")
dt[, keep := keep_90d(date), by = .(eid, coding, code)]

# --- Filter and clean -------------------------------------------------------
clinical_history_icd_90d <- dt[keep == TRUE][, keep := NULL]
cat("After 90-day deduplication:", nrow(clinical_history_icd_90d), "\n")
cat("Rows removed:", nrow(dt) - nrow(clinical_history_icd_90d), "\n")

# --- Save -------------------------------------------------------------------
cat("Saving...\n")
saveRDS(as.data.frame(clinical_history_icd_90d), output_path)
cat("Done! Saved to:", output_path, "\n")
