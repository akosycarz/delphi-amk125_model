library(data.table)

in_path  <- "/rds/general/project/hda_24-25/live/amk125_thesis/General/extraction_and_recoding/outputs/ukb_cleaning.rds"
out_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sources"
out_path <- file.path(out_dir, "ukb_bulk.rds")   # rename if you like

# --- columns to EXCLUDE ---------------------------------------------------

# Blood biochemistry (already in blood_biochemistry.rds)
biochem_cols <- c(
  "alanine_aminotransferase.0.0", "albumin_bio.0.0", "alkaline_phosphatase.0.0",
  "apolipoprotein_a.0.0", "apolipoprotein_b_bio.0.0", "aspartate_aminotransferase.0.0",
  "c_reactive_protein.0.0", "calcium.0.0", "cholesterol.0.0", "creatinine_bio.0.0",
  "cystatin_c.0.0", "direct_bilirubin.0.0", "gamma_glutamyltransferase.0.0",
  "glucose_bio.0.0", "glycated_haemoglobin_hba1c.0.0", "hdl_cholesterol_bio.0.0",
  "igf_1.0.0", "ldl_direct.0.0", "lipoprotein_a.0.0", "oestradiol.0.0", "phosphate.0.0",
  "rheumatoid_factor.0.0", "shbg.0.0", "testosterone.0.0", "total_bilirubin.0.0",
  "total_protein.0.0", "triglycerides.0.0", "urate.0.0", "urea.0.0", "vitamin_d.0.0"
)

# Demographics (already in demographics.rds)
demo_cols <- c("smoking_status", "BMI.0.0", "sex.0.0", "alcohol_status")



# Self-reported illness / cancer-date arrays (regex)
drop_patterns <- c(
  "^non_cancer_illness_code_self_reported\\.0\\.[0-9]+$",
  "^cancer_code_self_reported_date\\.0\\.[0-9]+$",
  "^non_cancer_illness_code_self_reported_date\\.0\\.[0-9]+$",
  "^cancer_code_self_reported\\.0\\.[0-9]+$"
)

# --- read -----------------------------------------------------------------
df <- readRDS(in_path)

# This cleaning output may already carry an explicit eid column; only pull
# from row names if it doesn't.
if ("eid" %in% colnames(df)) {
  setDT(df)
} else {
  setDT(df, keep.rownames = "eid")
}

all_cols <- colnames(df)

# columns caught by the regex patterns
pattern_hits <- unique(unlist(
  lapply(drop_patterns, grep, x = all_cols, value = TRUE)
))

# full exclusion set
exclude_cols <- unique(c(biochem_cols, demo_cols, pattern_hits))

# (optional) flag literal exclusion names that aren't in the data — catches typos
not_found <- setdiff(c(biochem_cols, demo_cols), all_cols)
if (length(not_found)) {
  message("Note: these exclusion names weren't found in ukb_cleaning: ",
          paste(not_found, collapse = ", "))
}

# keep everything that isn't excluded (and isn't eid itself)
keep_cols <- setdiff(all_cols, c("eid", exclude_cols))
df_sub <- df[, c("eid", keep_cols), with = FALSE]

# coerce all measure columns to character so melt has a single target type
df_sub[, (keep_cols) := lapply(.SD, as.character), .SDcols = keep_cols]

# --- wide -> long ---------------------------------------------------------
long_dt <- melt(
  df_sub,
  id.vars         = "eid",
  variable.name   = "coding",
  value.name      = "code",
  variable.factor = FALSE,
  na.rm           = TRUE
)

# --- strip the .0.N array suffix from coding -------------------------------
long_dt[, coding := sub("\\.0\\.[0-9]+$", "", coding)]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(long_dt, out_path)