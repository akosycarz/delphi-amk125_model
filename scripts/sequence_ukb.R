# sequence_ukb.R
# Creates a unified sequence dataframe: eid, coding, code, age
# Sources:
#   1. clinical_history_icd.rds  - ICD clinical events (already long format)
#   2. ukb_cleaning.rds          - UKB bulk variables (wide format -> long)
library(dplyr)
library(lubridate)
#  Paths
clinical_path <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/clinical_history_icd.rds"
cleaning_path <- "/rds/general/project/hda_24-25/live/amk125_thesis/General/extraction_and_recoding/outputs/ukb_cleaning.rds"
out_path <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sequence_ukb.rds"
#  1. Clinical history (ICD)
cat("Loading clinical history...\n")
clinical <- readRDS(clinical_path)  # cols: eid, code, coding, date
cat("Loading ukb_cleaning for birth dates...\n")
ukb <- readRDS(cleaning_path)
clinical$eid <- as.character(clinical$eid)
ukb$eid      <- as.character(ukb$eid)
# Join approx_birth_date from ukb onto clinical events
birth_dates <- ukb %>% select(eid, approx_birth_date)
clinical <- clinical %>%
  left_join(birth_dates, by = "eid")
# Calculate age at event in years (float)
clinical$date              <- as.Date(clinical$date)
clinical$approx_birth_date <- as.Date(clinical$approx_birth_date)
clinical$age <- as.numeric(
  difftime(clinical$date, clinical$approx_birth_date, units = "days")
) / 365.25
# Filter implausible ages
clinical <- clinical %>% filter(age >= 0 & age <= 120)
# Keep only required columns
seq_clinical <- clinical %>%
  select(eid, coding, code, age)
cat("Clinical records:", nrow(seq_clinical), "\n")
#  2. Bulk UKB variables
cat("Building bulk variable rows...\n")
# Columns to EXCLUDE entirely
exclude_cols <- c(
  "month_of_birth.0.0",
  "age_of_assessment.0.0",
  "age_of_recruitment.0.0",
  "year_of_birth.0.0",
  "approx_birth_date"
)
exclude_cols <- intersect(exclude_cols, names(ukb_cleaning))

# Regex patterns to EXCLUDE
cancer_pat         <- "^cancer_code_self_reported\\.0\\.[0-9]+$"
noncancer_pat      <- "^non_cancer_illness_code_self_reported\\.0\\.[0-9]+$"
cancer_date_pat    <- "^cancer_code_self_reported_date\\.0\\.[0-9]+$"
noncancer_date_pat <- "^non_cancer_illness_code_self_reported_date\\.0\\.[0-9]+$"
all_cols  <- names(ukb)
bulk_cols <- all_cols[
  !all_cols %in% exclude_cols &
    !grepl(cancer_pat,         all_cols) &
    !grepl(noncancer_pat,      all_cols) &
    !grepl(cancer_date_pat,    all_cols) &
    !grepl(noncancer_date_pat, all_cols) &
    all_cols != "eid"
]
cat("Bulk columns to include:", length(bulk_cols), "\n")
# Reference age of recruitment (for randomisation)
age_recruit <- as.numeric(ukb$age_of_recruitment.0.0)
# Set seed for reproducibility
set.seed(42)
# Matrix approach: vectorised over all participants x columns at once
bulk_mat  <- as.matrix(ukb[, bulk_cols])
noise_mat <- matrix(
  runif(nrow(ukb) * length(bulk_cols), min = -5, max = 5),
  nrow = nrow(ukb), ncol = length(bulk_cols)
)
# Age = recruitment age + noise (broadcast age_recruit across columns)
age_mat <- age_recruit + noise_mat
# Sex gets age 0
sex_idx <- which(bulk_cols == "sex.0.0")
if (length(sex_idx)) age_mat[, sex_idx] <- 0
# Extract non-NA positions in one pass
non_na <- which(!is.na(bulk_mat), arr.ind = TRUE)
seq_bulk <- data.frame(
  eid    = ukb$eid[non_na[, 1]],
  coding = "bulk",
  code   = bulk_cols[non_na[, 2]],
  age    = age_mat[non_na],
  stringsAsFactors = FALSE
)
cat("Bulk records (after NA filter):", nrow(seq_bulk), "\n")
#  3. Combine and save
cat("Combining...\n")
seq_ukb <- bind_rows(seq_clinical, seq_bulk)
cat("Total records:", nrow(seq_ukb), "\n")
cat("Columns:", paste(names(seq_ukb), collapse = ", "), "\n")
saveRDS(seq_ukb, out_path)
cat("Saved to", out_path, "\n")

