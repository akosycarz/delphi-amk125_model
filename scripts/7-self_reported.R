# self_reported_to_seq.R
# Extracts self-reported cancer and non-cancer illness data from ukb_cleaning.rds,
# reshapes to long format, and converts diagnosis year to age at diagnosis.
# Output matches seq_ukb format: eid, coding, code, age

library(dplyr)
library(tidyr)

# ── Paths ─────────────────────────────────────────────────────────────────────
cleaning_path    <- "/rds/general/project/hda_24-25/live/amk125_thesis/General/extraction_and_recoding/outputs/ukb_cleaning.rds"
out_cancer_path <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sources/seq_cancer_self_reported.rds"
out_noncancer_path <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sources/seq_noncancer_self_reported.rds"

# ── Load ──────────────────────────────────────────────────────────────────────
cat("Loading ukb_cleaning...\n")
ukb <- readRDS(cleaning_path)
ukb$eid <- as.character(ukb$eid)

# Birth year and recruitment age for age calculation / imputation
birth_year   <- as.numeric(format(as.Date(ukb$approx_birth_date), "%Y"))
age_recruit  <- as.numeric(ukb$age_of_recruitment.0.0)
birth_lookup <- data.frame(eid        = ukb$eid,
                           birth_year  = birth_year,
                           age_recruit = age_recruit,
                           stringsAsFactors = FALSE)
set.seed(42)

# ── 1. Four raw matrices ──────────────────────────────────────────────────────
cat("Extracting column subsets...\n")

cancer_codes_mat <- ukb %>%
  select(eid, matches("^cancer_code_self_reported\\.0\\.[0-9]+$"))

cancer_dates_mat <- ukb %>%
  select(eid, matches("^cancer_code_self_reported_date\\.0\\.[0-9]+$")) %>%
  mutate(across(-eid, as.numeric))   # unify mixed char/dbl columns

noncancer_codes_mat <- ukb %>%
  select(eid, matches("^non_cancer_illness_code_self_reported\\.0\\.[0-9]+$"))

noncancer_dates_mat <- ukb %>%
  select(eid, matches("^non_cancer_illness_code_self_reported_date\\.0\\.[0-9]+$")) %>%
  mutate(across(-eid, as.numeric))   # unify mixed char/dbl columns

cat("Cancer code columns:     ", ncol(cancer_codes_mat)    - 1, "\n")
cat("Cancer date columns:     ", ncol(cancer_dates_mat)    - 1, "\n")
cat("Non-cancer code columns: ", ncol(noncancer_codes_mat) - 1, "\n")
cat("Non-cancer date columns: ", ncol(noncancer_dates_mat) - 1, "\n")

# ── 2. Pivot to long, preserving per-entry index for alignment ────────────────

# Helper: pivot codes and dates to long, then join on eid + entry index
pivot_and_join <- function(codes_mat, dates_mat,
                           code_pattern, date_pattern) {
  codes_long <- codes_mat %>%
    pivot_longer(
      cols      = -eid,
      names_to  = "idx",
      values_to = "code",
      names_pattern = paste0(code_pattern, "\\.0\\.([0-9]+)$")
    )
  
  dates_long <- dates_mat %>%
    pivot_longer(
      cols      = -eid,
      names_to  = "idx",
      values_to = "date_year",
      names_pattern = paste0(date_pattern, "\\.0\\.([0-9]+)$")
    )
  
  codes_long %>%
    left_join(dates_long, by = c("eid", "idx"))
}

cat("Pivoting cancer data...\n")
cancer_long <- pivot_and_join(
  cancer_codes_mat, cancer_dates_mat,
  code_pattern = "cancer_code_self_reported",
  date_pattern = "cancer_code_self_reported_date"
)

cat("Pivoting non-cancer data...\n")
noncancer_long <- pivot_and_join(
  noncancer_codes_mat, noncancer_dates_mat,
  code_pattern = "non_cancer_illness_code_self_reported",
  date_pattern = "non_cancer_illness_code_self_reported_date"
)

# ── 3. Remove NAs, compute age, format to match seq_ukb ──────────────────────

prepare_seq <- function(long_df, coding_label, birth_lookup) {
  long_df %>%
    filter(!is.na(code)) %>%
    left_join(birth_lookup, by = "eid") %>%
    mutate(
      age = as.numeric(date_year) - birth_year   # diagnosis year - birth year
    ) %>%
    select(eid, coding = code, code = coding_label, age)
  # NOTE: 'coding' holds the numeric UKB code; 'code' holds the source label
  # to match seq_ukb convention (coding = source type, code = event code).
  # Swap the rename below if your convention is the reverse.
}

# Actually: in seq_ukb, 'coding' = source/vocabulary (e.g. "icd10", "bulk"),
# and 'code' = the actual event code value.
# So here: coding = "self_reported_cancer" / "self_reported_non_cancer",
#           code  = the numeric UKB self-report code.

ukb_date_unknown <- c("Date uncertain or unknown", "Preferred not to answer")

finalize_seq <- function(long_df, coding_label, birth_lookup) {
  df <- long_df %>%
    filter(!is.na(code)) %>%
    # Drop rows with explicitly unknown/refused dates (string or numeric UKB codes)
    filter(!as.character(date_year) %in% ukb_date_unknown) %>%
    filter(is.na(date_year) | !as.numeric(date_year) %in% c(-1, -3)) %>%
    left_join(birth_lookup, by = "eid") %>%
    mutate(date_year = as.numeric(date_year))
  
  n <- nrow(df)
  
  df %>% mutate(
    age = case_when(
      # Date provided: use diagnosis year - birth year
      !is.na(date_year) & date_year > 1900 ~ date_year - birth_year,
      # Date genuinely missing: impute from recruitment age ± uniform noise
      TRUE ~ age_recruit + runif(n, min = -5, max = 5)
    ),
    coding = coding_label,
    code   = as.character(code)
  ) %>%
    select(eid, coding, code, age)
}

cat("Finalising cancer sequence...\n")
seq_cancer <- finalize_seq(cancer_long, "self_reported_cancer", birth_lookup)
cat("Cancer records (non-NA):", nrow(seq_cancer), "\n")

cat("Finalising non-cancer sequence...\n")
seq_noncancer <- finalize_seq(noncancer_long, "self_reported_non_cancer", birth_lookup)
cat("Non-cancer records (non-NA):", nrow(seq_noncancer), "\n")

# ── 4. Save individual files ──────────────────────────────────────────────────
saveRDS(seq_cancer,    out_cancer_path)
saveRDS(seq_noncancer, out_noncancer_path)
cat("Saved cancer to:     ", out_cancer_path, "\n")
cat("Saved non-cancer to: ", out_noncancer_path, "\n")

