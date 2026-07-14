# add_age_to_blood_biochemistry.R
# Creates age columns for all selected RDS files using simple eid lookup tables.

library(data.table)

tmp_dir <- "/rds/general/project/hda_24-25/live/amk125_thesis/tmp"
input_dir <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/sources"
output_dir <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/outputs_with_age"

blood_biochemistry_path <- file.path(input_dir, "blood_biochemistry.rds")
clinical_history_icd_path <- file.path(input_dir, "clinical_history_icd.rds")
demographics_path <- file.path(input_dir, "demographics.rds")
self_reported_cancer_path <- file.path(input_dir, "seq_cancer_self_reported_with_chapter.rds")
self_reported_noncancer_path <- file.path(input_dir, "seq_noncancer_self_reported_with_chapter.rds")
ukb_bulk_path <- file.path(input_dir, "ukb_bulk.rds")

blood_biochemistry_out_path <- file.path(output_dir, "blood_biochemistry.rds")
clinical_history_icd_out_path <- file.path(output_dir, "clinical_history_icd.rds")
demographics_out_path <- file.path(output_dir, "demographics.rds")
self_reported_cancer_out_path <- file.path(output_dir, "seq_cancer_self_reported_with_chapter.rds")
self_reported_noncancer_out_path <- file.path(output_dir, "seq_noncancer_self_reported_with_chapter.rds")
ukb_bulk_out_path <- file.path(output_dir, "ukb_bulk.rds")
final_out_path <- file.path(output_dir, "final.rds")

dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)
Sys.setenv(TMPDIR = tmp_dir)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(42L)

input_paths <- c(
  blood_biochemistry_path,
  clinical_history_icd_path,
  demographics_path,
  self_reported_cancer_path,
  self_reported_noncancer_path,
  ukb_bulk_path
)

missing_paths <- input_paths[!file.exists(input_paths)]
if (length(missing_paths)) {
  stop("These input RDS files do not exist:\n", paste(missing_paths, collapse = "\n"))
}

as_integer_age <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x)] <- NA_real_
  as.integer(floor(x))
}

round_integer_age <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x)] <- NA_real_
  as.integer(round(x))
}

coerce_date <- function(x) {
  if (inherits(x, "Date")) return(x)
  if (inherits(x, "POSIXt")) return(as.Date(x))
  
  x_chr <- trimws(as.character(x))
  x_chr[x_chr == ""] <- NA_character_
  
  out <- suppressWarnings(as.Date(x_chr, format = "%Y-%m-%d"))
  missing <- is.na(out)
  out[missing] <- suppressWarnings(as.Date(x_chr[missing], format = "%d/%m/%Y"))
  out
}

calculate_age_from_date <- function(event_date, approx_birth_date) {
  event_date <- coerce_date(event_date)
  approx_birth_date <- coerce_date(approx_birth_date)
  as_integer_age(as.numeric(difftime(event_date, approx_birth_date, units = "days")) / 365.25)
}

get_value_column <- function(dt) {
  value_candidates <- c("value", "val", "field_value", "measurement", "result", "code")
  value_col <- intersect(value_candidates, names(dt))
  if (length(value_col)) return(value_col[1])
  
  stop(
    "Could not find the value column in ukb_bulk.\n",
    "Expected one of: ", paste(value_candidates, collapse = ", "), "\n",
    "Available columns are: ", paste(names(dt), collapse = ", ")
  )
}

make_age_of_recruitment_matrix <- function(ukb_bulk) {
  ukb_bulk <- as.data.table(ukb_bulk)
  required_cols <- c("eid", "coding")
  missing_cols <- setdiff(required_cols, names(ukb_bulk))
  if (length(missing_cols)) {
    stop("ukb_bulk is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  value_col <- get_value_column(ukb_bulk)
  
  age_of_recruitment_matrix <- ukb_bulk[coding == "age_of_recruitment", .(
    eid = as.character(eid),
    age_of_recruitment = suppressWarnings(as.numeric(get(value_col)))
  )]
  
  age_of_recruitment_matrix <- age_of_recruitment_matrix[
    !is.na(eid) & eid != "" & !is.na(age_of_recruitment)
  ]
  age_of_recruitment_matrix <- age_of_recruitment_matrix[
    , .(age_of_recruitment = age_of_recruitment[1]), by = eid
  ]
  setkey(age_of_recruitment_matrix, eid)
  
  age_of_recruitment_matrix
}

make_approx_birth_date_matrix <- function(ukb_bulk) {
  ukb_bulk <- as.data.table(ukb_bulk)
  required_cols <- c("eid", "coding")
  missing_cols <- setdiff(required_cols, names(ukb_bulk))
  if (length(missing_cols)) {
    stop("ukb_bulk is missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  value_col <- get_value_column(ukb_bulk)
  
  approx_birth_date_matrix <- ukb_bulk[coding == "approx_birth_date", .(
    eid = as.character(eid),
    approx_birth_date = coerce_date(get(value_col))
  )]
  
  approx_birth_date_matrix <- approx_birth_date_matrix[
    !is.na(eid) & eid != "" & !is.na(approx_birth_date)
  ]
  approx_birth_date_matrix <- approx_birth_date_matrix[
    , .(approx_birth_date = approx_birth_date[1]), by = eid
  ]
  setkey(approx_birth_date_matrix, eid)
  
  approx_birth_date_matrix
}

add_age_from_recruitment <- function(dt, age_of_recruitment_matrix) {
  dt <- as.data.table(dt)
  if (!"eid" %in% names(dt)) stop("Input data must contain eid.")
  
  dt[, eid_join_tmp := as.character(eid)]
  dt[age_of_recruitment_matrix, age := as_integer_age(i.age_of_recruitment), on = c(eid_join_tmp = "eid")]
  dt[, eid_join_tmp := NULL]
  
  dt
}

add_age_from_event_date <- function(dt, approx_birth_date_matrix, date_col = "date", remove_date = FALSE) {
  dt <- as.data.table(dt)
  if (!"eid" %in% names(dt)) stop("Input data must contain eid.")
  if (!date_col %in% names(dt)) stop("Input data must contain ", date_col, ".")
  
  dt[, eid_join_tmp := as.character(eid)]
  dt[approx_birth_date_matrix, approx_birth_date_tmp := i.approx_birth_date, on = c(eid_join_tmp = "eid")]
  dt[, age := calculate_age_from_date(get(date_col), approx_birth_date_tmp)]
  dt[, c("eid_join_tmp", "approx_birth_date_tmp") := NULL]
  if (remove_date) {
    dt[, (date_col) := NULL]
  }
  
  dt
}

add_noisy_demographics_age <- function(demographics, age_of_recruitment_matrix) {
  demographics <- as.data.table(demographics)
  demographics <- add_age_from_recruitment(demographics, age_of_recruitment_matrix)
  
  has_age <- !is.na(demographics[["age"]])
  random_noise <- sample(-5L:5L, nrow(demographics), replace = TRUE)
  demographics[has_age, age := as_integer_age(age + random_noise)]
  
  if ("coding" %in% names(demographics)) {
    coding_lower <- tolower(as.character(demographics[["coding"]]))
    sex_or_ethnicity <- grepl("sex|ethnicity|ethnic", coding_lower)
    demographics[sex_or_ethnicity, age := 0L]
  }
  
  demographics
}

round_existing_age_column <- function(dt, file_label) {
  dt <- as.data.table(dt)
  if (!"age" %in% names(dt)) {
    stop(file_label, " must contain an existing age column.")
  }
  
  dt[, age := round_integer_age(age)]
  dt
}

clean_ukb_bulk <- function(ukb_bulk, age_of_recruitment_matrix) {
  ukb_bulk <- as.data.table(ukb_bulk)
  ukb_bulk <- add_age_from_recruitment(ukb_bulk, age_of_recruitment_matrix)
  
  helper_names <- c(
    "approx_birth_date",
    "month_of_birth",
    "age_of_assessment",
    "age_of_recruitment",
    "year_of_birth"
  )
  
  if ("coding" %in% names(ukb_bulk)) {
    ukb_bulk <- ukb_bulk[!coding %in% helper_names]
  } else {
    cols_to_remove <- intersect(helper_names, names(ukb_bulk))
    if (length(cols_to_remove)) {
      ukb_bulk[, (cols_to_remove) := NULL]
    }
  }
  
  ukb_bulk
}

save_age_rds <- function(dt, out_path, label) {
  saveRDS(dt, out_path, compress = FALSE)
  cat("\n", label, "\n", sep = "")
  cat("Rows:", nrow(dt), "\n")
  cat("Rows with non-missing age:", dt[!is.na(age), .N], "\n")
  cat("Saved:", out_path, "\n")
}

ensure_chapter_column <- function(dt) {
  dt <- as.data.table(dt)
  if (!"chapter" %in% names(dt)) {
    dt[, chapter := NA_character_]
  }
  dt
}

# ---- Build lookup matrices --------------------------------------------------

ukb_bulk <- as.data.table(readRDS(ukb_bulk_path))

age_of_recruitment_matrix <- make_age_of_recruitment_matrix(ukb_bulk)
approx_birth_date_matrix <- make_approx_birth_date_matrix(ukb_bulk)

cat("age_of_recruitment_matrix rows:", nrow(age_of_recruitment_matrix), "\n")
cat("approx_birth_date_matrix rows:", nrow(approx_birth_date_matrix), "\n")

# ---- blood_biochemistry.rds -------------------------------------------------

blood_biochemistry <- readRDS(blood_biochemistry_path)
blood_biochemistry <- add_age_from_recruitment(blood_biochemistry, age_of_recruitment_matrix)
save_age_rds(blood_biochemistry, blood_biochemistry_out_path, "blood_biochemistry.rds")

# ---- clinical_history_icd.rds ----------------------------------------------

clinical_history_icd <- readRDS(clinical_history_icd_path)
clinical_history_icd <- add_age_from_event_date(
  clinical_history_icd,
  approx_birth_date_matrix,
  date_col = "date",
  remove_date = TRUE
)
save_age_rds(clinical_history_icd, clinical_history_icd_out_path, "clinical_history_icd.rds")

# ---- demographics.rds -------------------------------------------------------

demographics <- readRDS(demographics_path)
demographics <- add_noisy_demographics_age(demographics, age_of_recruitment_matrix)
save_age_rds(demographics, demographics_out_path, "demographics.rds")

# ---- seq_cancer_self_reported_with_chapter.rds ------------------------------

self_reported_cancer <- readRDS(self_reported_cancer_path)
self_reported_cancer <- round_existing_age_column(
  self_reported_cancer,
  "seq_cancer_self_reported_with_chapter.rds"
)
save_age_rds(
  self_reported_cancer,
  self_reported_cancer_out_path,
  "seq_cancer_self_reported_with_chapter.rds"
)

# ---- seq_noncancer_self_reported_with_chapter.rds ---------------------------

self_reported_noncancer <- readRDS(self_reported_noncancer_path)
self_reported_noncancer <- round_existing_age_column(
  self_reported_noncancer,
  "seq_noncancer_self_reported_with_chapter.rds"
)
save_age_rds(
  self_reported_noncancer,
  self_reported_noncancer_out_path,
  "seq_noncancer_self_reported_with_chapter.rds"
)

# ---- ukb_bulk.rds -----------------------------------------------------------

ukb_bulk_clean <- clean_ukb_bulk(ukb_bulk, age_of_recruitment_matrix)
save_age_rds(ukb_bulk_clean, ukb_bulk_out_path, "ukb_bulk.rds")

# ---- final.rds --------------------------------------------------------------

final <- rbindlist(
  list(
    ensure_chapter_column(blood_biochemistry),
    ensure_chapter_column(clinical_history_icd),
    ensure_chapter_column(demographics),
    ensure_chapter_column(self_reported_cancer),
    ensure_chapter_column(self_reported_noncancer),
    ensure_chapter_column(ukb_bulk_clean)
  ),
  use.names = TRUE,
  fill = TRUE
)

saveRDS(final, final_out_path, compress = FALSE)

cat("\nfinal.rds\n")
cat("Rows:", nrow(final), "\n")
cat("Rows with non-missing age:", final[!is.na(age), .N], "\n")
cat("Saved:", final_out_path, "\n")

cat("\nDone. Updated RDS files are in:", output_dir, "\n")
