# Adds a chapter column to self-reported sequence RDS files.
#
# By default this writes new files:
#   seq_cancer_self_reported_with_chapter.rds
#   seq_noncancer_self_reported_with_chapter.rds
#
# Set overwrite_inputs <- TRUE below if you want to replace the original files.

library(data.table)

tmp_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/tmp"
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)
Sys.setenv(TMPDIR = tmp_dir)

out_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sources"

cancer_path           <- file.path(out_dir, "seq_cancer_self_reported.rds")
noncancer_path        <- file.path(out_dir, "seq_noncancer_self_reported.rds")

mapping_csv_candidates <- file.path("/rds/general/project/hda_24-25/live/amk125_thesis/scripts/icd10-mapping.csv")

mapping_csv_path <- mapping_csv_candidates[file.exists(mapping_csv_candidates)][1]

cancer_chapter <- "Chapter II Neoplasms"

overwrite_inputs <- FALSE

cancer_out_path <- if (overwrite_inputs) {
  cancer_path
} else {
  file.path(out_dir, "seq_cancer_self_reported_with_chapter.rds")
}

noncancer_out_path <- if (overwrite_inputs) {
  noncancer_path
} else {
  file.path(out_dir, "seq_noncancer_self_reported_with_chapter.rds")
}


clean_category_name <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x <- gsub("[[:space:]]+", " ", x)
  x <- gsub("[[:space:]]*/[[:space:]]*", "/", x)
  x <- gsub("[[:space:]]*\\+/-[[:space:]]*", "+/-", x)
  x <- gsub("[[:space:]]*&[[:space:]]*", " & ", x)
  x <- gsub("-$", "", x)
  x <- gsub("[0-9]+$", "", x)
  trimws(x)
}

is_zip_file <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  identical(readBin(con, "raw", n = 2), charToRaw("PK"))
}

load_name_chapter_mapping <- function(path) {
  if (is.na(path) || !file.exists(path)) {
    stop(
      "No ICD chapter mapping CSV found. Save/export it as ",
      file.path(out_dir, "icd10-mapping.csv"),
      " with columns like value and suggested_icd10_chapter."
    )
  }
  
  if (is_zip_file(path)) {
    stop(
      "The mapping file at ", path, " is an Apple Numbers file, not a plain CSV. ",
      "In Numbers, use File > Export To > CSV, then save it as ",
      file.path(out_dir, "icd10-mapping.csv"),
      "."
    )
  }
  
  cat("Loading ICD chapter mapping CSV:", path, "\n")
  mapping <- fread(path)
  setnames(mapping, names(mapping), tolower(gsub("[^A-Za-z0-9]+", "_", names(mapping))))
  
  name_col <- intersect(c("value", "category", "name", "code", "illness", "condition"), names(mapping))[1]
  chapter_col <- intersect(
    c("suggested_icd10_chapter", "icd10_chapter", "chapter", "chapter_name"),
    names(mapping)
  )[1]
  
  if (is.na(name_col) || is.na(chapter_col)) {
    stop(
      "Mapping CSV must contain a disease/category name column and chapter column. ",
      "Expected columns like value and suggested_icd10_chapter. Found: ",
      paste(names(mapping), collapse = ", ")
    )
  }
  
  if ("is_heading" %in% names(mapping)) {
    mapping <- mapping[
      is.na(is_heading) |
        !(tolower(as.character(is_heading)) %chin% c("true", "t", "1", "yes", "y"))
    ]
  }
  
  mapping <- mapping[, .(
    category_name = clean_category_name(get(name_col)),
    mapped_chapter = trimws(as.character(get(chapter_col)))
  )]
  
  mapping <- mapping[
    !is.na(category_name) & category_name != "" &
      !is.na(mapped_chapter) & mapped_chapter != ""
  ]
  mapping <- unique(mapping)
  
  duplicate_names <- mapping[, .N, by = category_name][N > 1, category_name]
  if (length(duplicate_names)) {
    warning(
      "Mapping CSV has duplicate disease/category names; keeping the first for: ",
      paste(head(duplicate_names, 30), collapse = ", ")
    )
    mapping <- mapping[, .SD[1], by = category_name]
  }
  
  cat("Mapping rows loaded:", nrow(mapping), "\n")
  mapping
}

load_sequence <- function(input_path) {
  cat("\nLoading sequence file:", input_path, "\n")
  seq_dt <- as.data.table(readRDS(input_path))
  
  if (!"code" %in% names(seq_dt)) {
    stop(input_path, " is missing required column: code")
  }
  
  if ("chapter" %in% names(seq_dt)) {
    seq_dt[, chapter := NULL]
  }
  
  cat("Rows:", nrow(seq_dt), "\n")
  seq_dt
}

build_cancer_mapping <- function(seq_dt) {
  mapping <- unique(seq_dt[, .(code = as.character(code))])
  mapping <- mapping[!is.na(code) & code != ""]
  mapping[, code_num := suppressWarnings(as.integer(code))]
  mapping[, ukb_meaning := code]
  mapping[, chapter := cancer_chapter]
  setorder(mapping, code_num, code)
  mapping[, .(code, code_num, ukb_meaning, chapter)]
}

build_noncancer_mapping <- function(seq_dt, csv_mapping) {
  mapping <- unique(seq_dt[, .(code = as.character(code))])
  mapping <- mapping[!is.na(code) & code != ""]
  mapping[, code_num := suppressWarnings(as.integer(code))]
  mapping[, category_name := clean_category_name(code)]
  mapping[, ukb_meaning := category_name]
  mapping[, chapter := NA_character_]
  mapping[csv_mapping, chapter := i.mapped_chapter, on = "category_name"]
  
  unmapped <- mapping[is.na(chapter), category_name]
  if (length(unmapped)) {
    warning(
      "Non-cancer categories missing from mapping CSV: ",
      paste(head(unmapped, 50), collapse = ", ")
    )
  }
  
  setorder(mapping, code_num, code)
  mapping[, .(code, code_num, category_name, ukb_meaning, chapter)]
}

apply_code_mapping <- function(seq_dt, mapping, source_label) {
  if ("chapter" %in% names(seq_dt)) {
    seq_dt[, chapter := NULL]
  }
  
  mapping_keyed <- copy(mapping)
  mapping_keyed <- unique(mapping_keyed[, .(code = as.character(code), chapter)])
  seq_dt[, code := as.character(code)]
  
  seq_dt <- mapping_keyed[seq_dt, on = "code"]
  setcolorder(seq_dt, c(setdiff(names(seq_dt), "chapter"), "chapter"))
  
  unmatched <- seq_dt[is.na(chapter), unique(code)]
  if (length(unmatched)) {
    warning(
      source_label,
      " codes without chapter mapping: ",
      paste(head(unmatched, 50), collapse = ", ")
    )
  }
  
  cat(source_label, "rows with chapter:", seq_dt[!is.na(chapter), .N], "of", nrow(seq_dt), "\n")
  
  seq_dt
}

name_chapter_mapping <- load_name_chapter_mapping(mapping_csv_path)

seq_cancer_with_chapter <- load_sequence(cancer_path)
seq_noncancer_with_chapter <- load_sequence(noncancer_path)

cancer_code_chapter_mapping <- build_cancer_mapping(seq_cancer_with_chapter)
seq_cancer_with_chapter <- apply_code_mapping(
  seq_cancer_with_chapter,
  cancer_code_chapter_mapping,
  "Cancer"
)

noncancer_code_chapter_mapping <- build_noncancer_mapping(
  seq_noncancer_with_chapter,
  name_chapter_mapping
)
seq_noncancer_with_chapter <- apply_code_mapping(
  seq_noncancer_with_chapter,
  noncancer_code_chapter_mapping,
  "Non-cancer"
)

saveRDS(seq_cancer_with_chapter, cancer_out_path, compress = FALSE)
saveRDS(seq_noncancer_with_chapter, noncancer_out_path, compress = FALSE)


cat("\nCancer chapter counts:\n")
print(seq_cancer_with_chapter[, .N, by = chapter][order(-N)])

cat("\nNon-cancer chapter counts:\n")
print(seq_noncancer_with_chapter[, .N, by = chapter][order(-N)])


# ==================================================================
# 5. Remove self-reported records duplicated in clinical_history_icd
#    Deduplication key: eid + chapter within a 3-month (90-day) window
# ==================================================================

cat("\nDeduplicating self-reported data against clinical_history_icd (3-month window)...\n")

# Load clinical_history_icd
clinical_hist_path <- file.path(out_dir, "clinical_history_icd.rds")
cat("Loading clinical_history_icd from:", clinical_hist_path, "\n")
clinical_icd <- as.data.table(readRDS(clinical_hist_path))

# Ensure date column is a proper Date object
clinical_icd[, icd_date := as.Date(date)]

# Build (eid, chapter, icd_date) lookup from clinical data (drop rows with NA chapter or date)
clinical_keys <- unique(clinical_icd[!is.na(chapter) & !is.na(icd_date), .(eid, chapter, icd_date)])
cat("Unique (eid, chapter, date) clinical keys:", nrow(clinical_keys), "\n")

# Load demographics to get birth year per eid
demog_path <- file.path(out_dir, "demographics.rds")
demog <- as.data.table(readRDS(demog_path))
birth_yr <- demog[, .(eid, birth_year = year_of_birth)]

# Helper: convert fractional age (years since birth) to an approximate Date.
# birth_year is an integer year; age is fractional years at UKB assessment.
# We approximate: date = as.Date(paste0(birth_year, "-01-01")) + round(age * 365.25)
age_to_date <- function(birth_year, age) {
  as.Date(paste0(as.integer(birth_year), "-01-01")) + round(age * 365.25)
}

# --- Deduplicate cancer self-reported data ---
cat("\nDeduplicating cancer self-reported data...\n")
cat("Rows before dedup:", nrow(seq_cancer_with_chapter), "\n")

sr_cancer <- copy(seq_cancer_with_chapter)
sr_cancer <- merge(sr_cancer, birth_yr, by = "eid", all.x = TRUE)
sr_cancer[, sr_date := age_to_date(birth_year, age)]

# For each self-reported record, check if there is a clinical ICD record
# for the same (eid, chapter) within +/-90 days
# Use a non-equi join: for each sr row, find any clinical row with
#   same eid & chapter, and icd_date in [sr_date - 90, sr_date + 90]
sr_cancer[, sr_date_lo := sr_date - 90L]
sr_cancer[, sr_date_hi := sr_date + 90L]

setkey(clinical_keys, eid, chapter, icd_date)

matched_cancer <- clinical_keys[sr_cancer,
  on = .(eid = eid, chapter = chapter, icd_date >= sr_date_lo, icd_date <= sr_date_hi),
  nomatch = 0L,
  .(eid, chapter, age, coding, sr_date)]

# Rows to REMOVE: those that matched a clinical record within 90 days
cancer_dup_keys <- unique(matched_cancer[, .(eid, chapter, age)])
cat("Cancer rows matched within 3-month window (to remove):", nrow(cancer_dup_keys), "\n")

seq_cancer_dedup <- sr_cancer[!cancer_dup_keys, on = .(eid, chapter, age)]
# Drop helper columns
seq_cancer_dedup[, c("birth_year", "sr_date", "sr_date_lo", "sr_date_hi") := NULL]
cat("Cancer rows after dedup:", nrow(seq_cancer_dedup), "\n")

# --- Deduplicate noncancer self-reported data ---
cat("\nDeduplicating noncancer self-reported data...\n")
cat("Rows before dedup:", nrow(seq_noncancer_with_chapter), "\n")

sr_noncancer <- copy(seq_noncancer_with_chapter)
sr_noncancer <- merge(sr_noncancer, birth_yr, by = "eid", all.x = TRUE)
sr_noncancer[, sr_date := age_to_date(birth_year, age)]
sr_noncancer[, sr_date_lo := sr_date - 90L]
sr_noncancer[, sr_date_hi := sr_date + 90L]

matched_noncancer <- clinical_keys[sr_noncancer,
  on = .(eid = eid, chapter = chapter, icd_date >= sr_date_lo, icd_date <= sr_date_hi),
  nomatch = 0L,
  .(eid, chapter, age, coding, sr_date)]

noncancer_dup_keys <- unique(matched_noncancer[, .(eid, chapter, age)])
cat("Noncancer rows matched within 3-month window (to remove):", nrow(noncancer_dup_keys), "\n")

seq_noncancer_dedup <- sr_noncancer[!noncancer_dup_keys, on = .(eid, chapter, age)]
seq_noncancer_dedup[, c("birth_year", "sr_date", "sr_date_lo", "sr_date_hi") := NULL]
cat("Noncancer rows after dedup:", nrow(seq_noncancer_dedup), "\n")

# ==================================================================
# 6. Save deduplicated self-reported data
# ==================================================================

cat("\nSaving deduplicated self-reported datasets...\n")

cancer_dedup_path    <- file.path(out_dir, "seq_cancer_self_reported_dedup.rds")
noncancer_dedup_path <- file.path(out_dir, "seq_noncancer_self_reported_dedup.rds")

saveRDS(seq_cancer_dedup,    cancer_dedup_path,    compress = FALSE)
saveRDS(seq_noncancer_dedup, noncancer_dedup_path, compress = FALSE)

cat("Saved cancer dedup to:",    cancer_dedup_path,    "\n")
cat("Saved noncancer dedup to:", noncancer_dedup_path, "\n")
cat("\nDone. Deduplication complete.\n")
