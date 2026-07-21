# =============================================================================
# Delphi Preprocessing (multi-variant): Build several Delphi/data/<variant>/
# folders from the age-annotated source RDS files, each producing:
#
#   train.bin, val.bin, labels.csv, config_values.py
#   (plus token_dictionary.csv, readable_event_rows.csv,
#    model_event_rows_readable.csv, vocab_meta.rds for inspection/reuse)
#
# This re-uses the same token-building logic as scripts/delphi_preprocess.R,
# applied to progressively larger subsets of sources:
#
#   1. ukb_icd_only           : clinical_history_icd
#   2. ukb_icd_demographics   : clinical_history_icd + demographics
#   3. ukb_icd_demographics_bulk
#                              : clinical_history_icd + demographics + ukb_bulk
#   4. ukb_biomarkers_q4      : clinical_history_icd + demographics + ukb_bulk
#                               + blood_biochemistry (binned into quartiles Q1-Q4)
#   5. ukb_amk125             : clinical_history_icd + demographics + ukb_bulk
#                               + blood_biochemistry + self-reported cancer and
#                               non-cancer conditions (with ICD chapters)
#
# config_values.py contains `vocab_size` and `ignore_tokens`, and is meant to be
# picked up from a Delphi training config via:
#   exec(open(os.path.join("data", dataset, "config_values.py")).read())
# =============================================================================

library(data.table)

# --- Paths -------------------------------------------------------------------
base_dir         <- "/rds/general/project/hda_24-25/live/amk125_thesis"
age_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/outputs_with_age"
delphi_data_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/Delphi/data"

val_frac     <- 0.05
n_value_bins <- 4L
set.seed(42)

# --- Settings shared across all variants -------------------------------------
SEX_CODING  <- "sex"
FEMALE_CODE <- "Female"
MALE_CODE   <- "Male"

# Codings whose raw `code` value should be kept as-is (not value-binned),
# because they are disease/diagnosis style categorical codes.
disease_codings <- c("ICD10", "self_reported_cancer", "self_reported_non_cancer")

ensure_chapter_column <- function(dt) {
  dt <- as.data.table(dt)
  if (!"chapter" %in% names(dt)) {
    dt[, chapter := NA_character_]
  }
  dt
}

# --- Load each source once, add a chapter column if missing ------------------
cat("Loading source RDS files...\n")
blood_biochemistry <- ensure_chapter_column(readRDS(file.path(age_dir, "blood_biochemistry.rds")))
clinical_history_icd <- ensure_chapter_column(readRDS(file.path(age_dir, "clinical_history_icd.rds")))
demographics <- ensure_chapter_column(readRDS(file.path(age_dir, "demographics.rds")))
seq_cancer_self_reported <- ensure_chapter_column(readRDS(file.path(age_dir, "seq_cancer_self_reported_with_chapter.rds")))
seq_noncancer_self_reported <- ensure_chapter_column(readRDS(file.path(age_dir, "seq_noncancer_self_reported_with_chapter.rds")))
ukb_bulk <- ensure_chapter_column(readRDS(file.path(age_dir, "ukb_bulk.rds")))
cat("Done loading sources.\n\n")

# --- Core pipeline: combined data.table -> Delphi bins + config_values.py ----
build_delphi_dataset <- function(sources, out_dir, val_frac, n_value_bins, disease_codings) {

  tag <- basename(out_dir)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  dat <- rbindlist(sources, use.names = TRUE, fill = TRUE)
  cat("[", tag, "] Rows combined:", nrow(dat), " | Patients:", uniqueN(dat[["eid"]]), "\n")

  required_cols <- c("eid", "coding", "code", "age")
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }
  if (!"chapter" %in% names(dat)) {
    dat[, chapter := NA_character_]
  }

  dat <- dat[!is.na(eid) & !is.na(coding) & !is.na(code) & !is.na(age)]
  cat("[", tag, "] Rows after dropping missing eid/coding/code/age:", nrow(dat), "\n")

  dat[, eid := as.character(eid)]
  dat[, coding := as.character(coding)]
  dat[, code_raw := as.character(code)]
  dat[, age_years := as.numeric(age)]
  dat <- dat[!is.na(age_years)]

  dat[, age_days := as.integer(round(age_years * 365.25))]
  dat[, age_days := pmax(0L, age_days)]

  dat[, source_type := fifelse(
    is.na(chapter) | chapter == "" | chapter == "NA",
    "blood_biochemistry",
    as.character(chapter)
  )]

  readable_event_rows <- dat[, .(
    eid,
    age_days,
    token_wording = paste0(source_type, "::", coding),
    token_value = code_raw
  )]
  setorder(readable_event_rows, eid, age_days, token_wording, token_value)
  fwrite(readable_event_rows, file.path(out_dir, "readable_event_rows.csv"))
  cat("[", tag, "] Written readable_event_rows.csv with", nrow(readable_event_rows), "rows\n")

  dat[, numeric_value := suppressWarnings(as.numeric(code_raw))]
  dat[, is_sex := coding == SEX_CODING & code_raw %in% c(FEMALE_CODE, MALE_CODE)]
  dat[, is_disease := coding %in% disease_codings]
  dat[, is_continuous := !is_sex & !is_disease & !is.na(numeric_value)]

  dat[is_continuous == TRUE, value_bin := {
    ranks <- frank(numeric_value, ties.method = "average", na.last = "keep")
    bins <- ceiling(ranks / .N * n_value_bins)
    paste0("Q", pmin(pmax(as.integer(bins), 1L), n_value_bins))
  }, by = coding]

  dat[is_sex == TRUE, token_wording_model := paste0("demographics::", code_raw)]
  dat[is_disease == TRUE, token_wording_model := paste0(coding, "::", code_raw)]
  dat[is_continuous == TRUE, token_wording_model := paste0(source_type, "::", coding, "::", value_bin)]
  dat[is.na(token_wording_model), token_wording_model := paste0(source_type, "::", coding, "::", code_raw)]

  custom_vocab <- unique(dat[is_sex == FALSE, .(
    token_wording = token_wording_model,
    source_type,
    coding,
    value_bin = fifelse(is.na(value_bin), "", value_bin)
  )])
  setorder(custom_vocab, source_type, coding, value_bin, token_wording)
  custom_vocab[, token_id := .I + 3L]

  token_dictionary <- rbind(
    data.table(
      token_id = c(0L, 1L, 2L, 3L),
      token_wording = c("Padding", "No event", "demographics::Female", "demographics::Male"),
      source_type = c("reserved", "reserved", "demographics", "demographics"),
      coding = c("padding", "no_event", SEX_CODING, SEX_CODING),
      value_bin = c("", "", "", "")
    ),
    custom_vocab[, .(token_id, token_wording, source_type, coding, value_bin)]
  )
  setorder(token_dictionary, token_id)
  fwrite(token_dictionary, file.path(out_dir, "token_dictionary.csv"))
  cat("[", tag, "] Written token_dictionary.csv with", nrow(token_dictionary), "rows\n")

  dat <- merge(
    dat,
    custom_vocab[, .(token_wording_model = token_wording, token_id)],
    by = "token_wording_model",
    all.x = TRUE
  )
  dat[is_sex == TRUE & code_raw == FEMALE_CODE, token_id := 2L]
  dat[is_sex == TRUE & code_raw == MALE_CODE, token_id := 3L]

  dat <- dat[!is.na(token_id)]
  dat[, token_id := as.integer(token_id)]
  dat[, bin_token_id := token_id - 1L]
  cat("[", tag, "] Rows after token assignment:", nrow(dat), "\n")

  model_event_rows_readable <- dat[, .(
    eid,
    age_days,
    token_wording = token_wording_model,
    token_value = code_raw,
    model_token_id = token_id,
    bin_token_id
  )]
  setorder(model_event_rows_readable, eid, age_days, token_wording, token_value)
  fwrite(model_event_rows_readable, file.path(out_dir, "model_event_rows_readable.csv"))

  setorder(dat, eid, age_days, token_id)

  patients <- unique(dat$eid)
  n_val <- as.integer(round(val_frac * length(patients)))
  val_eids <- sample(patients, n_val)
  cat("[", tag, "] Train patients:", length(patients) - n_val, " | Val patients:", n_val, "\n")

  train_dat <- copy(dat[!eid %in% val_eids])
  val_dat <- copy(dat[ eid %in% val_eids])

  train_dat[, patient_id := .GRP - 1L, by = eid]
  val_dat[, patient_id := .GRP - 1L, by = eid]

  setorder(train_dat, patient_id, age_days, token_id)
  setorder(val_dat, patient_id, age_days, token_id)

  write_bin <- function(dt, path) {
    m <- as.matrix(dt[, .(patient_id, age_days, bin_token_id)])
    storage.mode(m) <- "integer"
    con <- file(path, "wb")
    on.exit(close(con), add = TRUE)
    writeBin(as.integer(t(m)), con, size = 4L, endian = "little")
    cat("[", tag, "] Written:", path, " | Rows:", nrow(dt), "\n")
  }

  write_bin(train_dat, file.path(out_dir, "train.bin"))
  write_bin(val_dat, file.path(out_dir, "val.bin"))

  labels <- token_dictionary[, .(event_name = token_wording)]
  fwrite(labels, file.path(out_dir, "labels.csv"))
  cat("[", tag, "] Written labels.csv with", nrow(labels), "rows\n")

  vocab_size <- max(token_dictionary$token_id) + 1L
  ignore_tokens_vec <- c(
    0L, 1L, 2L, 3L,
    custom_vocab[!coding %in% disease_codings, token_id]
  )
  ignore_tokens_vec <- sort(unique(as.integer(ignore_tokens_vec)))

  saveRDS(
    list(
      vocab_size = vocab_size,
      ignore_tokens = ignore_tokens_vec,
      token_dictionary = token_dictionary,
      disease_codings = disease_codings,
      n_value_bins = n_value_bins
    ),
    file.path(out_dir, "vocab_meta.rds")
  )

  # --- config_values.py -------------------------------------------------------
  # Picked up via exec(open(...).read()) from Delphi training configs, e.g.
  #   config_values_path <- file.path("data", dataset, "config_values.py")
  #   exec(open(config_values_path).read())
  config_lines <- c(
    "# Auto-generated by scripts/build_delphi_data_variants.R -- do not edit by hand.",
    paste0("vocab_size = ", vocab_size),
    paste0(
      "ignore_tokens = [",
      paste(ignore_tokens_vec, collapse = ", "),
      "]"
    )
  )
  writeLines(config_lines, file.path(out_dir, "config_values.py"))
  cat("[", tag, "] Written config_values.py (vocab_size =", vocab_size,
      ", ignore_tokens length =", length(ignore_tokens_vec), ")\n")

  cat("[", tag, "] Done.\n\n")
  invisible(NULL)
}

# --- Variant 1: ICD only -------------------------------------------------------
build_delphi_dataset(
  sources = list(clinical_history_icd),
  out_dir = file.path(delphi_data_dir, "ukb_icd_only"),
  val_frac = val_frac,
  n_value_bins = n_value_bins,
  disease_codings = disease_codings
)

# --- Variant 2: ICD + demographics --------------------------------------------
build_delphi_dataset(
  sources = list(clinical_history_icd, demographics),
  out_dir = file.path(delphi_data_dir, "ukb_icd_demographics"),
  val_frac = val_frac,
  n_value_bins = n_value_bins,
  disease_codings = disease_codings
)

# --- Variant 3: ICD + demographics + ukb_bulk ---------------------------------
build_delphi_dataset(
  sources = list(clinical_history_icd, demographics, ukb_bulk),
  out_dir = file.path(delphi_data_dir, "ukb_icd_demographics_bulk"),
  val_frac = val_frac,
  n_value_bins = n_value_bins,
  disease_codings = disease_codings
)

# --- Variant 4: ICD + demographics + ukb_bulk + blood biochemistry (Q1-Q4) ----
build_delphi_dataset(
  sources = list(clinical_history_icd, demographics, ukb_bulk, blood_biochemistry),
  out_dir = file.path(delphi_data_dir, "ukb_biomarkers_q4"),
  val_frac = val_frac,
  n_value_bins = n_value_bins,
  disease_codings = disease_codings
)

# --- Variant 5: everything, incl. self-reported cancer/non-cancer w/ chapters -
build_delphi_dataset(
  sources = list(
    clinical_history_icd, demographics, ukb_bulk, blood_biochemistry,
    seq_cancer_self_reported, seq_noncancer_self_reported
  ),
  out_dir = file.path(delphi_data_dir, "ukb_amk125"),
  val_frac = val_frac,
  n_value_bins = n_value_bins,
  disease_codings = disease_codings
)

cat("All variants complete.\n")
