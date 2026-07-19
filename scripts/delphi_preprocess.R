# =============================================================================
# Delphi Preprocessing: Build Delphi binary datasets from individual source files
#
# Reads each source RDS (output of 9-adding_age.R) individually and builds
# five cumulative data configurations, each in its own output folder:
#
#   ukb_amk125_clinical_icd
#   ukb_amk125_clinical_demographics_icd
#   ukb_amk125_clinical_demographics_ukb_icd
#   ukb_amk125_clinical_demographics_ukb_biochem_icd
#   ukb_amk125_clinical_demographics_ukb_biochem_icd_self_reported
#
# Each folder contains:
#   train.bin / val.bin / test.bin      (60 / 20 / 20 patient split)
#   token_dictionary.csv
#   labels.csv
#   readable_event_rows.csv
#   model_event_rows_readable.csv
#   vocab_meta.rds
#   config_values.py                    (read by Delphi training configs)
#
# The cohort and split are anchored to patients with clinical ICD history and
# applied consistently across all configurations. Extra sources therefore
# cannot add participants or move them between splits.
#
# Important Delphi detail:
#   utils.py adds +1 to token IDs inside get_batch().
#   Therefore the token_id written to *.bin must be one lower than
#   the model-facing token_id in token_dictionary.csv.
#
#   Model-facing token convention after utils.py adds +1:
#     0 = padding
#     1 = no_event
#     2 = Female
#     3 = Male
#     4+ = custom event tokens
# =============================================================================

library(data.table)

set.seed(42L)

# --- Paths -------------------------------------------------------------------
sources_dir <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/outputs_with_age"
out_base    <- "/rds/general/project/hda_24-25/live/amk125_thesis/Delphi/data"

# --- Global settings ---------------------------------------------------------
TRAIN_FRAC <- 0.60
VAL_FRAC   <- 0.20
TEST_FRAC  <- 0.20   # implied: 1 - TRAIN_FRAC - VAL_FRAC

SEX_CODING  <- "sex"
FEMALE_CODE <- "Female"
MALE_CODE   <- "Male"

# Disease/code events: the code itself becomes part of the token (e.g. ICD10::E11).
# Continuous numeric values are NOT binned for these codings.
disease_codings <- c("ICD10", "self_reported_cancer", "self_reported_non_cancer")

# Number of quantile bins for continuous measurements (e.g. albumin).
n_value_bins <- 4L

# --- Helper ------------------------------------------------------------------
ensure_chapter_column <- function(dt) {
  dt <- as.data.table(dt)
  if (!"chapter" %in% names(dt)) dt[, chapter := NA_character_]
  dt
}

# --- Load individual source files --------------------------------------------
cat("Loading source files from", sources_dir, "...\n")

blood_biochemistry     <- ensure_chapter_column(readRDS(file.path(sources_dir, "blood_biochemistry.rds")))
clinical_history_icd   <- ensure_chapter_column(readRDS(file.path(sources_dir, "clinical_history_icd.rds")))
demographics           <- ensure_chapter_column(readRDS(file.path(sources_dir, "demographics.rds")))
self_reported_cancer   <- ensure_chapter_column(readRDS(file.path(sources_dir, "seq_cancer_self_reported_with_chapter.rds")))
self_reported_noncancer <- ensure_chapter_column(readRDS(file.path(sources_dir, "seq_noncancer_self_reported_with_chapter.rds")))
ukb_bulk               <- ensure_chapter_column(readRDS(file.path(sources_dir, "ukb_bulk.rds")))

cat("All source files loaded.\n\n")

# --- Define configurations ---------------------------------------------------
# Each entry is a named list of data.tables to combine for that configuration.
configs <- list(
  ukb_amk125_clinical_icd = list(clinical_history_icd),
  ukb_amk125_clinical_demographics_icd = list(
    clinical_history_icd, demographics
  ),
  ukb_amk125_clinical_demographics_ukb_icd = list(
    clinical_history_icd, demographics, ukb_bulk
  ),
  ukb_amk125_clinical_demographics_ukb_biochem_icd = list(
    clinical_history_icd, demographics, ukb_bulk, blood_biochemistry
  ),
  ukb_amk125_clinical_demographics_ukb_biochem_icd_self_reported = list(
    clinical_history_icd, demographics, ukb_bulk, blood_biochemistry,
    self_reported_cancer, self_reported_noncancer
  )
)

# --- Compute shared 60/20/20 patient split -----------------------------------
# Clinical ICD history defines the comparison cohort. This prevents optional
# sources (especially self-report) from changing the evaluated population.
clinical_cohort <- as.data.table(clinical_history_icd)[
  !is.na(eid) & !is.na(coding) & !is.na(code) & !is.na(age)
]
clinical_cohort[, age_numeric := suppressWarnings(as.numeric(age))]
all_eids <- sort(unique(as.character(
  clinical_cohort[!is.na(age_numeric), eid]
)))
if (length(all_eids) == 0L) stop("No eligible clinical ICD participants found")

n_total <- length(all_eids)
n_val   <- as.integer(round(VAL_FRAC  * n_total))
n_test  <- as.integer(round(TEST_FRAC * n_total))

shuffled   <- sample(all_eids)
val_eids   <- shuffled[seq_len(n_val)]
test_eids  <- shuffled[n_val + seq_len(n_test)]
train_eids <- shuffled[(n_val + n_test + 1L):n_total]

split_assignments <- data.table(
  eid = c(train_eids, val_eids, test_eids),
  split = c(
    rep("train", length(train_eids)),
    rep("val", length(val_eids)),
    rep("test", length(test_eids))
  )
)
setorder(split_assignments, eid)
dir.create(out_base, recursive = TRUE, showWarnings = FALSE)
fwrite(split_assignments, file.path(out_base, "split_assignments.csv"))
cat("Written shared split assignments:",
    file.path(out_base, "split_assignments.csv"), "\n")

cat(sprintf(
  "Global patient split: %d train (%.0f%%) | %d val (%.0f%%) | %d test (%.0f%%)\n\n",
  length(train_eids), 100 * length(train_eids) / n_total,
  length(val_eids),   100 * length(val_eids)   / n_total,
  length(test_eids),  100 * length(test_eids)  / n_total
))

# --- Preprocessing function --------------------------------------------------
preprocess_config <- function(cfg_name, sources_list, out_dir,
                               train_eids, val_eids, test_eids) {

  cat("=================================================================\n")
  cat("Configuration:", cfg_name, "\n")
  cat("Output:       ", out_dir, "\n")
  cat("=================================================================\n")

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  # -- Combine and clean -------------------------------------------------------
  dat <- rbindlist(sources_list, use.names = TRUE, fill = TRUE)

  required_cols <- c("eid", "coding", "code", "age")
  missing_cols  <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0L)
    stop(cfg_name, ": missing required columns: ", paste(missing_cols, collapse = ", "))

  if (!"chapter" %in% names(dat)) dat[, chapter := NA_character_]

  dat <- dat[!is.na(eid) & !is.na(coding) & !is.na(code) & !is.na(age)]

  dat[, eid      := as.character(eid)]
  dat[, coding   := as.character(coding)]
  dat[, code_raw := as.character(code)]
  dat[, age_years := suppressWarnings(as.numeric(age))]
  dat <- dat[!is.na(age_years)]

  # Enforce the shared clinical-ICD cohort for every experiment.
  dat <- dat[eid %in% all_eids]

  dat[, age_days := as.integer(round(age_years * 365.25))]
  dat[, age_days := pmax(0L, age_days)]

  dat[, source_type := fifelse(
    is.na(chapter) | chapter == "" | chapter == "NA",
    "blood_biochemistry",
    as.character(chapter)
  )]

  cat("Rows after cleaning:", nrow(dat), " | Patients:", uniqueN(dat$eid), "\n")

  # -- Readable event rows -----------------------------------------------------
  readable_event_rows <- dat[, .(
    eid,
    age_days,
    token_wording = paste0(source_type, "::", coding),
    token_value   = code_raw
  )]
  setorder(readable_event_rows, eid, age_days, token_wording, token_value)
  fwrite(readable_event_rows, file.path(out_dir, "readable_event_rows.csv"))
  cat("Written readable_event_rows.csv\n")

  # -- Token labelling ---------------------------------------------------------
  dat[, numeric_value := suppressWarnings(as.numeric(code_raw))]
  dat[, is_sex      := coding == SEX_CODING & code_raw %in% c(FEMALE_CODE, MALE_CODE)]
  dat[, is_disease  := coding %in% disease_codings]
  dat[, is_continuous := !is_sex & !is_disease & !is.na(numeric_value)]

  dat[is_continuous == TRUE, value_bin := {
    ranks <- frank(numeric_value, ties.method = "average", na.last = "keep")
    bins  <- ceiling(ranks / .N * n_value_bins)
    paste0("Q", pmin(pmax(as.integer(bins), 1L), n_value_bins))
  }, by = coding]

  dat[is_sex == TRUE,        token_wording_model := paste0("demographics::", code_raw)]
  dat[is_disease == TRUE,    token_wording_model := paste0(coding, "::", code_raw)]
  dat[is_continuous == TRUE, token_wording_model := paste0(source_type, "::", coding, "::", value_bin)]
  dat[is.na(token_wording_model),
      token_wording_model := paste0(source_type, "::", coding, "::", code_raw)]

  # -- Token vocabulary --------------------------------------------------------
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
      token_id      = c(0L, 1L, 2L, 3L),
      token_wording = c("Padding", "No event", "demographics::Female", "demographics::Male"),
      source_type   = c("reserved", "reserved", "demographics", "demographics"),
      coding        = c("padding", "no_event", SEX_CODING, SEX_CODING),
      value_bin     = c("", "", "", "")
    ),
    custom_vocab[, .(token_id, token_wording, source_type, coding, value_bin)]
  )
  setorder(token_dictionary, token_id)
  fwrite(token_dictionary, file.path(out_dir, "token_dictionary.csv"))
  cat("Written token_dictionary.csv with", nrow(token_dictionary), "tokens\n")

  # -- Assign token IDs --------------------------------------------------------
  dat <- merge(
    dat,
    custom_vocab[, .(token_wording_model = token_wording, token_id)],
    by    = "token_wording_model",
    all.x = TRUE
  )
  dat[is_sex == TRUE & code_raw == FEMALE_CODE, token_id := 2L]
  dat[is_sex == TRUE & code_raw == MALE_CODE,   token_id := 3L]
  dat <- dat[!is.na(token_id)]
  dat[, token_id    := as.integer(token_id)]
  dat[, bin_token_id := token_id - 1L]
  cat("Rows after token assignment:", nrow(dat), "\n")

  # -- Model-readable event table ----------------------------------------------
  model_event_rows_readable <- dat[, .(
    eid, age_days,
    token_wording  = token_wording_model,
    token_value    = code_raw,
    model_token_id = token_id,
    bin_token_id
  )]
  setorder(model_event_rows_readable, eid, age_days, token_wording, token_value)
  fwrite(model_event_rows_readable, file.path(out_dir, "model_event_rows_readable.csv"))
  cat("Written model_event_rows_readable.csv\n")

  # -- Sort before splitting ---------------------------------------------------
  setorder(dat, eid, age_days, token_id)

  # -- Apply the shared patient split ------------------------------------------
  # Restrict to patients present in this configuration's data
  cfg_eids <- unique(dat$eid)
  cfg_train <- intersect(train_eids, cfg_eids)
  cfg_val   <- intersect(val_eids,   cfg_eids)
  cfg_test  <- intersect(test_eids,  cfg_eids)

  train_dat <- copy(dat[eid %in% cfg_train])
  val_dat   <- copy(dat[eid %in% cfg_val])
  test_dat  <- copy(dat[eid %in% cfg_test])

  # Consecutive patient IDs within each split (Delphi uses this only for grouping)
  train_dat[, patient_id := .GRP - 1L, by = eid]
  val_dat[,   patient_id := .GRP - 1L, by = eid]
  test_dat[,  patient_id := .GRP - 1L, by = eid]

  setorder(train_dat, patient_id, age_days, token_id)
  setorder(val_dat,   patient_id, age_days, token_id)
  setorder(test_dat,  patient_id, age_days, token_id)

  cat(sprintf(
    "Patients  — train: %d | val: %d | test: %d\n",
    uniqueN(train_dat$eid), uniqueN(val_dat$eid), uniqueN(test_dat$eid)
  ))
  cat(sprintf(
    "Events    — train: %d | val: %d | test: %d\n",
    nrow(train_dat), nrow(val_dat), nrow(test_dat)
  ))

  # -- Write binary files ------------------------------------------------------
  write_bin <- function(dt, path) {
    m <- as.matrix(dt[, .(patient_id, age_days, bin_token_id)])
    storage.mode(m) <- "integer"
    con <- file(path, "wb")
    on.exit(close(con), add = TRUE)
    writeBin(as.integer(t(m)), con, size = 4L, endian = "little")
    cat("Written:", path, "(", nrow(dt), "rows )\n")
  }

  write_bin(train_dat, file.path(out_dir, "train.bin"))
  write_bin(val_dat,   file.path(out_dir, "val.bin"))
  write_bin(test_dat,  file.path(out_dir, "test.bin"))

  # -- Labels ------------------------------------------------------------------
  labels <- token_dictionary[, .(event_name = token_wording)]
  fwrite(labels, file.path(out_dir, "labels.csv"))
  cat("Written labels.csv\n")

  # -- Vocabulary metadata (rds) -----------------------------------------------
  vocab_size <- max(token_dictionary$token_id) + 1L

  # Tokens that are context-only (not disease targets) are excluded from the loss
  ignore_tokens_vec <- sort(unique(as.integer(c(
    0L, 1L, 2L, 3L,
    custom_vocab[!coding %in% disease_codings, token_id]
  ))))

  saveRDS(
    list(
      vocab_size       = vocab_size,
      ignore_tokens    = ignore_tokens_vec,
      token_dictionary = token_dictionary,
      disease_codings  = disease_codings,
      n_value_bins     = n_value_bins
    ),
    file.path(out_dir, "vocab_meta.rds")
  )
  cat("Written vocab_meta.rds\n")

  # -- config_values.py (read by Delphi training configs) ----------------------
  ignore_tokens_py <- paste0("[", paste(ignore_tokens_vec, collapse = ", "), "]")
  writeLines(
    c(
      paste0("vocab_size = ",    vocab_size),
      paste0("ignore_tokens = ", ignore_tokens_py)
    ),
    file.path(out_dir, "config_values.py")
  )
  cat("Written config_values.py  (vocab_size =", vocab_size, ")\n")
  cat("\n")
}

# --- Run all configurations --------------------------------------------------
for (cfg_name in names(configs)) {
  out_dir <- file.path(out_base, cfg_name)
  preprocess_config(
    cfg_name    = cfg_name,
    sources_list = configs[[cfg_name]],
    out_dir     = out_dir,
    train_eids  = train_eids,
    val_eids    = val_eids,
    test_eids   = test_eids
  )
}

cat("All configurations complete.\n")
cat("Output root:", out_base, "\n")
