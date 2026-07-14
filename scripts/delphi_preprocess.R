# =============================================================================
# Delphi Preprocessing: Convert final.rds to readable event rows and Delphi bins
#
# Input expected columns:
#   eid      - patient identifier
#   coding   - event / variable name, e.g. albumin, ICD10, sex
#   code     - value / code / label, e.g. 49.12, E11, Female
#   age      - age in YEARS at event/measurement
#   chapter  - optional source/group; can be NA
#
# Main outputs:
#   readable_event_rows.csv
#     eid | age_days | token_wording | token_value
#
#   token_dictionary.csv
#     token_id | token_wording | source_type | coding | value_bin
#
#   train.bin / val.bin
#     Delphi numeric rows: patient_id | age_days | token_id
#
# Important Delphi detail:
#   utils.py adds +1 to token IDs inside get_batch().
#   Therefore the token_id written to train.bin/val.bin must be one lower than
#   the model-facing token_id in token_dictionary.csv.
#
#   Example:
#     token_dictionary.csv says Female is token_id 2.
#     train.bin stores Female as token_id 1.
#     get_batch() adds +1, so the model receives token_id 2.
#
# Model-facing token convention after utils.py adds +1:
#   0 = padding
#   1 = no_event
#   2 = Female
#   3 = Male
#   4+ = custom event tokens
# =============================================================================

library(data.table)

# --- Paths -------------------------------------------------------------------
final_path <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/outputs_with_age/final.rds"
out_dir    <- "/rds/general/project/hda_24-25/live/amk125_thesis/Delphi/data/ukb_amk125"
val_frac   <- 0.05
set.seed(42)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- Settings ----------------------------------------------------------------
SEX_CODING  <- "sex"
FEMALE_CODE <- "Female"
MALE_CODE   <- "Male"

# These are treated as disease/code events: the code itself becomes part of the
# token, e.g. ICD10::E11. Continuous numeric values are not binned for these.
disease_codings <- c("ICD10", "self_reported_cancer", "self_reported_non_cancer")

# Number of bins for continuous measurements such as albumin.
n_value_bins <- 4L

# --- Load data ---------------------------------------------------------------
cat("Loading final.rds...\n")
dat <- as.data.table(readRDS(final_path))
cat("Rows:", nrow(dat), " | Patients:", uniqueN(dat$eid), "\n")

required_cols <- c("eid", "coding", "code", "age")
missing_cols <- setdiff(required_cols, names(dat))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

if (!"chapter" %in% names(dat)) {
  dat[, chapter := NA_character_]
}

# Drop rows that cannot be placed on a timeline.
dat <- dat[!is.na(eid) & !is.na(coding) & !is.na(code) & !is.na(age)]
cat("Rows after dropping missing eid/coding/code/age:", nrow(dat), "\n")

# --- Standardise types -------------------------------------------------------
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

# --- Build readable 4-column event rows -------------------------------------
# This is the human-readable table you can inspect and keep using.
readable_event_rows <- dat[, .(
  eid,
  age_days,
  token_wording = paste0(source_type, "::", coding),
  token_value = code_raw
)]
setorder(readable_event_rows, eid, age_days, token_wording, token_value)

fwrite(readable_event_rows, file.path(out_dir, "readable_event_rows.csv"))
cat("Written readable_event_rows.csv with", nrow(readable_event_rows), "rows\n")

# --- Convert values into token labels ---------------------------------------
# Delphi needs one integer token_id per event. For continuous numeric variables,
# using exact values would explode the vocabulary, so we bin them within each
# coding variable. For disease/code variables, we keep the original code label.
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

# --- Build token vocabulary --------------------------------------------------
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
cat("Written token_dictionary.csv with", nrow(token_dictionary), "rows\n")

# Assign token IDs.
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
cat("Rows after token assignment:", nrow(dat), "\n")

# Save a model-ready readable table too.
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
cat("Written model_event_rows_readable.csv with", nrow(model_event_rows_readable), "rows\n")

# --- Sort by patient, then age -----------------------------------------------
setorder(dat, eid, age_days, token_id)

# --- Split into train / val by original eid ----------------------------------
patients <- unique(dat$eid)
n_val <- as.integer(round(val_frac * length(patients)))
val_eids <- sample(patients, n_val)
cat("Train patients:", length(patients) - n_val, " | Val patients:", n_val, "\n")

train_dat <- copy(dat[!eid %in% val_eids])
val_dat   <- copy(dat[ eid %in% val_eids])

# Re-index patient IDs consecutively within each split. Delphi uses this only
# to group rows; it does not need to be the original UKB eid.
train_dat[, patient_id := .GRP - 1L, by = eid]
val_dat[, patient_id := .GRP - 1L, by = eid]

setorder(train_dat, patient_id, age_days, token_id)
setorder(val_dat, patient_id, age_days, token_id)

# --- Write binary files -------------------------------------------------------
write_bin <- function(dt, path) {
  # bin_token_id is the pre-shift token ID. Delphi's get_batch() adds +1.
  m <- as.matrix(dt[, .(patient_id, age_days, bin_token_id)])
  storage.mode(m) <- "integer"

  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.integer(t(m)), con, size = 4L, endian = "little")

  cat("Written:", path, " | Rows:", nrow(dt), "\n")
}

cat("\nWriting train.bin...\n")
write_bin(train_dat, file.path(out_dir, "train.bin"))

cat("Writing val.bin...\n")
write_bin(val_dat, file.path(out_dir, "val.bin"))

# --- Write labels.csv --------------------------------------------------------
# Row N+1 corresponds to token N, matching Delphi's expected label convention.
labels <- token_dictionary[, .(event_name = token_wording)]
fwrite(labels, file.path(out_dir, "labels.csv"))
cat("Written labels.csv with", nrow(labels), "rows\n")

# --- Save vocab metadata -----------------------------------------------------
vocab_size <- max(token_dictionary$token_id) + 1L

# For training, ignore tokens that are context-only rather than disease targets:
# padding, no-event, sex, and non-disease custom tokens.
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

cat("\nPreprocessing complete.\n")
cat("vocab_size:", vocab_size, "\n")
cat("Number of ignore_tokens:", length(ignore_tokens_vec), "\n")
cat("Output directory:", out_dir, "\n")

