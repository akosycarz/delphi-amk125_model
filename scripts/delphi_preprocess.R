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

# Healthy-cohort reporting settings. A participant is healthy at an index age
# when no qualifying diagnosis has been recorded on or before that age and no
# death has been recorded before that age. Reports are produced for both an
# ICD-only definition and an ICD + self-reported definition.
HEALTHY_AGE_MIN  <- 40L
HEALTHY_AGE_MAX  <- 80L
HEALTHY_AGE_STEP <- 5L
HEALTHY_DETAIL_AGE <- 50L

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
  !is.na(eid) & !is.na(coding) & !is.na(code) & !is.na(age) &
    coding %in% c("ICD10", "ICD9")
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

# --- Healthy-participant reports ---------------------------------------------
make_healthy_reports <- function() {
  cat("Creating healthy-participant reports...\n")

  prepare_diagnoses <- function(dt, allowed_codings) {
    x <- as.data.table(dt)
    x <- x[!is.na(eid) & !is.na(coding) & !is.na(age)]
    x[, `:=`(
      eid = as.character(eid),
      coding = as.character(coding),
      diagnosis_age = suppressWarnings(as.numeric(age))
    )]
    x[eid %in% all_eids & coding %in% allowed_codings & !is.na(diagnosis_age),
      .(first_diagnosis_age = min(diagnosis_age)), by = eid]
  }

  icd_first <- prepare_diagnoses(clinical_history_icd, c("ICD10", "ICD9"))
  all_diagnosis_sources <- rbindlist(
    list(clinical_history_icd, self_reported_cancer, self_reported_noncancer),
    use.names = TRUE, fill = TRUE
  )
  icd_sr_first <- prepare_diagnoses(
    all_diagnosis_sources,
    c("ICD10", "ICD9", "self_reported_cancer", "self_reported_non_cancer")
  )

  death <- as.data.table(clinical_history_icd)[
    !is.na(eid) & !is.na(coding) & !is.na(age)
  ]
  death[, `:=`(
    eid = as.character(eid),
    coding = as.character(coding),
    death_age = suppressWarnings(as.numeric(age))
  )]
  death <- death[eid %in% all_eids & coding == "DEATH" & !is.na(death_age)]
  if (nrow(death) > 0L) {
    death <- death[, .(death_age = min(death_age)), by = eid]
  } else {
    death <- data.table(eid = character(), death_age = numeric())
  }

  sex <- as.data.table(demographics)[
    !is.na(eid) & !is.na(coding) & !is.na(code)
  ]
  sex[, `:=`(eid = as.character(eid), coding = as.character(coding))]
  sex <- sex[
    eid %in% all_eids & coding == SEX_CODING,
    .(sex = as.character(code)[1L]), by = eid
  ]
  sex[!sex %in% c(FEMALE_CODE, MALE_CODE), sex := "Unknown"]

  base <- merge(split_assignments, sex, by = "eid", all.x = TRUE)
  base[is.na(sex), sex := "Unknown"]
  base <- merge(base, death, by = "eid", all.x = TRUE)

  ages <- seq(HEALTHY_AGE_MIN, HEALTHY_AGE_MAX, by = HEALTHY_AGE_STEP)
  definitions <- list(
    icd_only = icd_first,
    icd_plus_self_reported = icd_sr_first
  )

  # Explicit Cartesian expansion (works across data.table versions).
  detail_list <- lapply(names(definitions), function(definition_name) {
    x <- merge(base, definitions[[definition_name]], by = "eid", all.x = TRUE)
    x <- x[rep(seq_len(.N), each = length(ages))]
    x[, index_age := rep(ages, times = nrow(base))]
    x[, definition := definition_name]
    x[, eligible := is.na(death_age) | death_age >= index_age]
    x[, healthy := eligible &
      (is.na(first_diagnosis_age) | first_diagnosis_age > index_age)]
    x[, age_bin := sprintf("%d-%d", index_age, index_age + HEALTHY_AGE_STEP - 1L)]
    x
  })
  healthy_detail <- rbindlist(detail_list, use.names = TRUE, fill = TRUE)

  summarise_healthy <- function(dt, group_cols, split_label = NULL, sex_label = NULL) {
    out <- dt[, .(
      n_participants = uniqueN(eid),
      n_eligible = uniqueN(eid[eligible]),
      n_healthy = uniqueN(eid[healthy])
    ), by = group_cols]
    if (!is.null(split_label)) out[, split := split_label]
    if (!is.null(sex_label)) out[, sex := sex_label]
    out[, healthy_percent := fifelse(
      n_eligible > 0L, 100 * n_healthy / n_eligible, NA_real_
    )]
    out
  }

  summary <- rbindlist(list(
    summarise_healthy(healthy_detail,
                      c("definition", "index_age", "age_bin", "split", "sex")),
    summarise_healthy(healthy_detail,
                      c("definition", "index_age", "age_bin", "split"),
                      sex_label = "All"),
    summarise_healthy(healthy_detail,
                      c("definition", "index_age", "age_bin", "sex"),
                      split_label = "all"),
    summarise_healthy(healthy_detail,
                      c("definition", "index_age", "age_bin"),
                      split_label = "all", sex_label = "All")
  ), use.names = TRUE, fill = TRUE)
  setcolorder(summary, c(
    "definition", "index_age", "age_bin", "split", "sex",
    "n_participants", "n_eligible", "n_healthy", "healthy_percent"
  ))
  setorder(summary, definition, index_age, split, sex)

  detail_at_age <- healthy_detail[index_age == HEALTHY_DETAIL_AGE, .(
    eid, split, sex, definition, index_age, first_diagnosis_age,
    death_age, eligible, healthy
  )]
  setorder(detail_at_age, definition, split, eid)

  fwrite(summary, file.path(out_base, "healthy_participant_summary.csv"))
  fwrite(detail_at_age, file.path(
    out_base, paste0("healthy_participants_at_age_", HEALTHY_DETAIL_AGE, ".csv")
  ))
  writeLines(c(
    "Healthy-participant definition",
    "==============================",
    "Eligible: no recorded death before the index age.",
    "Healthy: eligible and no qualifying diagnosis recorded on or before the index age.",
    "icd_only includes ICD-10 and ICD-9 diagnoses.",
    "icd_plus_self_reported additionally includes self-reported cancer and non-cancer diagnoses.",
    "Absence of a recorded diagnosis does not prove absence of disease; this is an operational data definition.",
    paste0("Index ages: ", paste(ages, collapse = ", "), "."),
    paste0("Participant-level detail is saved at age ", HEALTHY_DETAIL_AGE, ".")
  ), file.path(out_base, "healthy_definition.txt"))
  cat("Written healthy participant definition, summary, and age-",
      HEALTHY_DETAIL_AGE, " detail reports.\n", sep = "")
}

make_healthy_reports()

# --- Age/split distribution verification -------------------------------------
make_age_split_reports <- function() {
  cat("Verifying age distributions across splits...\n")

  first_icd_age <- clinical_cohort[
    !is.na(age_numeric),
    .(first_icd_age = min(age_numeric)), by = .(eid = as.character(eid))
  ]

  sex_lookup <- as.data.table(demographics)[
    !is.na(eid) & !is.na(coding) & !is.na(code)
  ]
  sex_lookup[, `:=`(eid = as.character(eid), coding = as.character(coding))]
  sex_lookup <- sex_lookup[
    eid %in% all_eids & coding == SEX_CODING,
    .(sex = as.character(code)[1L]), by = eid
  ]
  sex_lookup[!sex %in% c(FEMALE_CODE, MALE_CODE), sex := "Unknown"]

  ages <- merge(split_assignments, first_icd_age, by = "eid", all.x = TRUE)
  ages <- merge(ages, sex_lookup, by = "eid", all.x = TRUE)
  ages[is.na(sex), sex := "Unknown"]
  if (ages[, anyNA(first_icd_age)]) {
    stop("Shared cohort contains participants without a valid first ICD age")
  }

  full_ages <- copy(ages)
  full_ages[, split := "full"]
  report_ages <- rbind(ages, full_ages, use.names = TRUE)

  age_summary <- report_ages[, .(
    n_participants = uniqueN(eid),
    mean_age = mean(first_icd_age),
    sd_age = sd(first_icd_age),
    median_age = median(first_icd_age),
    q1_age = as.numeric(quantile(first_icd_age, 0.25)),
    q3_age = as.numeric(quantile(first_icd_age, 0.75)),
    min_age = min(first_icd_age),
    max_age = max(first_icd_age)
  ), by = .(split, sex)]
  all_sex <- report_ages[, .(
    n_participants = uniqueN(eid),
    mean_age = mean(first_icd_age),
    sd_age = sd(first_icd_age),
    median_age = median(first_icd_age),
    q1_age = as.numeric(quantile(first_icd_age, 0.25)),
    q3_age = as.numeric(quantile(first_icd_age, 0.75)),
    min_age = min(first_icd_age),
    max_age = max(first_icd_age)
  ), by = split]
  all_sex[, sex := "All"]
  age_summary <- rbind(age_summary, all_sex, use.names = TRUE, fill = TRUE)
  setcolorder(age_summary, c(
    "split", "sex", "n_participants", "mean_age", "sd_age", "median_age",
    "q1_age", "q3_age", "min_age", "max_age"
  ))
  setorder(age_summary, split, sex)

  bin_min <- floor(min(ages$first_icd_age) / 5) * 5
  bin_max <- ceiling(max(ages$first_icd_age) / 5) * 5 + 5
  breaks <- seq(bin_min, bin_max, by = 5)
  report_ages[, age_bin := cut(
    first_icd_age, breaks = breaks, right = FALSE, include.lowest = TRUE
  )]
  age_bins <- report_ages[, .(n_participants = uniqueN(eid)),
                           by = .(split, sex, age_bin)]
  age_bins[, proportion := n_participants / sum(n_participants),
           by = .(split, sex)]
  all_sex_bins <- report_ages[, .(n_participants = uniqueN(eid)),
                              by = .(split, age_bin)]
  all_sex_bins[, `:=`(
    sex = "All",
    proportion = n_participants / sum(n_participants)
  ), by = split]
  age_bins <- rbind(age_bins, all_sex_bins, use.names = TRUE, fill = TRUE)
  setorder(age_bins, split, sex, age_bin)

  split_pairs <- list(c("train", "val"), c("train", "test"), c("val", "test"))
  test_rows <- lapply(split_pairs, function(pair) {
    x <- ages[split == pair[1L], first_icd_age]
    y <- ages[split == pair[2L], first_icd_age]
    ks <- suppressWarnings(ks.test(x, y, exact = FALSE))
    data.table(
      split_1 = pair[1L],
      split_2 = pair[2L],
      n_1 = length(x),
      n_2 = length(y),
      mean_difference_years = mean(x) - mean(y),
      median_difference_years = median(x) - median(y),
      ks_statistic = unname(ks$statistic),
      ks_p_value = ks$p.value
    )
  })
  age_tests <- rbindlist(test_rows)

  # Large cohorts make tiny, unimportant differences statistically significant.
  # Flag practical differences using effect-size thresholds as well as reporting
  # the unadjusted KS p-value.
  age_tests[, practical_warning :=
    abs(mean_difference_years) > 1 | abs(median_difference_years) > 1 |
    ks_statistic > 0.05]
  verification_status <- if (any(age_tests$practical_warning)) "WARNING" else "PASS"

  fwrite(age_summary, file.path(out_base, "age_split_summary.csv"))
  fwrite(age_bins, file.path(out_base, "age_split_5year_bins.csv"))
  fwrite(age_tests, file.path(out_base, "age_split_pairwise_tests.csv"))

  png(file.path(out_base, "age_split_distribution.png"),
      width = 1400, height = 900, res = 150)
  plot_data <- age_bins[sex == "All"]
  split_order <- c("full", "train", "val", "test")
  colours <- c(full = "black", train = "#0072B2", val = "#E69F00", test = "#009E73")
  mids <- seq_along(levels(plot_data$age_bin))
  plot(mids, rep(0, length(mids)), type = "n", ylim = c(0, max(plot_data$proportion) * 1.1),
       xaxt = "n", xlab = "Age at first recorded ICD diagnosis (5-year bins)",
       ylab = "Proportion of participants", main = "Age distribution by data split")
  axis(1, at = mids, labels = levels(plot_data$age_bin), las = 2, cex.axis = 0.8)
  for (s in split_order) {
    d <- plot_data[split == s]
    values <- d$proportion[match(levels(plot_data$age_bin), as.character(d$age_bin))]
    values[is.na(values)] <- 0
    lines(mids, values, type = "o", col = colours[[s]], lwd = 2, pch = 16)
  }
  legend("topright", legend = split_order, col = colours[split_order],
         lwd = 2, pch = 16, bty = "n")
  dev.off()

  writeLines(c(
    paste0("Age/split distribution verification: ", verification_status),
    "=================================================",
    "Age measure: age at first valid ICD-9 or ICD-10 diagnosis.",
    "Compared splits: train vs validation, train vs test, validation vs test.",
    "Practical warning if absolute mean or median difference exceeds 1 year,",
    "or the two-sample Kolmogorov-Smirnov statistic exceeds 0.05.",
    "KS p-values are supplied but are not used alone because very large samples",
    "can make negligible distribution differences statistically significant.",
    paste0("Result: ", verification_status),
    if (verification_status == "WARNING")
      "Review age_split_pairwise_tests.csv before training." else
      "No practically important split difference was detected."
  ), file.path(out_base, "age_split_verification.txt"))

  cat("Written age/split summaries, five-year bins, tests, plot, and verification.\n")
}

make_age_split_reports()

# --- Preprocessing function --------------------------------------------------
preprocess_config <- function(cfg_name, sources_list, out_dir,
                               train_eids, val_eids, test_eids) {

  cat("=================================================================\n")
  cat("Configuration:", cfg_name, "\n")
  cat("Output:       ", out_dir, "\n")
  cat("=================================================================\n")

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  flow_rows <- list()
  add_flow <- function(stage, before, after, reason) {
    flow_rows[[length(flow_rows) + 1L]] <<- data.table(
      configuration = cfg_name,
      stage_order = length(flow_rows) + 1L,
      stage = stage,
      removal_reason = reason,
      input_participants = uniqueN(before$eid),
      output_participants = uniqueN(after$eid),
      participants_removed = uniqueN(before$eid) - uniqueN(after$eid),
      input_events = nrow(before),
      output_events = nrow(after),
      events_removed = nrow(before) - nrow(after)
    )
  }

  # -- Combine and clean -------------------------------------------------------
  dat <- rbindlist(sources_list, use.names = TRUE, fill = TRUE)
  add_flow("combined_sources", dat, dat, "No filtering; raw combined input")

  required_cols <- c("eid", "coding", "code", "age")
  missing_cols  <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0L)
    stop(cfg_name, ": missing required columns: ", paste(missing_cols, collapse = ", "))

  if (!"chapter" %in% names(dat)) dat[, chapter := NA_character_]

  before <- copy(dat)
  dat <- dat[!is.na(eid) & !is.na(coding) & !is.na(code) & !is.na(age)]
  add_flow("complete_required_fields", before, dat,
           "Missing eid, coding, code, or age")

  dat[, eid      := as.character(eid)]
  dat[, coding   := as.character(coding)]
  dat[, code_raw := as.character(code)]
  dat[, age_years := suppressWarnings(as.numeric(age))]
  before <- copy(dat)
  dat <- dat[!is.na(age_years)]
  add_flow("numeric_age", before, dat, "Age could not be converted to numeric")

  # Enforce the shared clinical-ICD cohort for every experiment.
  before <- copy(dat)
  dat <- dat[eid %in% all_eids]
  add_flow("shared_clinical_cohort", before, dat,
           "Participant not in eligible clinical ICD cohort")

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
  before <- copy(dat)
  dat <- dat[!is.na(token_id)]
  add_flow("token_assignment", before, dat, "No model token could be assigned")
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

  final_flow <- rbindlist(list(
    rbindlist(flow_rows, use.names = TRUE, fill = TRUE),
    data.table(
      configuration = cfg_name,
      stage_order = length(flow_rows) + 1:3,
      stage = c("final_train", "final_validation", "final_test"),
      removal_reason = "Shared split assignment",
      input_participants = uniqueN(dat$eid),
      output_participants = c(
        uniqueN(train_dat$eid), uniqueN(val_dat$eid), uniqueN(test_dat$eid)
      ),
      participants_removed = NA_integer_,
      input_events = nrow(dat),
      output_events = c(nrow(train_dat), nrow(val_dat), nrow(test_dat)),
      events_removed = NA_integer_
    )
  ), use.names = TRUE, fill = TRUE)
  final_flow[, participant_retention_percent := fifelse(
    input_participants > 0L, 100 * output_participants / input_participants, NA_real_
  )]
  final_flow[, event_retention_percent := fifelse(
    input_events > 0L, 100 * output_events / input_events, NA_real_
  )]
  fwrite(final_flow, file.path(out_dir, "participant_flow.csv"))
  cat("Written participant_flow.csv\n")
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

flow_files <- file.path(out_base, names(configs), "participant_flow.csv")
participant_flow_all <- rbindlist(lapply(flow_files, fread), use.names = TRUE, fill = TRUE)
fwrite(participant_flow_all, file.path(out_base, "participant_flow_all.csv"))
cat("Written combined participant_flow_all.csv\n")

cat("All configurations complete.\n")
cat("Output root:", out_base, "\n")
