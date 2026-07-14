df <- readRDS("/rds/general/project/hda_24-25/live/amk125_thesis/General/extraction_and_recoding/outputs/ukb_recoded.rds")

names_1x <- grep("\\.1\\.[0-9]$", names(df), value = TRUE)
names_2x <- grep("\\.2\\.[0-9]$", names(df), value = TRUE)
names_3x <- grep("\\.3\\.[0-9]$", names(df), value = TRUE)

df <- df[, !names(df) %in% c(names_1x, names_2x, names_3x)]

# ── Qualifications — keep highest credential ──────────────────────────────────
q_cols <- intersect(paste0("qualifications.0.", 0:5), names(df))

if (length(q_cols) > 0) {
  rank_map_q <- c(
    "College or University degree"                                 = 7,
    "NVQ or HND or HNC or equivalent"                             = 6,
    "A levels/AS levels or equivalent"                            = 5,
    "O levels/GCSEs or equivalent"                                = 4,
    "CSEs or equivalent"                                          = 3,
    "Other professional qualifications eg: nursing, teaching"     = 2,
    "Other professional qualifications (e.g., nursing, teaching)" = 2,
    "None of the above"                                           = 1,
    "Prefer not to answer"                                        = 0
  )
  
  
  df$qualifications <- apply(df[, q_cols, drop = FALSE], 1, function(x) {
    x <- trimws(as.character(x)); x <- x[!is.na(x) & x != "" & x != "NA"]
    if (length(x) == 0) return(NA_character_)
    r <- rank_map_q[x]; r[is.na(r)] <- -Inf
    x[which.max(r)]
  })
  
  df[q_cols] <- NULL
}

# ── Smoking status — keep highest-priority category ───────────────────────────
smk_cols <- intersect(paste0("smoking_status.0.", 0:5), names(df))

if (length(smk_cols) > 0) {
  rank_map_smk <- c(
    "Current"              = 3,
    "Previous"             = 2,
    "Never"                = 1,
    "Prefer not to answer" = 0
  )
  
  df$smoking_status <- apply(df[, smk_cols, drop = FALSE], 1, function(x) {
    x <- trimws(as.character(x)); x <- x[!is.na(x) & x != "" & x != "NA"]
    if (length(x) == 0) return(NA_character_)
    r <- rank_map_smk[x]; r[is.na(r)] <- -Inf
    x[which.max(r)]
  })
  
  df[smk_cols] <- NULL
}



# ── Alcohol status — keep highest-priority category ───────────────────────────
alc_cols <- intersect(paste0("alcohol_status.0.", 0:5), names(df))

if (length(alc_cols) > 0) {
  rank_map_alc <- c(
    "Current"              = 3,
    "Previous"             = 2,
    "Never"                = 1,
    "Prefer not to answer" = 0
  )
  
  df$alcohol_status <- apply(df[, alc_cols, drop = FALSE], 1, function(x) {
    x <- trimws(as.character(x)); x <- x[!is.na(x) & x != "" & x != "NA"]
    if (length(x) == 0) return(NA_character_)
    r <- rank_map_alc[x]; r[is.na(r)] <- -Inf
    x[which.max(r)]
  })
  
  df[alc_cols] <- NULL
}

df$approx_birth_date <- as.Date(
  paste(
    df$year_of_birth.0.0,
    match(as.character(df$month_of_birth.0.0), month.name),
    "15",
    sep = "-"
  )
)

df <- tibble::rownames_to_column(df, var = "eid")

saveRDS(df, "/rds/general/project/hda_24-25/live/amk125_thesis/General/extraction_and_recoding/outputs/ukb_cleaning.rds")
