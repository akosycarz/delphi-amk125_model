# ============================================================
# Time-Window Analysis for Repeated Diagnoses
# Clinical History ICD – Episode Deduplication
# ============================================================

library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)
library(stringr)

# ---- 0. Load input data ------------------------------------

in_file <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sources/clinical_history_icd.rds"

if (!file.exists(in_file)) {
  stop("Input file not found: ", in_file)
}

clinical_history_icd <- readRDS(in_file)

cat("Loaded:", in_file, "\n")
cat(
  "Class:", paste(class(clinical_history_icd), collapse = "/"),
  "| rows:", format(nrow(clinical_history_icd), big.mark = ","),
  "| cols:", ncol(clinical_history_icd), "\n"
)

required_cols <- c("eid", "code", "date", "coding")
missing_cols  <- setdiff(required_cols, names(clinical_history_icd))

if (length(missing_cols) > 0) {
  stop(
    "Input is missing required column(s): ",
    paste(missing_cols, collapse = ", "),
    "\nColumns present: ",
    paste(names(clinical_history_icd), collapse = ", ")
  )
}

cat("All required columns present:", paste(required_cols, collapse = ", "), "\n\n")


# ---- 1. Prepare sample data --------------------------------

set.seed(42)

n_available <- length(unique(clinical_history_icd$eid))
n_sample <- min(1000, n_available)

sample_eids <- sample(unique(clinical_history_icd$eid), n_sample)

df_sample <- clinical_history_icd %>%
  filter(eid %in% sample_eids) %>%
  mutate(date = as.Date(date)) %>%
  filter(!is.na(date)) %>%
  arrange(eid, code, date)

n_participants <- n_distinct(df_sample$eid)

cat("Sample:", nrow(df_sample), "records |", n_participants, "participants\n")
cat("Date range:", format(min(df_sample$date)), "to", format(max(df_sample$date)), "\n")
cat(
  "Coding types:",
  paste(
    names(table(df_sample$coding)),
    table(df_sample$coding),
    sep = "=",
    collapse = ", "
  ),
  "\n\n"
)


# ---- 1b. Show why coding must be ignored in the key ---------

repeats_with_key_fix <- df_sample %>%
  group_by(eid, code) %>%
  filter(n() > 1) %>%
  ungroup() %>%
  nrow()

repeats_old_key <- df_sample %>%
  group_by(eid, code, coding) %>%
  filter(n() > 1) %>%
  ungroup() %>%
  nrow()

cat("Records in repeated eid+code groups (CORRECT key):  ", repeats_with_key_fix, "\n")
cat("Records in repeated eid+code+coding groups (OLD key):", repeats_old_key, "\n")
cat(
  "Difference:",
  repeats_with_key_fix - repeats_old_key,
  "records missed by old key (same code, different coding system)\n\n"
)

cat("Examples of same code on different dates with different coding:\n")

df_sample %>%
  group_by(eid, code) %>%
  filter(n() > 1, n_distinct(coding) > 1) %>%
  ungroup() %>%
  arrange(eid, code, date) %>%
  select(eid, code, coding, date) %>%
  head(12) %>%
  print(row.names = FALSE)


# ---- 2. ICD-10 chapter mapping -----------------------------

icd10_chapter <- function(code) {
  
  ch <- toupper(substr(trimws(code), 1, 1))
  num <- suppressWarnings(as.numeric(substr(trimws(code), 2, 4)))
  
  dplyr::case_when(
    ch %in% c("A", "B") ~
      "Chapter I Certain infectious and parasitic diseases",
    
    ch == "C" ~
      "Chapter II Neoplasms",
    
    ch == "D" & !is.na(num) & num <= 48 ~
      "Chapter II Neoplasms",
    
    ch == "D" & (is.na(num) | num > 48) ~
      "Chapter III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    
    ch == "E" ~
      "Chapter IV Endocrine, nutritional and metabolic diseases",
    
    ch == "F" ~
      "Chapter V Mental and behavioural disorders",
    
    ch == "G" ~
      "Chapter VI Diseases of the nervous system",
    
    ch == "H" ~
      "Chapter VII/VIII Diseases of the eye, adnexa, ear and mastoid process",
    
    ch == "I" ~
      "Chapter IX Diseases of the circulatory system",
    
    ch == "J" ~
      "Chapter X Diseases of the respiratory system",
    
    ch == "K" ~
      "Chapter XI Diseases of the digestive system",
    
    ch == "L" ~
      "Chapter XII Diseases of the skin and subcutaneous tissue",
    
    ch == "M" ~
      "Chapter XIII Diseases of the musculoskeletal system and connective tissue",
    
    ch == "N" ~
      "Chapter XIV Diseases of the genitourinary system",
    
    ch == "O" ~
      "Chapter XV Pregnancy, childbirth and the puerperium",
    
    ch == "P" ~
      "Chapter XVI Certain conditions originating in the perinatal period",
    
    ch == "Q" ~
      "Chapter XVII Congenital malformations, deformations and chromosomal abnormalities",
    
    ch == "R" ~
      "Chapter XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
    
    ch %in% c("S", "T") ~
      "Chapter XIX Injury, poisoning and certain other consequences of external causes",
    
    ch %in% c("V", "W", "X", "Y") ~
      "Chapter XX External causes of morbidity and mortality",
    
    ch == "Z" ~
      "Chapter XXI Factors influencing health status and contact with health services",
    
    TRUE ~
      "Other / non-ICD10"
  )
}


# ---- 2b. Broad chapter / category for all coding systems ----

assign_chapter_broad <- function(code, coding) {
  dplyr::case_when(
    coding %in% c("ICD10", "ICD9") ~
      icd10_chapter(code),
    
    coding == "OPCS4" ~
      "Procedure (OPCS-4)",
    
    coding == "DEATH" ~
      "Death record",
    
    grepl("cancer_code_self_reported", coding, fixed = TRUE) ~
      "Chapter II Neoplasms",
    
    grepl("non_cancer_illness", coding, fixed = TRUE) ~
      "Self-reported non-cancer condition",
    
    TRUE ~
      "Other / non-ICD10"
  )
}


chapter_order <- c(
  "Chapter I Certain infectious and parasitic diseases",
  "Chapter II Neoplasms",
  "Chapter III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
  "Chapter IV Endocrine, nutritional and metabolic diseases",
  "Chapter V Mental and behavioural disorders",
  "Chapter VI Diseases of the nervous system",
  "Chapter VII/VIII Diseases of the eye, adnexa, ear and mastoid process",
  "Chapter IX Diseases of the circulatory system",
  "Chapter X Diseases of the respiratory system",
  "Chapter XI Diseases of the digestive system",
  "Chapter XII Diseases of the skin and subcutaneous tissue",
  "Chapter XIII Diseases of the musculoskeletal system and connective tissue",
  "Chapter XIV Diseases of the genitourinary system",
  "Chapter XV Pregnancy, childbirth and the puerperium",
  "Chapter XVI Certain conditions originating in the perinatal period",
  "Chapter XVII Congenital malformations, deformations and chromosomal abnormalities",
  "Chapter XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
  "Chapter XIX Injury, poisoning and certain other consequences of external causes",
  "Chapter XX External causes of morbidity and mortality",
  "Chapter XXI Factors influencing health status and contact with health services",
  "Procedure (OPCS-4)",
  "Death record",
  "Self-reported non-cancer condition",
  "Other / non-ICD10"
)


df_sample <- df_sample %>%
  mutate(
    chapter = assign_chapter_broad(code, coding)
  )


# ---- 3. Episode-counting function --------------------------

count_episodes <- function(dates, window_days) {
  
  if (length(dates) == 0L) {
    return(0L)
  }
  
  if (window_days < 0) {
    return(length(dates))
  }
  
  dates <- sort(dates)
  
  if (length(dates) == 1L) {
    return(1L)
  }
  
  episodes <- 1L
  
  for (i in 2:length(dates)) {
    if (as.numeric(dates[i] - dates[i - 1L], units = "days") > window_days) {
      episodes <- episodes + 1L
    }
  }
  
  episodes
}


# ---- 4. Apply time windows ---------------------------------

windows <- c(raw = -1, w30 = 30, w60 = 60, w90 = 90)


# ---- 4a. Overall summary -----------------------------------

overall_list <- lapply(names(windows), function(w_name) {
  
  w <- windows[[w_name]]
  
  df_sample %>%
    group_by(eid, code) %>%
    summarise(
      n_eps = count_episodes(date, w),
      .groups = "drop"
    ) %>%
    group_by(eid) %>%
    summarise(
      eps_per_person = sum(n_eps),
      .groups = "drop"
    ) %>%
    summarise(
      window_name       = w_name,
      total_episodes    = sum(eps_per_person),
      avg_per_person    = round(mean(eps_per_person), 2),
      median_per_person = round(median(eps_per_person), 1),
      .groups = "drop"
    )
})

overall_summary <- bind_rows(overall_list) %>%
  mutate(
    window_label = dplyr::recode(
      window_name,
      "raw" = "No window (raw)",
      "w30" = "30-day window",
      "w60" = "60-day window",
      "w90" = "90-day window"
    ),
    raw_episodes  = total_episodes[window_name == "raw"],
    abs_reduction = raw_episodes - total_episodes,
    pct_reduction = round(100 * abs_reduction / raw_episodes, 1)
  ) %>%
  select(
    window_label,
    total_episodes,
    avg_per_person,
    median_per_person,
    abs_reduction,
    pct_reduction
  )

cat("\n============================================================\n")
cat("METRICS 1 + 2 + 4: Overall summary\n")
cat("Sample:", n_participants, "participants | group key: eid + code\n")
cat("============================================================\n")

print(overall_summary, row.names = FALSE)


# ---- 4b. Per-chapter summary -------------------------------

chapter_list <- lapply(names(windows), function(w_name) {
  
  w <- windows[[w_name]]
  
  df_sample %>%
    group_by(eid, chapter, code) %>%
    summarise(
      n_eps = count_episodes(date, w),
      .groups = "drop"
    ) %>%
    group_by(chapter) %>%
    summarise(
      window_name = w_name,
      n_episodes  = sum(n_eps),
      .groups = "drop"
    )
})

chapter_summary_wide <- bind_rows(chapter_list) %>%
  pivot_wider(
    names_from = window_name,
    values_from = n_episodes,
    values_fill = 0
  ) %>%
  rename(
    raw_eps = raw,
    eps_30  = w30,
    eps_60  = w60,
    eps_90  = w90
  ) %>%
  mutate(
    red_30 = raw_eps - eps_30,
    red_60 = raw_eps - eps_60,
    red_90 = raw_eps - eps_90,
    pct_30 = round(100 * red_30 / raw_eps, 1),
    pct_60 = round(100 * red_60 / raw_eps, 1),
    pct_90 = round(100 * red_90 / raw_eps, 1),
    chapter = factor(chapter, levels = chapter_order)
  ) %>%
  arrange(chapter)

cat("\n============================================================\n")
cat("METRIC 3 + 4: Episodes by broad disease chapter/category\n")
cat("============================================================\n")

print(
  chapter_summary_wide %>%
    select(
      chapter,
      raw_eps,
      eps_30,
      eps_60,
      eps_90,
      pct_30,
      pct_60,
      pct_90
    ),
  row.names = FALSE
)


# ---- 5. Generate PNG Table 1: Overall summary --------------

make_summary_png <- function(
    df,
    out_file,
    title = "Repeated Diagnosis Deduplication",
    sub = paste0(
      "Random 1000 participants  ·  clinical_history_icd  ·  ",
      format(Sys.Date(), "%d %b %Y")
    )
) {
  
  col_headers <- c(
    "Time Window",
    "Total\nEpisodes",
    "Avg /\nPerson",
    "Median /\nPerson",
    "Reduction\n(abs)",
    "Reduction\n(%)"
  )
  
  col_x  <- c(0.20, 0.38, 0.52, 0.64, 0.78, 0.91)
  n_rows <- nrow(df)
  row_h  <- 0.13
  hdr_y  <- 0.72
  
  row_y <- seq(
    hdr_y - row_h,
    hdr_y - row_h * (n_rows + 0.5),
    length.out = n_rows
  )
  
  cell <- df %>%
    transmute(
      c1 = window_label,
      c2 = format(total_episodes, big.mark = ","),
      c3 = format(avg_per_person, nsmall = 2),
      c4 = format(median_per_person, nsmall = 1),
      c5 = ifelse(
        abs_reduction == 0,
        "—",
        paste0("-", format(abs_reduction, big.mark = ","))
      ),
      c6 = ifelse(
        pct_reduction == 0,
        "—",
        paste0("-", pct_reduction, "%")
      )
    )
  
  p <- ggplot() +
    annotate(
      "rect",
      xmin = 0,
      xmax = 1,
      ymin = 0.85,
      ymax = 1,
      fill = "#1F4E79",
      colour = NA
    ) +
    annotate(
      "text",
      x = 0.5,
      y = 0.935,
      label = title,
      hjust = 0.5,
      size = 5.5,
      fontface = "bold",
      colour = "white"
    ) +
    annotate(
      "text",
      x = 0.5,
      y = 0.873,
      label = sub,
      hjust = 0.5,
      size = 3.0,
      colour = "#A8C8E8"
    ) +
    annotate(
      "rect",
      xmin = 0,
      xmax = 1,
      ymin = hdr_y - row_h / 2,
      ymax = hdr_y + row_h / 2,
      fill = "#2E6DA4",
      colour = NA
    ) +
    annotate(
      "text",
      x = col_x,
      y = rep(hdr_y, 6),
      label = col_headers,
      hjust = 0.5,
      vjust = 0.5,
      size = 3.2,
      fontface = "bold",
      colour = "white",
      lineheight = 0.9
    ) +
    {
      lapply(seq(1, n_rows, 2), function(r) {
        annotate(
          "rect",
          xmin = 0,
          xmax = 1,
          ymin = row_y[r] - row_h / 2,
          ymax = row_y[r] + row_h / 2,
          fill = "#EBF3FA",
          colour = NA
        )
      })
    } +
    {
      mapply(
        function(r, c1, c2, c3, c4, c5, c6) {
          list(
            annotate(
              "text",
              x = col_x[1],
              y = row_y[r],
              label = c1,
              hjust = 0.5,
              size = 3.3,
              colour = "grey15",
              fontface = "bold"
            ),
            annotate(
              "text",
              x = col_x[2],
              y = row_y[r],
              label = c2,
              hjust = 0.5,
              size = 3.3,
              colour = "grey15"
            ),
            annotate(
              "text",
              x = col_x[3],
              y = row_y[r],
              label = c3,
              hjust = 0.5,
              size = 3.3,
              colour = "grey15"
            ),
            annotate(
              "text",
              x = col_x[4],
              y = row_y[r],
              label = c4,
              hjust = 0.5,
              size = 3.3,
              colour = "grey15"
            ),
            annotate(
              "text",
              x = col_x[5],
              y = row_y[r],
              label = c5,
              hjust = 0.5,
              size = 3.3,
              fontface = ifelse(c5 == "—", "plain", "bold"),
              colour = ifelse(c5 == "—", "grey60", "#1A6B2A")
            ),
            annotate(
              "text",
              x = col_x[6],
              y = row_y[r],
              label = c6,
              hjust = 0.5,
              size = 3.3,
              fontface = ifelse(c6 == "—", "plain", "bold"),
              colour = ifelse(c6 == "—", "grey60", "#1A6B2A")
            )
          )
        },
        seq_len(n_rows),
        cell$c1,
        cell$c2,
        cell$c3,
        cell$c4,
        cell$c5,
        cell$c6,
        SIMPLIFY = FALSE
      )
    } +
    {
      lapply(row_y, function(y) {
        annotate(
          "segment",
          x = 0,
          xend = 1,
          y = y - row_h / 2,
          yend = y - row_h / 2,
          colour = "#CCDDEE",
          linewidth = 0.3
        )
      })
    } +
    {
      lapply(c(0.29, 0.44, 0.58, 0.71, 0.845), function(xv) {
        annotate(
          "segment",
          x = xv,
          xend = xv,
          y = hdr_y - row_h / 2,
          yend = row_y[n_rows] - row_h / 2,
          colour = "#CCDDEE",
          linewidth = 0.3
        )
      })
    } +
    annotate(
      "rect",
      xmin = 0,
      xmax = 1,
      ymin = 0,
      ymax = 1,
      fill = NA,
      colour = "#1F4E79",
      linewidth = 1.0
    ) +
    annotate(
      "text",
      x = 0.02,
      y = 0.025,
      label = "New episode counted when gap between consecutive records exceeds window threshold  ·  Grouped by eid + code",
      hjust = 0,
      size = 2.3,
      colour = "grey55"
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    theme_void() +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(0, 0, 0, 0)
    )
  
  ggsave(
    out_file,
    plot = p,
    width = 10,
    height = 3.8,
    dpi = 200,
    bg = "white"
  )
  
  cat("Saved:", out_file, "\n")
}


# ---- 6. Generate PNG Table 2: Chapter summary --------------

make_chapter_png <- function(df, out_file,
                             title = "Episodes by ICD-10 Disease Chapter",
                             sub   = paste0("Random 1000 participants  ·  eid + code grouping  ·  ",
                                            format(Sys.Date(), "%d %b %Y"))) {
  
  wrap_chapter <- function(x, width = 58) {
    stringr::str_wrap(stringr::str_squish(as.character(x)), width = width)
  }
  
  df <- df %>%
    filter(raw_eps > 0) %>%
    arrange(chapter) %>%
    mutate(chapter_display = as.character(chapter))
  
  col_headers <- c("ICD-10 Chapter", "Raw", "30-day", "60-day", "90-day",
                   "30d %", "60d %", "90d %")
  
  # The full chapter names need a wide first column; numeric columns are
  # kept narrower because their contents are short.
  col_x  <- c(0.17, 0.40, 0.51, 0.62, 0.73, 0.82, 0.895, 0.965)
  
  n_rows <- nrow(df)
  table_top <- 0.89
  table_bottom <- 0.07
  row_h  <- (table_top - table_bottom) / (n_rows + 0.5)
  hdr_y  <- 0.89
  
  row_y <- seq(
    hdr_y - row_h,
    hdr_y - row_h * n_rows,
    length.out = n_rows
  )
  
  cell <- df %>%
    transmute(
      c1 = wrap_chapter(chapter_display),
      c2 = format(raw_eps, big.mark = ","),
      c3 = format(eps_30,  big.mark = ","),
      c4 = format(eps_60,  big.mark = ","),
      c5 = format(eps_90,  big.mark = ","),
      c6 = ifelse(pct_30 == 0, "—", paste0("-", pct_30, "%")),
      c7 = ifelse(pct_60 == 0, "—", paste0("-", pct_60, "%")),
      c8 = ifelse(pct_90 == 0, "—", paste0("-", pct_90, "%"))
    )
  
  red_col <- function(v) ifelse(v == "—", "grey60", "#1A6B2A")
  red_fw  <- function(v) ifelse(v == "—", "plain", "bold")
  
  fs <- max(2.5, min(3.1, 3.1 - (n_rows - 15) * 0.03))
  
  p <- ggplot() +
    annotate("rect", xmin = 0, xmax = 1, ymin = 0.935, ymax = 1,
             fill = "#1F4E79", colour = NA) +
    annotate("text", x = 0.5, y = 0.974, label = title,
             hjust = 0.5, size = 5.0, fontface = "bold", colour = "white") +
    annotate("text", x = 0.5, y = 0.945, label = sub,
             hjust = 0.5, size = 2.7, colour = "#A8C8E8") +
    
    annotate("rect", xmin = 0, xmax = 1,
             ymin = hdr_y - row_h / 2, ymax = hdr_y + row_h / 2,
             fill = "#2E6DA4", colour = NA) +
    annotate("text", x = col_x, y = rep(hdr_y, 8), label = col_headers,
             hjust = 0.5, vjust = 0.5, size = fs + 0.3,
             fontface = "bold", colour = "white", lineheight = 0.85) +
    
    {lapply(seq(1, n_rows, 2), function(r)
      annotate("rect", xmin = 0, xmax = 1,
               ymin = row_y[r] - row_h / 2,
               ymax = row_y[r] + row_h / 2,
               fill = "#EBF3FA", colour = NA))} +
    
    {mapply(function(r, c1, c2, c3, c4, c5, c6, c7, c8) list(
      annotate("text", x = 0.02, y = row_y[r], label = c1,
               hjust = 0, vjust = 0.5, size = fs, colour = "grey15",
               lineheight = 0.9),
      annotate("text", x = col_x[2], y = row_y[r], label = c2,
               hjust = 0.5, size = fs, colour = "grey15", fontface = "bold"),
      annotate("text", x = col_x[3], y = row_y[r], label = c3,
               hjust = 0.5, size = fs, colour = "grey30"),
      annotate("text", x = col_x[4], y = row_y[r], label = c4,
               hjust = 0.5, size = fs, colour = "grey30"),
      annotate("text", x = col_x[5], y = row_y[r], label = c5,
               hjust = 0.5, size = fs, colour = "grey30"),
      annotate("text", x = col_x[6], y = row_y[r], label = c6,
               hjust = 0.5, size = fs, colour = red_col(c6), fontface = red_fw(c6)),
      annotate("text", x = col_x[7], y = row_y[r], label = c7,
               hjust = 0.5, size = fs, colour = red_col(c7), fontface = red_fw(c7)),
      annotate("text", x = col_x[8], y = row_y[r], label = c8,
               hjust = 0.5, size = fs, colour = red_col(c8), fontface = red_fw(c8))
    ), seq_len(n_rows),
    cell$c1, cell$c2, cell$c3, cell$c4, cell$c5,
    cell$c6, cell$c7, cell$c8, SIMPLIFY = FALSE)} +
    
    {lapply(row_y, function(y)
      annotate("segment", x = 0, xend = 1,
               y = y - row_h / 2, yend = y - row_h / 2,
               colour = "#CCDDEE", linewidth = 0.25))} +
    
    {lapply(c(0.34, 0.46, 0.57, 0.68, 0.78, 0.86, 0.93), function(xv)
      annotate("segment", x = xv, xend = xv,
               y = hdr_y - row_h / 2,
               yend = row_y[n_rows] - row_h / 2,
               colour = "#CCDDEE", linewidth = 0.25))} +
    
    annotate("segment", x = 0.34, xend = 0.34,
             y = hdr_y - row_h / 2,
             yend = row_y[n_rows] - row_h / 2,
             colour = "#8AAFC8", linewidth = 0.6) +
    
    annotate("segment", x = 0.78, xend = 0.78,
             y = hdr_y - row_h / 2,
             yend = row_y[n_rows] - row_h / 2,
             colour = "#8AAFC8", linewidth = 0.6) +
    
    annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1,
             fill = NA, colour = "#1F4E79", linewidth = 1.0) +
    
    annotate("text", x = 0.02, y = 0.018,
             label = "New episode counted when gap between consecutive records exceeds window threshold  ·  Grouped by eid + code",
             hjust = 0, size = 2.1, colour = "grey55") +
    
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    theme_void() +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(0, 0, 0, 0)
    )
  
  ggsave(
    out_file,
    plot = p,
    width = 16,
    height = max(9.5, n_rows * 0.45 + 1.5),
    dpi = 200,
    bg = "white"
  )
  
  cat("Saved:", out_file, "\n")
}

# ---- 7. Save PNGs ------------------------------------------

out_dir <- "/rds/general/user/amk125/home/delphi-amk125_model/final_outputs/time_window/"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

out_path <- file.path(out_dir, "time_window_comparison_table.png")
out_chap <- file.path(out_dir, "time_window_chapter_table.png")

make_summary_png(overall_summary, out_path)
make_chapter_png(chapter_summary_wide, out_chap)

cat("\nDone. Objects: overall_summary, chapter_summary_wide\n")


# ---- 8. free memory ------------------------------

clinical_history_icd <- NULL
df_sample <- NULL
gc()
