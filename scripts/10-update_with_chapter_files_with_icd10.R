# 10-update_with_chapter_files_with_icd10.R
#
# Adds specific ICD-10 code columns (matched_code, matched_meaning, match_type)
# directly onto the existing per-person "with_chapter" self-reported datasets,
# for BOTH non-cancer and cancer, and OVERWRITES them in place.
#
# Temporary files are written beside each destination RDS file. This avoids
# relying on an ephemeral filesystem that may be read-only on compute nodes.
#
# Final RDS files are stored in:
#   /rds/general/project/hda_24-25/live/amk125_thesis/outputs/outputs_with_age
#
# Non-cancer uses the existing category-level mapping from:
#   scripts/self_reported_to_icd10_specific_mapping.csv
#
# Cancer categories are matched against coding19.tsv within:
#   Chapter II Neoplasms
#
# Matching order:
#   1. Exact
#   2. Contains
#   3. Fuzzy
#   4. Chapter-level "other" or "unspecified" fallback

library(data.table)

# -------------------------------------------------------------------------
# Directories and input/output paths
# -------------------------------------------------------------------------

base_dir <- "/rds/general/project/hda_24-25/live/amk125_thesis"

scripts_dir <- file.path(
  base_dir,
  "scripts"
)

outputs_with_age_dir <- file.path(
  base_dir,
  "outputs",
  "outputs_with_age"
)

coding19_path <- file.path(
  scripts_dir,
  "coding19.tsv"
)

noncancer_mapping_path <- file.path(
  scripts_dir,
  "self_reported_to_icd10_specific_mapping.csv"
)

noncancer_path <- file.path(
  outputs_with_age_dir,
  "seq_noncancer_self_reported_with_chapter.rds"
)

cancer_path <- file.path(
  outputs_with_age_dir,
  "seq_cancer_self_reported_with_chapter.rds"
)

# Check actual write access rather than only checking whether the directory
# exists. A directory can exist but be mounted read-only on a compute node.
if (!dir.exists(outputs_with_age_dir)) {
  stop("Output directory does not exist: ", outputs_with_age_dir)
}

write_test <- tempfile(
  pattern = ".icd10_write_test_",
  tmpdir = outputs_with_age_dir
)

if (!file.create(write_test)) {
  stop("Output directory is not writable: ", outputs_with_age_dir)
}

unlink(write_test)

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

normalize_text <- function(x) {
  x <- tolower(x)
  x <- gsub("[^a-z0-9 ]", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

strip_code_prefix <- function(meaning) {
  sub(
    "^[A-Za-z][0-9]{2}(\\.[0-9]+)?\\s+",
    "",
    meaning
  )
}

chapter_key <- function(x) {
  ifelse(
    grepl("^Chapter\\s+[IVXLCDM]+\\b", x),
    sub(
      "^(Chapter\\s+[IVXLCDM]+)\\b.*",
      "\\1",
      x
    ),
    NA_character_
  )
}

# -------------------------------------------------------------------------
# Load and prepare the ICD-10 hierarchy
# -------------------------------------------------------------------------

cat("Loading coding19.tsv...\n")

cd19 <- fread(
  coding19_path,
  colClasses = "character"
)

cd19[, node_id := as.integer(node_id)]
cd19[, parent_id := as.integer(parent_id)]

cd19[, desc := strip_code_prefix(meaning)]
cd19[, desc_norm := normalize_text(desc)]

# Create a lookup from node ID to parent node ID
parent_lookup <- setNames(
  cd19$parent_id,
  cd19$node_id
)

# Follow the hierarchy until the top-level chapter node is reached
get_chapter_node <- function(node_id) {
  cur <- node_id
  
  repeat {
    p <- parent_lookup[as.character(cur)]
    
    if (is.na(p) || p == 0L) {
      break
    }
    
    cur <- p
  }
  
  cur
}

cat("Computing chapter ancestor for each ICD-10 node...\n")

cd19[, chapter_node_id := vapply(
  node_id,
  get_chapter_node,
  integer(1)
)]

# Get the names of the top-level ICD-10 chapters
chapter_nodes <- cd19[
  parent_id == 0L,
  .(
    chapter_node_id = node_id,
    chapter_meaning = meaning
  )
]

# Add chapter information to every ICD-10 row
cd19 <- merge(
  cd19,
  chapter_nodes,
  by = "chapter_node_id",
  all.x = TRUE
)

cd19[, chapter_key := chapter_key(chapter_meaning)]

# Treat codes without a decimal point as block-level codes
is_block_level <- function(coding) {
  !grepl("\\.", coding)
}

# -------------------------------------------------------------------------
# Fallback matching
# -------------------------------------------------------------------------

find_other_fallback <- function(chapter_key_val) {
  chapter_rows <- cd19[
    chapter_key == chapter_key_val &
      selectable == "Y"
  ]
  
  # First try a block-level code containing "other"
  other_rows <- chapter_rows[
    grepl("\\bother\\b", desc_norm) &
      is_block_level(coding)
  ]
  
  # Then try any selectable code containing "other"
  if (nrow(other_rows) == 0) {
    other_rows <- chapter_rows[
      grepl("\\bother\\b", desc_norm)
    ]
  }
  
  # Then try a block-level code containing "unspecified"
  if (nrow(other_rows) == 0) {
    other_rows <- chapter_rows[
      grepl("\\bunspecified\\b", desc_norm) &
        is_block_level(coding)
    ]
  }
  
  # Finally try any selectable code containing "unspecified"
  if (nrow(other_rows) == 0) {
    other_rows <- chapter_rows[
      grepl("\\bunspecified\\b", desc_norm)
    ]
  }
  
  if (nrow(other_rows) == 0) {
    return(
      data.table(
        coding = NA_character_,
        meaning = NA_character_
      )
    )
  }
  
  other_rows <- other_rows[order(node_id)]
  
  other_rows[
    .N,
    .(
      coding,
      meaning
    )
  ]
}

# -------------------------------------------------------------------------
# Match a category to a selectable ICD-10 code within one chapter
# -------------------------------------------------------------------------

match_in_chapter <- function(category_norm_val, chapter_key_val) {
  if (is.na(chapter_key_val)) {
    return(
      list(
        code = NA_character_,
        meaning = NA_character_,
        match_type = "no_chapter_candidates"
      )
    )
  }
  
  candidates <- cd19[
    chapter_key == chapter_key_val &
      selectable == "Y"
  ]
  
  if (nrow(candidates) == 0) {
    return(
      list(
        code = NA_character_,
        meaning = NA_character_,
        match_type = "no_chapter_candidates"
      )
    )
  }
  
  # 1. Exact match
  exact <- candidates[
    desc_norm == category_norm_val
  ]
  
  if (nrow(exact) > 0) {
    return(
      list(
        code = exact$coding[1],
        meaning = exact$meaning[1],
        match_type = "exact"
      )
    )
  }
  
  # 2. Contains match
  contains <- candidates[
    mapply(
      function(d) {
        grepl(
          category_norm_val,
          d,
          fixed = TRUE
        ) ||
          grepl(
            d,
            category_norm_val,
            fixed = TRUE
          )
      },
      desc_norm
    )
  ]
  
  if (nrow(contains) > 0) {
    contains <- contains[
      order(nchar(desc_norm))
    ]
    
    return(
      list(
        code = contains$coding[1],
        meaning = contains$meaning[1],
        match_type = "contains"
      )
    )
  }
  
  # 3. Fuzzy match
  idx <- agrep(
    category_norm_val,
    candidates$desc_norm,
    max.distance = 0.15
  )
  
  if (length(idx) > 0) {
    return(
      list(
        code = candidates$coding[idx[1]],
        meaning = candidates$meaning[idx[1]],
        match_type = "fuzzy"
      )
    )
  }
  
  # 4. No direct match
  list(
    code = NA_character_,
    meaning = NA_character_,
    match_type = "no_match"
  )
}

# -------------------------------------------------------------------------
# Apply an "other" or "unspecified" fallback
# -------------------------------------------------------------------------

apply_fallback <- function(map_dt, chapter_key_val) {
  fb <- find_other_fallback(chapter_key_val)
  
  needs_fallback <- (
    is.na(map_dt$matched_code) &
      map_dt$match_type != "no_chapter_candidates"
  )
  
  map_dt[
    needs_fallback,
    matched_code := fb$coding
  ]
  
  map_dt[
    needs_fallback,
    matched_meaning := fb$meaning
  ]
  
  map_dt[
    needs_fallback,
    match_type := "chapter_other_fallback"
  ]
  
  still_missing <- is.na(map_dt$matched_code)
  
  map_dt[
    still_missing,
    match_type := "unmapped_needs_manual_review"
  ]
  
  map_dt
}

# -------------------------------------------------------------------------
# Add specific codes to a dataset and save the result
# -------------------------------------------------------------------------

update_with_specific_codes <- function(seq_path, specific_map, label) {
  cat("\n---", label, "---\n")
  
  seq_dt <- as.data.table(
    readRDS(seq_path)
  )
  
  cat(
    "Loaded",
    nrow(seq_dt),
    "rows from",
    seq_path,
    "\n"
  )
  
  # Remove previous matching columns so the script can be rerun
  for (nm in c(
    "matched_code",
    "matched_meaning",
    "match_type"
  )) {
    if (nm %in% names(seq_dt)) {
      seq_dt[, (nm) := NULL]
    }
  }
  
  # Normalize the self-reported category
  seq_dt[, category_norm := normalize_text(code)]
  
  # Join the mapping onto every record
  seq_dt <- specific_map[
    seq_dt,
    on = "category_norm"
  ]
  
  seq_dt[, category_norm := NULL]
  
  
  matched_rows <- seq_dt[
    !is.na(matched_code),
    .N
  ]
  
  cat(
    "Rows with a specific ICD-10 code:",
    matched_rows,
    "of",
    nrow(seq_dt),
    "\n"
  )
  
  # For any records that still have no matched ICD-10 code, fall back to
  # a chapter-level "other/unspecified" code using this row's own chapter,
  # and finally to a generic unspecified-illness code if no chapter-level
  # fallback can be found either.
  still_missing <- is.na(seq_dt$matched_code)
  
  if (any(still_missing)) {
    seq_dt[, chapter_key_val := chapter_key(chapter)]
    
    missing_keys <- unique(
      seq_dt$chapter_key_val[still_missing & !is.na(seq_dt$chapter_key_val)]
    )
    
    if (length(missing_keys) > 0) {
      fallback_lookup <- rbindlist(
        lapply(missing_keys, function(k) {
          fb <- find_other_fallback(k)
          data.table(
            chapter_key_val = k,
            fallback_code = fb$coding,
            fallback_meaning = fb$meaning
          )
        }),
        use.names = TRUE
      )
      
      seq_dt <- fallback_lookup[seq_dt, on = "chapter_key_val"]
      
      use_chapter_fallback <- is.na(seq_dt$matched_code) & !is.na(seq_dt$fallback_code)
      
      seq_dt[use_chapter_fallback, matched_code := fallback_code]
      seq_dt[use_chapter_fallback, matched_meaning := fallback_meaning]
      seq_dt[use_chapter_fallback, match_type := "chapter_other_fallback"]
      
      seq_dt[, c("fallback_code", "fallback_meaning") := NULL]
    }
    
    seq_dt[, chapter_key_val := NULL]
    
    # Final catch-all for anything still unmatched (e.g. unknown chapter)
    still_missing_final <- is.na(seq_dt$matched_code)
    seq_dt[still_missing_final, matched_code := "R69"]
    seq_dt[
      still_missing_final,
      matched_meaning := "Unknown and unspecified causes of morbidity"
    ]
    seq_dt[still_missing_final, match_type := "global_other_fallback"]
  }
  
  # Look up the ICD-10 chapter for the matched code itself (via
  # coding19.tsv), rather than keeping the original self-reported chapter
  code_chapter_lookup <- unique(cd19[, .(coding, chapter_meaning)])
  
  seq_dt[
    code_chapter_lookup,
    on = c(matched_code = "coding"),
    new_chapter := i.chapter_meaning
  ]
  
  # Drop the original code, coding and chapter columns entirely and
  # replace them with the new values derived from coding19.tsv, so the
  # final columns match eid, code, coding, chapter, age
  seq_dt[, code := matched_code]
  seq_dt[, coding := "ICD10"]
  seq_dt[, chapter := new_chapter]
  
  seq_dt <- seq_dt[, .(eid, code, coding, chapter, age)]
  
  # Create a unique temporary file beside the destination. Keeping both files
  # on the same filesystem avoids the read-only ephemeral mount and makes the
  # subsequent replacement more reliable.
  tmp_path <- tempfile(
    pattern = paste0(basename(seq_path), "."),
    tmpdir = dirname(seq_path),
    fileext = ".tmp"
  )
  
  cat(
    "Writing temporary RDS file:",
    tmp_path,
    "\n"
  )
  
  # Write the complete dataset to a temporary file first
  saveRDS(
    seq_dt,
    tmp_path,
    compress = FALSE
  )
  
  if (!file.exists(tmp_path)) {
    stop(
      "Temporary RDS file was not created: ",
      tmp_path
    )
  }
  
  if (file.info(tmp_path)$size == 0) {
    stop(
      "Temporary RDS file is empty: ",
      tmp_path
    )
  }
  
  cat(
    "Copying completed RDS file to:",
    seq_path,
    "\n"
  )
  
  # file.rename() may fail across filesystems, so use file.copy()
  copied <- file.copy(
    from = tmp_path,
    to = seq_path,
    overwrite = TRUE
  )
  
  if (!copied) {
    stop(
      paste0(
        "Failed to copy the temporary RDS file.\n",
        "Temporary file remains at: ",
        tmp_path,
        "\n",
        "Final destination was: ",
        seq_path
      )
    )
  }
  
  # Confirm the final file exists
  if (!file.exists(seq_path)) {
    stop(
      "Final RDS file does not exist after copying: ",
      seq_path
    )
  }
  
  # Delete the temporary file after a successful copy
  removed <- file.remove(tmp_path)
  
  if (!removed) {
    warning(
      "Final file was saved, but the temporary file could not be removed: ",
      tmp_path
    )
  }
  
  cat(
    "Overwrote final file:",
    seq_path,
    "\n"
  )
  
  seq_dt
}

# -------------------------------------------------------------------------
# Non-cancer processing
# -------------------------------------------------------------------------

cat(
  "Loading non-cancer category to ICD-10 mapping CSV...\n"
)

noncancer_map <- fread(
  noncancer_mapping_path,
  colClasses = "character"
)

noncancer_map[
  ,
  category_norm := normalize_text(category)
]

specific_noncancer <- unique(
  noncancer_map[
    ,
    .(
      category_norm,
      matched_code,
      matched_meaning,
      match_type
    )
  ]
)

# Identify normalized categories with multiple mappings
dup_nc <- specific_noncancer[
  ,
  .N,
  by = category_norm
][
  N > 1,
  category_norm
]

# Keep the first mapping where duplicates exist
if (length(dup_nc) > 0) {
  specific_noncancer <- specific_noncancer[
    ,
    .SD[1],
    by = category_norm
  ]
}

seq_noncancer <- update_with_specific_codes(
  seq_path = noncancer_path,
  specific_map = specific_noncancer,
  label = "Non-cancer"
)

# -------------------------------------------------------------------------
# Cancer processing
# -------------------------------------------------------------------------

cat(
  "\nBuilding cancer category to ICD-10 mapping ",
  "(Chapter II Neoplasms only)...\n",
  sep = ""
)

seq_cancer_raw <- as.data.table(
  readRDS(cancer_path)
)

cancer_categories <- unique(
  as.character(seq_cancer_raw$code)
)

cancer_categories <- cancer_categories[
  !is.na(cancer_categories) &
    cancer_categories != ""
]

chapterII_key <- "Chapter II"

cancer_map <- data.table(
  category = cancer_categories
)

cancer_map[
  ,
  category_norm := normalize_text(category)
]

cancer_results <- vector(
  "list",
  nrow(cancer_map)
)

for (i in seq_len(nrow(cancer_map))) {
  cancer_results[[i]] <- match_in_chapter(
    category_norm_val = cancer_map$category_norm[i],
    chapter_key_val = chapterII_key
  )
}

cancer_map[
  ,
  matched_code := vapply(
    cancer_results,
    function(r) r$code,
    character(1)
  )
]

cancer_map[
  ,
  matched_meaning := vapply(
    cancer_results,
    function(r) r$meaning,
    character(1)
  )
]

cancer_map[
  ,
  match_type := vapply(
    cancer_results,
    function(r) r$match_type,
    character(1)
  )
]

cancer_map <- apply_fallback(
  cancer_map,
  chapterII_key
)

cat("Cancer category match summary:\n")

print(
  cancer_map[
    ,
    .N,
    by = match_type
  ][
    order(-N)
  ]
)

specific_cancer <- unique(
  cancer_map[
    ,
    .(
      category_norm,
      matched_code,
      matched_meaning,
      match_type
    )
  ]
)

# Identify normalized cancer categories with multiple mappings
dup_c <- specific_cancer[
  ,
  .N,
  by = category_norm
][
  N > 1,
  category_norm
]

# Keep the first mapping where duplicates exist
if (length(dup_c) > 0) {
  specific_cancer <- specific_cancer[
    ,
    .SD[1],
    by = category_norm
  ]
}

seq_cancer <- update_with_specific_codes(
  seq_path = cancer_path,
  specific_map = specific_cancer,
  label = "Cancer"
)

cat(
  "\nDone. Both with_chapter.rds files were updated ",
  "with specific ICD-10 codes.\n",
  sep = ""
)

cat(
  "Final files are located in:",
  outputs_with_age_dir,
  "\n"
)

cat(
  "Temporary files were written beside the final files and removed after ",
  "successful copying.\n",
  sep = ""
)
