library(data.table)

in_path  <- "/rds/general/project/hda_24-25/live/ukb_general_data/data_external.rds"
out_dir  <- "/rds/general/project/hda_24-25/live/amk125_thesis/outputs/sources"
out_path <- file.path(out_dir, "demographics.rds")

keep_cols <- c("smoke_status", "bmi", "employment_status",
               "ethnicity", "edu", "sex", "alcohol_category")

# Read the wide data.frame (eid as row names)
df <- readRDS(in_path)

# Make sure the requested columns are actually in data_external
missing <- setdiff(keep_cols, colnames(df))
if (length(missing) > 0) {
  warning("Skipping columns not found in data_external: ",
          paste(missing, collapse = ", "))
  keep_cols <- intersect(keep_cols, colnames(df))
}

# Pull row names into 'eid' and keep only the present columns
setDT(df, keep.rownames = "eid")
df <- df[, c("eid", keep_cols), with = FALSE]

# Wide -> long
long_dt <- melt(
  df,
  id.vars         = "eid",
  variable.name   = "coding",
  value.name      = "code",
  variable.factor = FALSE,
  na.rm           = TRUE
)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(long_dt, out_path)

long_dt = NULL
df = NULL
gc()