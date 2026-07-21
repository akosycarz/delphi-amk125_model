library(data.table)

in_path  <- "/rds/general/project/hda_24-25/live/ukb_general_data/data_internal.rds"
out_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sources"
out_path <- file.path(out_dir, "blood_biochemistry.rds")

# Read the wide data.frame (eid as row names, coding as column names)
df <- readRDS(in_path)

# Convert by reference and pull row names into an 'eid' column.
# setDT modifies in place (no copy), which is the cheaper option for large data.
setDT(df, keep.rownames = "eid")

# Wide -> long: every cell becomes one row
long_dt <- melt(
  df,
  id.vars        = "eid",
  variable.name  = "coding",   # from the column names
  value.name     = "code",     # the cell value
  variable.factor = FALSE      # keep 'coding' as character, not factor
)

long_dt <- na.omit(long_dt)
# Save
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(long_dt, out_path)

long_dt = NULL
df = NULL
gc()