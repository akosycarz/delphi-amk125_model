library(data.table)

set.seed(42L)

input_path <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/outputs_with_age/final.rds"
output_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/outputs_with_age/splits"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

final <- as.data.table(readRDS(input_path))

if (!"eid" %in% names(final)) {
  stop("final.rds does not contain an 'eid' column.")
}

# One fixed 80% / 20% participant split
eids <- unique(final$eid)
train_eids <- sample(eids, size = floor(0.80 * length(eids)))

final_train <- final[eid %in% train_eids]
final_val   <- final[!eid %in% train_eids]

# Safety checks
stopifnot(length(intersect(unique(final_train$eid), unique(final_val$eid))) == 0L)
stopifnot(nrow(final_train) + nrow(final_val) == nrow(final))

saveRDS(
  final_train,
  file.path(output_dir, "final_train.rds"),
  compress = FALSE
)

saveRDS(
  final_val,
  file.path(output_dir, "final_val.rds"),
  compress = FALSE
)

cat("Participants:\n")
cat("  train:", uniqueN(final_train$eid), "\n")
cat("  validation:", uniqueN(final_val$eid), "\n")

cat("\nRows/events:\n")
cat("  train:", nrow(final_train), "\n")
cat("  validation:", nrow(final_val), "\n")