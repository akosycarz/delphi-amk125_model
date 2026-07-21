library(data.table)
library(lubridate)

tmp_dir <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/tmp"
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)
Sys.setenv(TMPDIR = tmp_dir)
base_path <- "/rds/general/project/hda_24-25/live/TDS/General/Data/"
out_path <- "/rds/general/ephemeral/user/amk125/ephemeral/amk125_thesis/outputs/sources/"
dir.create(out_path, showWarnings = FALSE, recursive = TRUE)

# Helper: keep first occurrence per (eid, code, coding)
first_occurrence <- function(dt) {
  dt[, .(date = min(date, na.rm = TRUE)), by = .(eid, code, coding)]
}

# ============================================================
# Helper: ICD-10 3-char code -> UKB ICD-10 chapter label
# ============================================================
get_icd10_chapter <- function(code) {
  first_char <- substr(code, 1, 1)
  num_part   <- suppressWarnings(as.integer(substr(code, 2, 3)))
  
  fcase(
    first_char %in% c("A", "B"),
    "Chapter I Certain infectious and parasitic diseases",
    first_char == "C" | (first_char == "D" & !is.na(num_part) & num_part <= 48),
    "Chapter II Neoplasms",
    first_char == "D" & !is.na(num_part) & num_part >= 50,
    "Chapter III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    first_char == "E",
    "Chapter IV Endocrine, nutritional and metabolic diseases",
    first_char == "F",
    "Chapter V Mental and behavioural disorders",
    first_char == "G",
    "Chapter VI Diseases of the nervous system",
    first_char == "H" & !is.na(num_part) & num_part <= 59,
    "Chapter VII Diseases of the eye and adnexa",
    first_char == "H" & !is.na(num_part) & num_part >= 60,
    "Chapter VIII Diseases of the ear and mastoid process",
    first_char == "I",
    "Chapter IX Diseases of the circulatory system",
    first_char == "J",
    "Chapter X Diseases of the respiratory system",
    first_char == "K",
    "Chapter XI Diseases of the digestive system",
    first_char == "L",
    "Chapter XII Diseases of the skin and subcutaneous tissue",
    first_char == "M",
    "Chapter XIII Diseases of the musculoskeletal system and connective tissue",
    first_char == "N",
    "Chapter XIV Diseases of the genitourinary system",
    first_char == "O",
    "Chapter XV Pregnancy, childbirth and the puerperium",
    first_char == "P",
    "Chapter XVI Certain conditions originating in the perinatal period",
    first_char == "Q",
    "Chapter XVII Congenital malformations, deformations and chromosomal abnormalities",
    first_char == "R",
    "Chapter XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
    first_char %in% c("S", "T"),
    "Chapter XIX Injury, poisoning and certain other consequences of external causes",
    first_char %in% c("V", "W", "X", "Y"),
    "Chapter XX External causes of morbidity and mortality",
    first_char == "Z",
    "Chapter XXI Factors influencing health status and contact with health services",
    first_char == "U",
    "Chapter XXII Codes for special purposes",
    default = NA_character_
  )
}


# ============================================================
# 1. HES admission dates (needed by sections 1 & 2)
# ============================================================
hesin <- fread(paste0(base_path, "hesin.txt"),
               select = c("eid", "ins_index", "epistart", "admidate"),
               colClasses = list(character = c("eid", "ins_index")))
hesin[, date := as.Date(ifelse(!is.na(epistart) & epistart != "", epistart, admidate),
                        format = "%d/%m/%Y")]
hesin <- hesin[!is.na(date), .(eid, ins_index, date)]

# ============================================================
# 2. ICD-10 from HES
# ============================================================
cat("Loading ICD-10...\n")
hesin_diag <- fread(paste0(base_path, "hesin_diag.txt"),
                    select = c("eid", "ins_index", "diag_icd10"),
                    colClasses = list(character = c("eid", "ins_index")))

icd10 <- merge(hesin_diag, hesin, by = c("eid", "ins_index"), all.x = TRUE)
icd10 <- icd10[!is.na(diag_icd10) & diag_icd10 != "" & !is.na(date)]
icd10 <- icd10[, .(eid, code = substr(diag_icd10, 1, 3), coding = "ICD10", date)]

cat("ICD-10:", nrow(icd10), "\n")
rm(hesin_diag); gc()

# ============================================================
# 3. ICD-9 from HES
# ============================================================
cat("Loading ICD-9...\n")
hesin_diag9 <- fread(paste0(base_path, "hesin_diag.txt"),
                     select = c("eid", "ins_index", "diag_icd9"),
                     colClasses = list(character = c("eid", "ins_index")))

icd9 <- merge(hesin_diag9, hesin, by = c("eid", "ins_index"), all.x = TRUE)
icd9 <- icd9[!is.na(diag_icd9) & diag_icd9 != "" & !is.na(date)]
icd9 <- icd9[, .(eid, code = substr(diag_icd9, 1, 3), coding = "ICD9", date)]
cat("ICD-9:", nrow(icd9), "\n")
rm(hesin_diag9); gc()

# ============================================================
# 4. OPCS-4 from HES
# ============================================================
cat("Loading OPCS-4...\n")
hesin_oper <- fread(paste0(base_path, "hesin_oper.txt"),
                    select = c("eid", "ins_index", "opdate", "oper3"),
                    colClasses = list(character = c("eid", "ins_index", "oper3")))

opcs4 <- merge(hesin_oper, hesin, by = c("eid", "ins_index"), all.x = TRUE)
opcs4[, date := as.Date(ifelse(!is.na(opdate) & opdate != "", opdate, as.character(date)),
                        format = "%d/%m/%Y")]
opcs4 <- opcs4[!is.na(oper3) & oper3 != "" & !is.na(date)]
opcs4 <- opcs4[, .(eid, code = substr(oper3, 1, 3), coding = "OPCS4", date)]
cat("OPCS-4:", nrow(opcs4), "\n")
rm(hesin, hesin_oper); gc()

# ============================================================
# 5. Death registry
# ============================================================
cat("Loading death data...\n")
death <- fread(paste0(base_path, "death.txt"), colClasses = list(character = "eid"))

death <- death[, .(
  eid,
  code   = "DEATH",
  coding = "DEATH",
  date = as.Date(date_of_death, format = "%d/%m/%Y")
)]
death <- death[!is.na(date)]
death<- first_occurrence(death)
cat("Death records:", nrow(death), "\n")
gc()

# ============================================================
# 6  COVID-19
# ============================================================
cat("Loading COVID-19 data...\n")
covid <- readRDS("/rds/general/project/hda_24-25/live/ukb_general_data/covid_data_merged.rds")
setDT(covid)
cat("COVID-19 columns:", paste(colnames(covid), collapse = ", "), "\n")

covid19 <- covid[result_imputed == 1, .(
  eid,
  code   = "U07", # this is the icd10 code for covid Tanno, L.K., Casale, T. and Demoly, P. (2020) “Coronavirus Disease (COVID)-19: World Health Organization Definitions and Coding to Support the Allergy Community and Health Professionals,” The Journal of Allergy and Clinical Immunology. in Practice, 8(7), pp. 2144–2148. Available at: https://doi.org/10.1016/j.jaip.2020.05.002.
  coding = "ICD10",
  date   = as.Date(specdate_imputed, format = "%Y-%m-%d")
)]
covid19 <- covid19[!is.na(date)]
cat("COVID-19 positive occurrences:", nrow(covid19), "\n")
rm(covid); gc()
# ============================================================
# 6. Combine and save
# ============================================================
all_diagnoses <- rbindlist(list(icd10, icd9, opcs4, death, covid19))
cat("Total records:", nrow(all_diagnoses), "\n")
print(all_diagnoses[, .N, by = coding])

# ICD-10 chapter (NA for ICD-9, OPCS4, DEATH -- no direct mapping)
all_diagnoses[, chapter := NA_character_]
all_diagnoses[coding == "ICD10", chapter := get_icd10_chapter(code)]
 
saveRDS(all_diagnoses, paste0(out_path, "clinical_history_icd.rds"))
cat("Saved clinical_history_icd.rds\n")