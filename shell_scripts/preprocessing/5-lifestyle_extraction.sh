#!/bin/bash
#PBS -N lifestyle_extraction
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/lifestyle_extraction.stdout
#PBS -e /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/lifestyle_extraction.stderr

cd /rds/general/project/hda_24-25/live/amk125_thesis/scripts

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate r413

Rscript 5-lifestyle_extraction.R
