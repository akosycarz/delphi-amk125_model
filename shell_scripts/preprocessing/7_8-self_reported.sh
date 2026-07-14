#!/bin/bash
#PBS -N self_reported
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/self_reported.stdout
#PBS -e /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/self_reported.stderr

set -e

cd /rds/general/project/hda_24-25/live/amk125_thesis/scripts

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate r413

Rscript 7-self_reported.R
Rscript 8-self_reported_chapter_dedup.R