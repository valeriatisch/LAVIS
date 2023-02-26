#!/bin/bash -eux
#SBATCH --job-name=captioning-blip
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<MAIL>
#SBATCH --partition=<PARTITION_NAME> # -p
#SBATCH --account=<ACCOUNT_NAME> # -A
#SBATCH --mem-per-cpu=<NUMBER>G

eval "$(conda shell.bash hook)"
set +eu
source activate <ENV>
set +eu

python predict.py
