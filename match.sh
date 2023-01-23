#!/bin/bash -eux
#SBATCH --account=naumann
#SBATCH --job-name=filter
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<MAIL>
#SBATCH --partition=sorcery # -p
#SBATCH --gpus=1
#SBATCH --mem=48G

# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate <ENV>
set +eu
python caption_matching.py
