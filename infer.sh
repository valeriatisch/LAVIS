#!/bin/bash -eux
#SBATCH --job-name=captioning-blip
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elena.gensch@student.hpi.uni-potsdam.de
#SBATCH --partition=magic # -p
#SBATCH --account=naumann-mpws2022fn1 # -A
#SBATCH --mem-per-cpu=32G

eval "$(conda shell.bash hook)"
set +eu
source activate lavis-blip
set +eu

python predict.py
