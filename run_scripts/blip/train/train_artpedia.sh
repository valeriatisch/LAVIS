#!/bin/bash -eux
#SBATCH --job-name=captioning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smilla.fox@student.hpi.uni-potsdam.de
#SBATCH --partition=sorcery # -p
#SBATCH --gpus=1
#SBATCH --account=naumann # -A
#SBATCH --mem=48G
 
# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate lavis
set +eu
 
cd LAVIS
python train.py --cfg-path lavis/projects/blip/train/artpedia.yaml