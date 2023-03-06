#!/bin/bash -eux
#SBATCH --job-name=captioning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<EMAIL>
#SBATCH --partition=<PARTITION_NAME> # -p
#SBATCH --gpus=<NUMBER>
#SBATCH --account=<ACCOUNT_NAME> # -A
#SBATCH --mem=<NUMBER>G
 
# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate lavis
set +eu
 
cd LAVIS
python train.py --cfg-path lavis/projects/blip/train/artpedia.yaml