#!/bin/bash -eux
#SBATCH --account=<ACCOUNT_NAME> # -A
#SBATCH --job-name=filter
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<MAIL>
#SBATCH --partition=<PARTITION_NAME> # -p
#SBATCH --gpus=<NUMBER>
#SBATCH --mem=<NUMBER>G

# Initialize conda:
eval "$(conda shell.bash hook)"
set +eu
conda activate <ENV>
set +eu
python caption_matching.py path_to_images path_to_annotations_json path_to_annotations_json_with_scores
