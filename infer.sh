#!/bin/bash -eux
#SBATCH --job-name=captioning-blip
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<MAIL>
#SBATCH --partition=<PARTITION_NAME> # -p
#SBATCH --account=<ACCOUNT_NAME> # -A
#SBATCH --mem-per-cpu=<NUMBER>G
#SBATCH --gpus=1


eval "$(conda shell.bash hook)"
set +eu
source activate blip
set +eu

# add optional arguments: --max_length=<integer>, --num_captions=<int>, --model_type=<str>, --use_nucleus_sampling=<bool>
python predict.py --file_path=<str - path to image> -force_words <str list, e.g: 'a' 'wood'>
