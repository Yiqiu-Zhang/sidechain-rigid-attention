#!/bin/bash
#SBATCH --job-name=distance
#SBATCH --output=output_0.0005.log
#SBATCH --error=error_0.0005.log
python train.py /mnt/petrelfs/lvying/code/sidechain-rigid-attention/config_jsons/cath_full_angles_cosine-1.json --dryrun -o resutl_5_9_v2