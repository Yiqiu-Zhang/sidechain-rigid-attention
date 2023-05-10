#!/bin/bash
#SBATCH --job-name=distance
#SBATCH --output=output_dis.log
#SBATCH --error=error_dis.log
python train.py /mnt/petrelfs/lvying/code/sidechain-rigid-attention/config_jsons/cath_full_angles_cosine.json --dryrun -o resutl_5_9