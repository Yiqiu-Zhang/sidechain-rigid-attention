#!/bin/bash
#SBATCH --job-name=add_angle_noise
#SBATCH --output=output_add_angle_noise_5_15.log
#SBATCH --error=error_add_angle_noise_5_15.log
python train.py /mnt/petrelfs/lvying/code/sidechain-rigid-attention/config_jsons/cath_full_angles_cosine-1.json --dryrun -o resutl_5_15