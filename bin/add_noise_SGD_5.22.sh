#!/bin/bash
#SBATCH --job-name=add_angle_noise_SGD
#SBATCH --output=output_add_angle_noise_SGD_5_22.log
#SBATCH --error=error_add_angle_noise_SGD__5_22.log
python train.py /mnt/petrelfs/lvying/code/sidechain-rigid-attention/config_jsons/cath_full_angles_cosine.json --dryrun -o resutl_5_22