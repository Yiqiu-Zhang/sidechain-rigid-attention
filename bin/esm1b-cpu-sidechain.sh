#!/bin/bash
#SBATCH --job-name=esm3B_cpu_job
#SBATCH --output=output_esm.log
#SBATCH --error=error_esm.log
python train.py /mnt/petrelfs/lvying/code/sidechain-diffusion/config_jsons/cath_full_angles_cosine.json --dryrun  -o resutl_esm3b                        
