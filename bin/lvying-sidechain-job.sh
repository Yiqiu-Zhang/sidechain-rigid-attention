#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --output=output.log
#SBATCH --error=error.log
python train.py /mnt/petrelfs/lvying/code/sidechain-rigid-attention/config_jsons/cath_full_angles_cosine.json --dryrun -o resutl_4_20