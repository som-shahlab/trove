#!/bin/bash

#SBATCH --job-name=chemical-bert
#SBATCH --output chemical-bert.log.%j
#SBATCH --error chemical-bert.err.%j
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --ntasks=1   
#SBATCH --mem 32gb

CONDA_ENV="source activate /share/pi/nigam/envs/transformers/"

srun bash -c "${CONDA_ENV} ; cd /share/pi/nigam/projects/jfries/code/inkNER ; ./chemical_end_model.sh"
