#!/bin/bash
#SBATCH --job-name=fish_exp19
#SBATCH --output=output_exp19.log
#SBATCH --error=error_exp19.log
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=rome
#SBATCH --account=gusr125350


module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source thesis_env/bin/activate

python Run_exp19.py