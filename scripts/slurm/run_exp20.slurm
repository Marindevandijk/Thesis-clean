#!/bin/bash
#SBATCH --job-name=fish_exp20
#SBATCH --output=output_exp20.log
#SBATCH --error=error_exp20.log
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=rome
#SBATCH --account=gusr125350


module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source thesis_env/bin/activate

python Run_exp20.py