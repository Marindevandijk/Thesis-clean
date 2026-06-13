#!/bin/bash
#SBATCH --job-name=fish_expall
#SBATCH --output=output_expall_pi2.log
#SBATCH --error=error_expall_order_pi2.log
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=rome
#SBATCH --account=gusr125350


module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source thesis_env/bin/activate

python -u Run_25%_basis_pi2_nosparsify.py