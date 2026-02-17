#!/bin/bash
#SBATCH --job-name=mito_sim
#SBATCH --output=slurm_outputs/mito_sim_%j.out
#SBATCH --error=slurm_outputs/mito_sim_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

date
echo "Running on node: $(hostname)"

# Activate environment if needed (edit/remove as appropriate)
# source ~/.bashrc
# conda activate myenv

# Move to submission directory
cd $SLURM_SUBMIT_DIR

# Run simulation
python main.py

date

