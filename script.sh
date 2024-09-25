#!/bin/bash
#SBATCH --job-name=example_job          # Job name
#SBATCH --output=output_%j.txt          # Standard output and error log (%j = JobID)
#SBATCH --error=error_%j.txt            # Error file (%j = JobID)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --time=010:00:00                 # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:8                  # Request 1 GPU (if needed)


# Activate any virtual environments (if using a custom Python environment)
source /raid/scratch/gourishanker/Presentation/Placements/GANs/GANS_env/bin/activate

# Run the script passed as an argument
python "$1"