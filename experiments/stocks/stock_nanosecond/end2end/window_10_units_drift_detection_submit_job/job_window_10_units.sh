#!/bin/bash
#SBATCH --job-name=window_10_drift
#SBATCH --mail-user=jinyli@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=64g
#SBATCH --time=10:00:00
#SBATCH --account=jag0
#SBATCH --partition=gpu
#SBATCH --output=./window_10_drift.log

# Load required modules
module load python3.9-anaconda/2021.11

# Create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install river
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn

# Run the Python script
srun python training_opt.py

# Deactivate the virtual environment
deactivate


