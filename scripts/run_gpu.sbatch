#!/bin/bash -l
# Runs a Python script on a single GPU
# Usage: sbatch run_gpu.sbatch <path/to/run.py> <args>
# Call from PROJECT_DIR so that logs are saved in PROJECT_DIR/logs/
# Note: This is specific to the Raven cluster
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH -D ./
#SBATCH -J cubnm_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=24:00:00

if [ -z $PROJECT_DIR ]; then
    echo "Please set the PROJECT_DIR environment variable."
    exit 1
fi

cd $PROJECT_DIR

VENV_DIR="${PROJECT_DIR}/venv"

# activate the environment
module load anaconda/3/2023.03 cuda/11.8-nvhpcsdk
source $VENV_DIR/bin/activate
srun $VENV_DIR/bin/python ${@}