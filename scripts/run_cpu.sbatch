#!/bin/bash -l
# Runs a Python script on a single CPU thread
# Usage: sbatch run_gpu.sbatch <path/to/run.py> <args>
# Call from PROJECT_DIR so that logs are saved in PROJECT_DIR/logs/
# Note: This is specific to the Raven cluster
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH -D ./
# Job name
#SBATCH -J cubnm_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=144

if [ -z $PROJECT_DIR ]; then
    echo "Please set the PROJECT_DIR environment variable."
    exit 1
fi

cd $PROJECT_DIR

VENV_DIR="${PROJECT_DIR}/venv"

# activate the environment
module load anaconda/3/2023.03
source $VENV_DIR/bin/activate
srun $VENV_DIR/bin/python ${@}