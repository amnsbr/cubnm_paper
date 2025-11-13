#!/bin/bash

if [[ $(hostname) != raven* ]]; then
    echo "This script is intended to be run on the Raven cluster."
    exit 1
fi

EXP_DIR=$(dirname $0)
cd $EXP_DIR
PROJECT_DIR=$(realpath "$(pwd)/../..")
venv_dir="${PROJECT_DIR}/venv"

CUBNM_DIR="${PROJECT_DIR}/../dev" # the cubnm development repo on Raven

# activate the conda environment for gsl
module load anaconda/3/2023.03
eval "$(conda shell.bash hook)" && \
    conda activate ~/projects/cubnm/env

# install gsl
[[ $(conda list | grep "gsl") ]] || conda install gsl==2.7

## make it available in LIBRARY_PATH
if [ -z LIBRARY_PATH ]; then
    export LIBRARY_PATH="$(realpath ~/projects/cubnm/env/lib)"
else
    export LIBRARY_PATH="$(realpath ~/projects/cubnm/env/lib):$LIBRARY_PATH"
fi

# deactivate conda
conda deactivate


# activate the venv
source $venv_dir/bin/activate 

# load cuda
module load cuda/11.8-nvhpcsdk

cd $CUBNM_DIR
python -m pip install build
python -m build .
python -m pip uninstall -y cubnm
python -m pip install $(ls -tr ./dist/*.whl | tail -n 1) # installs latest built wheel

python -m pip install cupy-cuda12x numba-cuda[cu12]