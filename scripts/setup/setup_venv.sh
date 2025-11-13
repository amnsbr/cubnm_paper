#!/bin/bash

cd $(dirname $0)
PROJECT_DIR=$(realpath "$(pwd)/../..")
venv_dir="${PROJECT_DIR}/venv"

if [[ $(hostname) == raven* ]]; then
    # load python 3.10.9 via anaconda module (specific to Raven cluster)
    module load anaconda/3/2023.03
fi


# create the venv if it doesn't exist
if [ ! -d $venv_dir ]; then
    python -m venv $venv_dir
fi

# activate the venv
source $venv_dir/bin/activate

# install the requirements
pip install -U 'pip<25.3'
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt

# install the paper scripts package (editable)
pip install -e "$PROJECT_DIR"

if [[ $(hostname) == raven* ]]; then
    # install the development version of cubnm
    bash ./setup_cubnm.sh
fi

# register the venv with ipykernel
python -m ipykernel install --user --name cubnm_paper