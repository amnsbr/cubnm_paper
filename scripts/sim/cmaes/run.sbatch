#!/bin/bash -l
# Usage: sbatch run.sbatch <subs_str> <args>
# Called via sbatch from run_qx_subs.sh
# Note: This is specific to the Raven cluster
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH -D ./
#SBATCH -J cubnm_cmaes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=24:00:00

subs=${@}

# activate the environment
module load anaconda/3/2023.03 cuda/11.8-nvhpcsdk
source ${VENV_DIR}/bin/activate

# generate comma-separated list of subs
subs_str=""
for sub in $subs; do
    subs_str="$subs_str,$sub"
done
subs_str=${subs_str:1} # remove the leading comma

echo "Running subs $subs_str"
srun \
    ${VENV_DIR}/bin/python \
    ${PROJECT_DIR}/scripts/sim/cmaes/run.py run --subs=$subs_str $run_cmaes_flags