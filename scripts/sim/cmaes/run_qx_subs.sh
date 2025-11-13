#!/bin/bash
# Runs individual-level CMAES on the next $n_x batch of $q subjects
# Usage: ./run_qx_subs.sh <q> <n_x> <args>

q=$1
n_x=$2
export run_cmaes_flags=$3

export PROJECT_DIR=$(realpath $(dirname $0)/../../..)

cd $PROJECT_DIR

export VENV_DIR="${PROJECT_DIR}/venv"


# activate the environment
module load anaconda/3/2023.03 cuda/11.8-nvhpcsdk
source ${VENV_DIR}/bin/activate

# get the list of next n_x * q subjects
n_subs=$(( $n_x * $q ))
all_subs=$(python ${PROJECT_DIR}/scripts/sim/cmaes/run.py print_subs --n_subs=$n_subs $run_cmaes_flags)
# count the actual number of subjects
n_subs=$(echo $all_subs | wc -w)

if [ $n_subs -eq 0 ]; then
    echo "No subjects to run."
    exit 0
fi

# recompute n_x based on actual n_subs
n_x=$(( (n_subs + q - 1) / q ))
echo "Running $n_subs subjects in $n_x jobs of $q subjects each."

# create a file including the arguments used in this
# call + the job IDs started
job_ids_filename="$PROJECT_DIR/logs/cmaes_job_ids_$(date +%Y%m%d_%H%M%S).txt"
echo "# Arguments:" $@ > "$job_ids_filename"

echo "Writing job IDs to $job_ids_filename"

# run the jobs (one job (node) per q subjects)
IFS=" " read -r -a all_subs <<< "$all_subs" # convert to array
for ((i=0;i<$n_x;i++)); do
    start_idx=$(( i * $q ))
    curr_subs="${all_subs[@]:start_idx:q}"
    curr_job_id=$(sbatch --parsable --export=ALL --ntasks=1 "${PROJECT_DIR}/scripts/sim/cmaes/run.sbatch" $curr_subs)
    echo "$curr_job_id is running $curr_subs" >> "$job_ids_filename"
done
