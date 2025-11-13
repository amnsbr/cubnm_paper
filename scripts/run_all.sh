#!/bin/bash
# Runs SLURM jobs for all analyses done in the paper (except for those run in figure notebooks)

if [ -z $PROJECT_DIR ]; then
    echo "Please set the PROJECT_DIR environment variable."
    exit 1
fi

cd $PROJECT_DIR

# single CPU single simulation run for reference compute time
sbatch --export=ALL scripts/run_cpu_single.sbatch scripts/sim/run_N_sim.py --N 1

# Fig. 3: Group-level Homogeneous Grid
sbatch --export=ALL scripts/run_gpu.sbatch scripts/sim/grid/run.py --sub group-train706 --grid_shape 22

# Fig. 4: Group-level Homogeneous CMA-ES
sbatch --export=ALL scripts/run_gpu.sbatch scripts/sim/cmaes/run.py run --subs group-train706 --het_mode homo --seed 1

# Fig. 5: Group-level Heterogeneous CMA-ES
# map-based
sbatch --export=ALL scripts/run_gpu.sbatch scripts/sim/cmaes/run.py run --subs group-train706 --het_mode 2maps --seed 1
# node-based
sbatch --export=ALL scripts/run_gpu.sbatch scripts/sim/cmaes/run.py run --subs group-train706 --het_mode yeo --seed 1

# Fig. 6: Individualized CMA-ES
# running yeo for REST1_LR and REST2_LR of all twins (for heritability and reliability and GOF comparison)
bash scripts/sim/cmaes/run_qx_subs.sh 8 54 "--ses=REST1_LR --n_runs=2 --het_mode=yeo --subset=twins"
bash scripts/sim/cmaes/run_qx_subs.sh 8 54 "--ses=REST2_LR --n_runs=2 --het_mode=yeo --subset=twins"
# running homo and 2maps for REST1_LR of 96 unrelated subjects (for GOF comparison)
bash scripts/sim/cmaes/run_qx_subs.sh 8 12 "--ses=REST1_LR --n_runs=2 --het_mode=homo --subset=twins_unrelated_96"
bash scripts/sim/cmaes/run_qx_subs.sh 8 12 "--ses=REST1_LR --n_runs=2 --het_mode=2maps --subset=twins_unrelated_96"
# heritability calculation averaged across sessions and session-specific
sbatch --export=ALL scripts/run_gpu.sbatch scripts/heritability/calculate_h2.py --sessions REST1_LR REST2_LR
sbatch --export=ALL scripts/run_gpu.sbatch scripts/heritability/calculate_h2.py --sessions REST1_LR
sbatch --export=ALL scripts/run_gpu.sbatch scripts/heritability/calculate_h2.py --sessions REST2_LR

# Fig. 7: Scaling
# run scaling analyses on Raven (A100, single and multicore CPU)
bash scripts/sim/scaling/run_all_raven.sh
# run scaling analyses on PC (4080S)
# bash scripts/sim/scaling/run_all_pc.sh

# Fig. S7: Group-level Homogeneous Grid - CPU-GPU Identity
sbatch --export=ALL scripts/run_gpu.sbatch scripts/sim/grid/run.py --sub group-train706 --grid_shape 10 --cpu_gpu_identity
sbatch --export=ALL scripts/run_cpu.sbatch scripts/sim/grid/run.py --sub group-train706 --grid_shape 10 --cpu_gpu_identity

# Supplementary Text A.3
# get compute time for a 128-simulation generation on multi-cpu node
sbatch --export=ALL scripts/run_cpu.sbatch scripts/sim/run_N_sim.py --N 128