#!/bin/bash

cd $PROJECT_DIR

# scaling with N
for N in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768; do
    sbatch --export=ALL scripts/run_gpu.sbatch scripts/sim/scaling/run.py --machine raven --N $N --repeats 2
    if [ $N -le 256 ]; then
        sbatch --export=ALL scripts/run_cpu_single.sbatch scripts/sim/scaling/run.py --machine raven --N $N --repeats 2 --use_cpu --cores 1
        sbatch --export=ALL scripts/run_cpu.sbatch scripts/sim/scaling/run.py --machine raven --N $N --repeats 2 --use_cpu
    fi
done

# scaling with number of nodes
for nodes in 10 100 1000 10000; do
    sbatch --export=ALL scripts/run_gpu.sbatch scripts/sim/scaling/run.py --machine raven --N 1 --nodes $nodes --full_SC --node_scaling --repeats 2
    # on CPU run two repeats as separate jobs
    sbatch --export=ALL scripts/run_cpu_single.sbatch scripts/sim/scaling/run.py --machine raven --N 1 --nodes $nodes --full_SC --node_scaling --repeats 1 --use_cpu --cores 1
    sbatch --export=ALL scripts/run_cpu_single.sbatch scripts/sim/scaling/run.py --machine raven --N 1 --nodes $nodes --full_SC --node_scaling --repeats 1 --use_cpu --cores 1
done