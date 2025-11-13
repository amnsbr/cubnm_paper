# cuBNM Paper: Reproducibility Code

This repository contains the code and analysis scripts for running the experiments in "cuBNM: GPU-Accelerated Brain Network Modeling" (Saberi et al.).

## Overview

This repository includes scripts for:
- Running grid search and CMA-ES optimization experiments
- Performing scaling analyses across CPU and GPU implementations
- Computing heritability and reliability estimates of simulated and empirical features
- Generating all figures and statistics reported in the paper

For the cuBNM toolbox itself, see the [main cuBNM repository](https://github.com/amnsbr/cubnm).

## Requirements

### System Requirements
- Linux x86_64 operating system
- NVIDIA GPU with CUDA support (tested on A100 and RTX 4080 Super)
- CUDA Toolkit v12.x
- SLURM (for running analyses as jobs on the cluster)

### Software Dependencies
- Python 3.10+ (tested with 3.10.9)
- Python packages listed in `./scripts/setup/requirements.in`
- SOLAR v9.0.0 dynamic (for heritability analysis)

### Installation

1. Install Python dependencies:
```bash
pip install -r scripts/setup/requirements.in
```

2. Install this repository as a package:
```bash
pip install -e .
```

3. Install SOLAR:
```bash
./scripts/setup/setup_solar.sh
```

## Data

### Included Data
Group-level modeling data is available through the `cubnm.datasets` module in the cuBNM package.

### Required External Data
Individualized modeling requires preprocessed Human Connectome Project (HCP) data. Due to data sharing restrictions, these files are not included. For preprocessing details, see the Methods section of the paper. The following directory structure is expected:
```
./data/hcp/
├── SC/                          # Structural connectivity matrices
├── FC/                          # Functional connectivity matrices
├── pedi/                        # Pedigree information (SOLAR format)
├── samples/                     # Subject ID lists (.txt files)
├── pheno_unrestricted.csv      # Unrestricted phenotype data
└── pheno_restricted.csv        # Restricted phenotype data
```

### Output Data
Running the scripts will generate:
- `./data/hcp/sim/`: Simulation and optimization results
- `./data/hcp/solar/`: Heritability analysis outputs
- `./data/scaling/`: Scaling analysis results (JSON files), which are combined into `./data/scaling.csv`

## Repository Structure
```
./scripts/
├── setup/                       # Environment setup scripts
├── cubnm_paper/                 # Paper package code
│   ├── data/                    # Data handling utilities
│   ├── config/                  # Configurations
│   └── utils/                   # Statistics and plotting functions
├── sim/                         # Simulation experiments
│   ├── grid/                    # Grid search
│   ├── cmaes/                   # CMA-ES optimization
│   ├── scaling/                 # Scaling analyses
│   └── run_N_sim.py             # CPU reference simulations
├── heritability/                # Heritability calculations
├── figures/                     # Figure generation and statistics
├── run_all.sh                   # Master script (SLURM cluster)
└── run_*.sbatch                 # Generic SLURM job scripts
```

## Usage

### Running All Experiments (Cluster)

Set the `PROJECT_DIR` environment variable and run:
```bash
export PROJECT_DIR=/path/to/cubnm-paper
./scripts/run_all.sh
```

This executes all simulations and analyses except:
- Figure generation notebooks (run manually from `./scripts/figures/`)
- Local PC scaling analysis (see below)

### Running Individual Components

**Figure 3 - Group-level homogeneous grid search:**
```bash
sbatch scripts/run_gpu.sbatch scripts/sim/grid/run.py --sub group-train706 --grid_shape 22
```

**Figure 4 - Group-level homogeneous CMA-ES:**
```bash
sbatch scripts/run_gpu.sbatch scripts/sim/cmaes/run.py run --subs group-train706 --het_mode homo --seed 1
```

**Figure 5 - Group-level heterogeneous CMA-ES:**
```bash
# Map-based heterogeneity
sbatch scripts/run_gpu.sbatch scripts/sim/cmaes/run.py run --subs group-train706 --het_mode 2maps --seed 1

# Node-based heterogeneity
sbatch scripts/run_gpu.sbatch scripts/sim/cmaes/run.py run --subs group-train706 --het_mode yeo --seed 1
```

**Figure 6 - Individualized CMA-ES:**
```bash
# Yeo model for all twins across both sessions
bash scripts/sim/cmaes/run_qx_subs.sh 8 54 "--ses=REST1_LR --n_runs=2 --het_mode=yeo --subset=twins"
bash scripts/sim/cmaes/run_qx_subs.sh 8 54 "--ses=REST2_LR --n_runs=2 --het_mode=yeo --subset=twins"

# Homogeneous and map-based heterogeneous models for GOF comparison on the 96 unrelated subjects (day 1)
bash scripts/sim/cmaes/run_qx_subs.sh 8 12 "--ses=REST1_LR --n_runs=2 --het_mode=homo --subset=twins_unrelated_96"
bash scripts/sim/cmaes/run_qx_subs.sh 8 12 "--ses=REST1_LR --n_runs=2 --het_mode=2maps --subset=twins_unrelated_96"

# Calculate heritability (requires completed CMA-ES runs)
sbatch scripts/run_gpu.sbatch scripts/heritability/calculate_h2.py --sessions REST1_LR REST2_LR
```

**Figure 7 - Scaling analysis:**
```bash
# On cluster (A100, single and multicore CPU)
bash scripts/sim/scaling/run_all_raven.sh

# On local PC (RTX 4080 Super)
bash scripts/sim/scaling/run_all_pc.sh
```

**Figure S7 - CPU-GPU identity verification:**
```bash
sbatch scripts/run_gpu.sbatch scripts/sim/grid/run.py --sub group-train706 --grid_shape 10 --cpu_gpu_identity
sbatch scripts/run_cpu.sbatch scripts/sim/grid/run.py --sub group-train706 --grid_shape 10 --cpu_gpu_identity
```

**Reference compute time benchmarks:**
```bash
# Single simulation (Main text)
sbatch scripts/run_cpu_single.sbatch scripts/sim/run_N_sim.py --N 1

# 128-simulation batch (Supplementary Text A.3)
sbatch scripts/run_cpu.sbatch scripts/sim/run_N_sim.py --N 128
```

## Support
Feel free to contact Amin Saberi (amnsbr[at]gmail.com) if you have any questions. For questions and issues related to the cuBNM toolbox itself, please refer to the [cuBNM GitHub repository](https://github.com/amnsbr/cubnm/issues).