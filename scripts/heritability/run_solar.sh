#!/bin/bash
### This a script to run solar on gpu cluster, make sure you have solar installed and CUDA available
### Usage: solar < run_solar.sh
### Must be run in the directory where input.csv and trait.header are located

# load input file
load pheno input.csv

# creating coviates
covar age^1,2#sex

# run heritability analysis using GPU 
gpu_fphi -list trait.header -o output.csv -all_gpus -use_covs