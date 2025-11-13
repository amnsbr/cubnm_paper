import os
import glob
import sys
import argparse
import time
import copy
import json
import numpy as np
import cubnm
import cubnm_paper

def run(
        sub="group-train706", 
        ses="",
        parc="schaefer-100",
        grid_shape=22,
        cpu_gpu_identity=False, 
        rerun=False
    ):
    """
    Run grid optimization for given subject/session.

    Parameters
    ----------
    sub : str
        Subject identifier. Use "group-train706" or "group-test303" for group-level averages.
        Use individual subject IDs for individual-level data.
    ses : str, optional
        Session name.
    parc : str, optional
        Parcellation scheme. Default is "schaefer-100".
    grid_shape : int, optional
        Grid shape used in optimization.
    cpu_gpu_identity : bool, optional
        Whether to run the grid for CPU/GPU identity testing (uses shorter simulation duration).
    rerun : bool, optional
        Whether to rerun despite existing grid optimizations.
    """
    # load SC, FC and FCD
    sc, emp_fc_tril, emp_fcd_tril = cubnm_paper.data.load_input_data(sub=sub, ses=ses, parc=parc)
    # simulation options
    sim_options = copy.deepcopy(cubnm_paper.config.HCP_SIM_OPTIONS)
    if cpu_gpu_identity:
        # run shorter simulations for CPU/GPU identity grid
        sim_options.update(dict(
            duration=60,
            bold_remove_s=0,
            TR=1.0,
            states_ts=True,
            window_size=10,
            window_step=2,
        ))
    
    out_dir = os.path.join(
        cubnm_paper.config.DATA_DIR, 'hcp',
        'sim', 'grid', sub, ses, f'ctx_parc-{parc}'
    )

    # check if grid optimization already exists
    hardware = 'gpu' if (cubnm.utils.avail_gpus() > 0) else 'cpu'
    if not rerun and os.path.exists(out_dir):
        existing_run_dirs = glob.glob(os.path.join(out_dir, 'grid*'))
        for run_dir in existing_run_dirs:
            with open(os.path.join(run_dir, 'problem.json'), 'r') as f:
                problem_config = json.load(f)
            with open(os.path.join(run_dir, 'optimizer.json'), 'r') as f:
                optimizer_config = json.load(f)
            if (problem_config['duration'] == sim_options['duration'] and
                optimizer_config['grid_shape']['G'] == grid_shape and \
                problem_config['hardware_used'] == hardware):
                print(f"Grid optimization already exists in {run_dir}. Skipping...")
                return

    # define BNM problem
    problem = cubnm.optimize.BNMProblem(
        model = 'rWW',
        params = {
            'G': (0.001, 10.0), 
            'w_p': (0, 2.0),
            'J_N': (0.001, 0.5)
        },
        sc = sc,
        emp_fc_tril = emp_fc_tril,
        emp_fcd_tril = emp_fcd_tril,
        out_dir = out_dir,
        **sim_options
    )
    # run and save grid optimization
    grid = cubnm.optimize.GridOptimizer()
    start = time.time()
    grid.optimize(problem, grid_shape=grid_shape)
    print(f"Grid optimization time: {time.time()-start}s")
    start = time.time()
    grid.save(save_obj=cpu_gpu_identity)
    print(f"Grid saving time: {time.time()-start}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default="group-train706", help='Group/Subject ID')
    parser.add_argument('--ses', type=str, default='', help='Session')
    parser.add_argument('--parc', type=str, default='schaefer-100', help='Parcellation scheme')
    parser.add_argument('--grid_shape', type=int, default=22, help='Grid shape')
    parser.add_argument('--cpu_gpu_identity', action='store_true', 
        help='CPU/GPU identity grid (uses shorter simulation duration)'
    )
    parser.add_argument('--rerun', action='store_true', 
        help='Whether to rerun despite existing grid optimizations'
    )
    args = parser.parse_args()
    run(
        sub=args.sub, 
        ses=args.ses, 
        grid_shape=args.grid_shape, 
        cpu_gpu_identity=args.cpu_gpu_identity,
        rerun=args.rerun
    )