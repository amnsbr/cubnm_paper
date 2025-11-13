import time
import argparse
import copy
import numpy as np
import cubnm
import cubnm_paper

def run(N, sub="group-train706", ses=""):
    """
    Run N example simulations (identical parameters). This is used for predicting 
    single-/multithreaded CPU compute time for HCP simulations in comparison to GPU.

    Parameters
    ----------
    N : int
        Number of identical simulations
    sub : str
        Subject identifier. Use "group-train706" or "group-test303" for group-level averages.
        Use individual subject IDs for individual-level data.
    ses : str, optional
        Session name.
    """
    # load SC, FC and FCD
    sc, emp_fc_tril, emp_fcd_tril = cubnm_paper.data.load_input_data(sub=sub, ses=ses)
    # simulation options
    sim_options = copy.deepcopy(cubnm_paper.config.HCP_SIM_OPTIONS)

    sg = cubnm.sim.rWWSimGroup(sc=sc, force_cpu=True, **sim_options)
    sg.N = N
    sg.param_lists['G'] = np.full(N, 0.5)
    sg._set_default_params(missing=True)
    start = time.time()
    sg.run()
    print(f"init+run time: {time.time()-start}s")
    start = time.time()
    sg.score(emp_fc_tril=emp_fc_tril, emp_fcd_tril=emp_fcd_tril)
    print(f"scoring time: {time.time()-start}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True, help='Number of identical simulations to run')
    parser.add_argument('--sub', type=str, default="group-train706", help='Group/Subject ID')
    parser.add_argument('--ses', type=str, default='', help='Session')
    args = parser.parse_args()
    run(args.N, sub=args.sub, ses=args.ses)