import os
import numpy as np
import argparse
import uuid
import json
import multiprocessing

import cubnm
import cubnm_paper

def parse_N(N_str):
    if "," in N_str:
        return [int(N) for N in N_str.split(",")]
    else:
        return [int(N_str)]
    
def parse_nodes(nodes_str):
    if "," in nodes_str:
        return [int(nodes) for nodes in nodes_str.split(",")]
    elif ":" in nodes_str:
        start, end = nodes_str.split(":")
        return list(range(int(start), int(end)+1))
    else:
        return [int(nodes_str)]
    

def run_and_time(
        machine, N=1, repeat_idx=0, 
        duration=60, nodes=100, node_scaling=False,
        full_SC=False, use_cpu=False, cores=1, 
        save=False, verbose=False
    ):
    """
    Runs N identical simulations with specified options and 
    store the configuration and compute time in a JSON file.

    Parameters
    ----------
    machine: {'raven', 'pc'}
        The machine on which the simulation is run.
    N: int
        The number of identical simulations to run.
    repeat_idx: int
        The index of the repeat for this simulation run.
    duration: int
        The duration of the simulations in seconds.
    nodes: int
        The number of nodes in the structural connectivity matrix.
    node_scaling: bool
        Whether this is for a node-scaling experiment, in which
        case dt will be set to 1.0 and FCD will not be calculated.
        This is to ensure that the cooperation mode (> 500 nodes)
        can be run.
    full_SC: bool
        Whether to use a fully connected structural connectivity matrix
    use_cpu: bool
        Whether to force the simulations to run on CPU.
    cores: int
        The number of CPU cores available.
        This is just to write in the JSON file and has no effects.
    save: bool
        Whether to save the simulation data.
    verbose: bool
        Whether to print verbose output during the simulation.
    
    Returns
    -------
    sim_group: sim.SimGroup
        The simulation group object containing the results.
    """
    # output directory for jsons
    OUTPUT_DIR = os.path.join(cubnm_paper.config.DATA_DIR, 'scaling', machine)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # json prefix / output directory name for simulation (when saved)
    out_dirname = f'N-{N}'
    if duration != 60:
        out_dirname += f'_duration-{duration}'
    if nodes != 100:
        out_dirname += f'_nodes-{nodes}'
    if node_scaling:
        out_dirname += '_node_scaling'
    if full_SC:
        out_dirname += '_fullSC'
    # load SC
    if full_SC:
        sc = (np.ones((nodes, nodes)) * 0.01) / (nodes/100) # setting mean to 0.01 and adjusting for number of nodes
        sc[np.diag_indices(nodes)] = 0
        do_fic = False
    else:
        sc = cubnm.datasets.load_sc('strength', f'schaefer-{nodes}')
        do_fic = True
    # variable simulation options
    if node_scaling or (nodes > 500):
        dt = '1.0'
        do_fc = True
        do_fcd = False
        gof_terms = ['+fc_corr']
    else:
        dt = '0.1'
        do_fc = True
        do_fcd = True
        gof_terms = ['+fc_corr', '+fcd_corr']
    # initialize simulation group
    sim_group = cubnm.sim.rWWSimGroup(
        duration=duration,
        TR=1,
        sc=sc,
        sc_dist=None,
        window_size=10,
        window_step=2,
        states_ts=False,
        sim_verbose=verbose,
        force_cpu=use_cpu,
        force_gpu=(not use_cpu),
        dt=dt,
        gof_terms=gof_terms,
        do_fc=do_fc,
        do_fcd=do_fcd,
        do_fic=do_fic,
        max_fic_trials=0,
        out_dir=os.path.join(OUTPUT_DIR, out_dirname),
    )
    sim_group.N = N
    sim_group.param_lists['G'] = np.full(N, 0.5)
    sim_group._set_default_params()
    if full_SC:
        # when using fully connected SC,
        # instead of doing FIC (undefined behavior)
        # set wIE to a fixed value
        sim_group.param_lists['wIE'] = np.full((N, nodes), 5.0)
    # run the simulations
    sim_group.run(force_reinit=True)
    # check if first and last simulations are equal
    first_last_equal = np.allclose(sim_group.sim_bold[0], sim_group.sim_bold[-1], atol=1e-6)
    if not first_last_equal:
        print("WARNING: First and last simulations are not equal.")
    # save the config and timing as a json file
    # using a UUID. These jsons will ultimately 
    # be combined into a csv. This is a safer solution
    # for parallel jobs compared to attempting to write
    # to a csv directly
    json_path = os.path.join(OUTPUT_DIR, out_dirname+'_'+str(uuid.uuid4())+".json")
    json_data = {
        'machine': machine,
        'N': N,
        'duration': duration,
        'nodes': nodes,
        'node_scaling': node_scaling,
        'co_launch': sim_group._co_launch,
        'full_SC': full_SC,
        'dt': float(sim_group.dt),
        'repeat': repeat_idx,
        'version': cubnm.__version__,
        'use_cpu': use_cpu,
        'cpu_cores': cores,
        'init_time': sim_group.init_time,
        'run_time': sim_group.run_time,
        'first_last_equal': first_last_equal,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f)
    # save simulation data if requested
    if save:
        sim_group.save()
    return sim_group

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, required=True,
                        choices=['raven', 'pc'],)
    parser.add_argument("--N", type=parse_N, default=[1])
    parser.add_argument("--duration", type=parse_N, default=[60])
    parser.add_argument("--nodes", type=parse_nodes, default=[100])
    parser.add_argument('--node_scaling', action='store_true')
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument('--full_SC', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--cores', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--save', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    
    for N in args.N:
        for duration in args.duration:
            for nodes in args.nodes:
                for repeat_idx in range(args.repeats):
                    run_and_time(
                        machine = args.machine,
                        N = N,
                        repeat_idx = repeat_idx,
                        duration = duration,
                        nodes = nodes,
                        node_scaling = args.node_scaling,
                        full_SC = args.full_SC,
                        use_cpu = args.use_cpu,
                        cores = args.cores,
                        save = args.save,
                        verbose = args.verbose
                    )