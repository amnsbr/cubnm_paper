import os
import glob
import copy
import pickle
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import cubnm.datasets
import cubnm_paper


def load_input_data(sub, ses="", parc="schaefer-100", paths_only=False):
    """
    Load structural connectome, empirical FC tril and FCD tril for given subject/session.

    Parameters
    ----------
    sub : str
        Subject identifier. Use "group-train706" or "group-test303" for group-level averages.
        Use individual subject IDs for individual-level data.
    ses : str, optional
        Session name.
    parc : str, optional
        Parcellation scheme. Default is "schaefer-100".
    paths_only : bool, optional
        If True, only return the file paths instead of loading the data.
        Only applicable for individual-level data.

    Returns
    -------
    sc : np.ndarray or str
        Structural connectome matrix or its file path if paths_only is True.
    emp_fc_tril : np.ndarray or str
        Empirical functional connectivity (FC) in tril format
        or its file path if paths_only is True.
    emp_fcd_tril : np.ndarray or str
        Empirical functional connectivity dynamics (FCD) in tril format
        or its file path if paths_only is True.
    """
    if sub in ["group-train706", "group-test303"]:
        # load group-level averages from toolbox
        # load structural connectome
        sc = cubnm.datasets.load_sc("strength", parc, "group-train706")
        # load empirical FC tril and FCD tril
        emp_fc_tril = cubnm.datasets.load_fc(
            parc, sub, exc_interhemispheric=True, return_tril=True
        )
        emp_fcd_tril = cubnm.datasets.load_fcd(
            parc, sub, exc_interhemispheric=True, return_tril=True
        )
    else:
        # load individual-level data from local files
        sc_path = os.path.join(
            cubnm_paper.config.DATA_DIR,
            "hcp",
            "SC",
            sub,
            f"ctx_parc-{parc}_mean001_thresh-1_desc-strength.txt",
        )
        fc_prefix = f"ctx_parc-{parc}_hemi-LR_exc-inter"
        fc_path = os.path.join(
            cubnm_paper.config.DATA_DIR,
            "hcp",
            "FC",
            sub,
            ses,
            f"{fc_prefix}_desc-FCtril.txt",
        )
        fcd_path = os.path.join(
            cubnm_paper.config.DATA_DIR,
            "hcp",
            "FC",
            sub,
            ses,
            f"{fc_prefix}_desc-FCDtril.txt",
        )
        if paths_only:
            return sc_path, fc_path, fcd_path
        sc = np.loadtxt(sc_path)
        emp_fc_tril = np.loadtxt(fc_path)
        emp_fcd_tril = np.loadtxt(fcd_path)
    return sc, emp_fc_tril, emp_fcd_tril


def load_grid(
    sub="group-train706",
    ses="",
    parc="schaefer-100",
    grid_shape=22,
    cpu_gpu_identity=False,
    hardware="gpu",
):
    """
    Load existing grid optimization results for given subject/session.
    The grid must have been run via cubnm_paper.sim.grid.run with the same parameters.

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
        Whether the grid was run for CPU/GPU identity testing.
    hardware : str, optional
        Hardware used in the grid optimization ('cpu' or 'gpu').
        Only used when cpu_gpu_identity is True.

    Returns
    -------
    grid : cubnm.optimize.GridOptimizer
        Grid optimizer object. For cpu_gpu_identity, loaded from
        a pickled object file. Otherwise, is reconstructed from saved data
        (with limited functionality).
    """
    # parent directory of all grid runs for this subject/session
    out_dir = os.path.join(
        cubnm_paper.config.DATA_DIR, "hcp", "sim", "grid", sub, ses, f"ctx_parc-{parc}"
    )

    # simulation options
    sim_options = copy.deepcopy(cubnm_paper.config.HCP_SIM_OPTIONS)
    if cpu_gpu_identity:
        # run shorter simulations for CPU/GPU identity grid
        sim_options.update(
            dict(
                duration=60,
                bold_remove_s=0,
                TR=1.0,
                states_ts=True,
                window_size=10,
                window_step=2,
            )
        )

    # find matching grid optimization run
    run_dir = None
    existing_run_dirs = glob.glob(os.path.join(out_dir, "grid*"))
    for curr_run_dir in existing_run_dirs:
        with open(os.path.join(curr_run_dir, "problem.json"), "r") as f:
            problem_config = json.load(f)
        with open(os.path.join(curr_run_dir, "optimizer.json"), "r") as f:
            optimizer_config = json.load(f)
        if (
            problem_config["duration"] == sim_options["duration"]
            and optimizer_config["grid_shape"]["G"] == grid_shape
        ):
            if cpu_gpu_identity:
                if problem_config["hardware_used"] == hardware:
                    run_dir = curr_run_dir
                    break
            else:
                run_dir = curr_run_dir
                break
    if run_dir is None:
        raise FileNotFoundError(
            f"No matching grid optimization found for sub={sub}, ses={ses}, "
            f"grid_shape={grid_shape}, cpu_gpu_identity={cpu_gpu_identity}"
        )

    # load/reconstruct grid optimizer
    if cpu_gpu_identity:
        # load pickled grid object
        with open(os.path.join(run_dir, "optimizer.pkl"), "rb") as f:
            grid = pickle.load(f)
    else:
        # reconstruct grid object from saved data
        sc, emp_fc_tril, emp_fcd_tril = load_input_data(sub=sub, ses=ses, parc=parc)
        # partially reconstruct problem and its associated sim group
        problem = cubnm.optimize.BNMProblem(
            model="rWW",
            params={"G": (0.001, 10.0), "w_p": (0, 2.0), "J_N": (0.001, 0.5)},
            sc=sc,
            emp_fc_tril=emp_fc_tril,
            emp_fcd_tril=emp_fcd_tril,
            **sim_options,
        )
        # sham get_state_averages function which loads from saved CSV
        state_averages_path = os.path.join(run_dir, "state_averages.csv")
        problem.sim_group.get_state_averages = lambda: pd.read_csv(
            state_averages_path, index_col=0
        )
        # partially reconstruct grid object
        grid = cubnm.optimize.GridOptimizer()
        grid.problem = problem
        grid.is_fit = True
        grid.history = pd.read_csv(os.path.join(run_dir, "history.csv"), index_col=0)
        grid.opt = pd.read_csv(os.path.join(run_dir, "opt.csv"), index_col=0).iloc[:, 0]
        # add run dir and optimal simulatino data to optimizer object
        # (these are not actual attributes of the normally created object)
        grid.run_dir = run_dir
        grid.opt_sim_data = np.load(os.path.join(run_dir, "opt_sim", "sim_data.npz"), allow_pickle=True)
    return grid


def load_cmaes(
    sub="group-train706",
    ses="",
    parc="schaefer-100",
    het_mode="homo",
    seed=1,
    popsize=128,
    n_iter=120,
    path_only=False,
):
    """
    Load existing grid optimization results for given subject/session.
    The grid must have been run via cubnm_paper.sim.grid.run with the same parameters.

    Parameters
    ----------
    sub : str
        Subject identifier. Use "group-train706" or "group-test303" for group-level averages.
        Use individual subject IDs for individual-level data.
    ses : str, optional
        Session name.
    parc : str, optional
        Parcellation scheme. Default is "schaefer-100".
    het_mode : str, optional
        Regional parameters heterogeneity mode.
        Either "homo", "2maps", or "yeo".
    seed : int, optional
        Random seed used in the CMA-ES optimization.
    popsize : int, optional
        Population size used in the CMA-ES optimization.
    n_iter : int, optional
        Number of iterations used in the CMA-ES optimization.
    path_only : bool, optional
        If True, only return the run path instead of loading the data.


    Returns
    -------
    cmaes : cubnm.optimize.CMAESOptimizer
        CMA-ES optimizer object. It is reconstructed from saved data
        (with limited functionality).
    or
    run_path : str
        Path to the CMA-ES run directory if path_only is True.
    """
    # parent directory of all grid runs for this subject/session
    out_dir = os.path.join(
        cubnm_paper.config.DATA_DIR,
        "hcp",
        "sim",
        het_mode,
        sub,
        ses,
        f"ctx_parc-{parc}",
    )

    # simulation options
    sim_options = copy.deepcopy(cubnm_paper.config.HCP_SIM_OPTIONS)

    # find matching optimization run
    run_dir = None
    existing_run_dirs = glob.glob(os.path.join(out_dir, "cmaes*"))
    for curr_run_dir in existing_run_dirs:
        with open(os.path.join(curr_run_dir, "optimizer.json"), "r") as f:
            optimizer_config = json.load(f)
        if (
            (optimizer_config.get("seed", None) == seed)
            and (optimizer_config.get("popsize", None) == popsize)
            and (optimizer_config.get("n_iter", None) == n_iter)
        ):
            run_dir = curr_run_dir
            break

    if path_only:
        return run_dir

    if run_dir is None:
        raise FileNotFoundError(
            f"No matching CMA-ES optimization found for sub={sub}, ses={ses}, seed={seed}"
        )

    # reconstruct optimizer
    # input data
    sc, emp_fc_tril, emp_fcd_tril = load_input_data(sub=sub, ses=ses, parc=parc)
    # heterogeneity mode config
    node_grouping = None
    maps = None
    maps_coef_range = None
    if het_mode == "homo":
        het_params = []
    else:
        het_params = ["w_p", "J_N"]
        if het_mode == "yeo":
            node_grouping = cubnm.datasets.load_maps(["yeo7"], parc)[0].astype(int)
        elif het_mode == "2maps":
            maps = cubnm.datasets.load_maps(
                ["myelinmap", "fcgradient01"], parc, "minmax"
            )
            maps_coef_range = (-5.0, 5.0)

    # partially reconstruct problem and its associated sim group
    problem = cubnm.optimize.BNMProblem(
        model="rWW",
        params={"G": (0.001, 10.0), "w_p": (0, 2.0), "J_N": (0.001, 0.5)},
        sc=sc,
        emp_fc_tril=emp_fc_tril,
        emp_fcd_tril=emp_fcd_tril,
        het_params=het_params,
        maps=maps,
        maps_coef_range=maps_coef_range,
        node_grouping=node_grouping,
        **sim_options,
    )
    # partially reconstruct optimizer object
    cmaes = cubnm.optimize.CMAESOptimizer(
        popsize=popsize, n_iter=n_iter, seed=seed, algorithm_kws=dict(tolfun=5e-3)
    )
    cmaes.setup_problem(problem)
    cmaes.is_fit = True
    cmaes.history = pd.read_csv(os.path.join(run_dir, "history.csv"), index_col=0)
    cmaes.opt = pd.read_csv(os.path.join(run_dir, "opt.csv"), index_col=0).iloc[:, 0]
    # add run dir and optimal simulatino data to optimizer object
    # (these are not actual attributes of the normally created object)
    cmaes.run_dir = run_dir
    cmaes.opt_sim_data = np.load(os.path.join(run_dir, "opt_sim", "sim_data.npz"), allow_pickle=True)
    return cmaes

def load_all_cmaes(
    subset,
    ses,
    het_mode,
    n_runs=2,
    seed=None
):
    """
    Loads all CMA-ES optimization results for given subset of subjects and session.
    Only loads the best run (lowest cost) for each subject.
    It stores the loaded data in a pickle file and reuses it if available.

    Parameters
    ----------
    subset : str
        Subset of subjects. Choices: "twins", "twins_unrelated", "twins_unrelated_96".
    ses : str
        Session name.
    het_mode : str
        Regional parameters heterogeneity mode. Choices: "homo", "2maps", or "yeo".
    n_runs : int, optional
        Number of CMA-ES runs per subject.
    seed : int or None, optional
        Specific seed to load. If None, loads all seeds from 1 to n_runs.
    
    Returns
    -------
    data : dict
        Dictionary of optimization and optimal simulation data for each subject.
    """
    pkl_path = os.path.join(
        cubnm_paper.config.DATA_DIR,
        "hcp",
        "sim",
        het_mode,
        f"all_cmaes_set-{subset}_ses-{ses}",
    )
    if seed is not None:
        pkl_path += f"_seed-{seed}"
    else:
        pkl_path += f"_nruns-{n_runs}"
    pkl_path += ".pkl"
    # load from pickle if available
    if os.path.exists(pkl_path):
        print(f"Loading all CMA-ES data from {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data
    
    # otherwise, load individually and save to pickle
    state_vars = ['I_E', 'r_E', 'S_E', 'I_I', 'r_I', 'S_I']
    regional_params = ['w_p', 'J_N']
    seeds = [seed] if seed is not None else range(1, n_runs + 1)

    all_subs = np.loadtxt(os.path.join(cubnm_paper.config.DATA_DIR, "hcp", "samples", f"{subset}.txt"), dtype=str)
    data = {}
    for sub in tqdm(all_subs):
        optimizer_runs = {}
        best_cost = np.inf
        optimizer = None # best run's object
        for _seed in seeds:
            try:
                optimizer_runs[_seed] = cubnm_paper.data.load_cmaes(
                    sub=sub,
                    ses=ses,
                    het_mode=het_mode,
                    seed=_seed,
                )
            except FileNotFoundError:
                print(f"CMA-ES results not found for sub={sub}, ses={ses}, seed={_seed}")
                continue
            if optimizer_runs[_seed].opt.cost < best_cost:
                optimizer = optimizer_runs[_seed]
                best_cost = optimizer.opt.cost
        if optimizer is not None:
            # select best run's useful data
            data[sub] = {
                "hist": optimizer.history,
                "opt": optimizer.opt,
                "emp_fc_tril": optimizer.problem.emp_fc_tril,
                "emp_fcd_tril": optimizer.problem.emp_fcd_tril,
                "sim_fc_tril": optimizer.opt_sim_data["sim_fc_trils"][0],
                "sim_fcd_tril": optimizer.opt_sim_data["sim_fcd_trils"][0],
            }
            data[sub]['states'] = {}
            for state in state_vars:
                data[sub]['states'][state] = optimizer.opt_sim_data[state][0]
            data[sub]['params'] = {}
            for param in regional_params:
                data[sub]['params'][param] = optimizer.opt_sim_data[param][0]
            # also add sc_tril (used in heritability and sc-fc calculations)
            # excluding interhemispheric connections
            sc = optimizer.problem.sim_group.sc.copy()
            sc[:50, 50:] = np.nan
            sc[50:, :50] = np.nan
            sc_tril = sc[np.tril_indices(100, -1)]
            data[sub]["sc_tril"] = sc_tril[~np.isnan(sc_tril)]
    
    # save to pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    return data