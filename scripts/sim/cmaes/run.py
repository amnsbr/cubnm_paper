import os
import argparse
import copy
import json
import time
import numpy as np

import cubnm
import cubnm_paper


def print_subs(
    ses,
    subset="twins",
    n_subs=0,
    parc="schaefer-100",
    n_runs=2,
    seed=None,
    popsize=128,
    n_iter=120,
    het_mode="yeo",
):
    """
    Print subject IDs from a given subset that do not yet have CMA-ES results
    but have the required input data.

    Parameters
    ----------
    ses : str
        Session ID.
    subset : str
        Subset of subjects to include. Choices: "twins", "twins_unrelated", "twins_unrelated_96".
    n_subs : int
        Max number of subjects to print.
    parc : str
        Parcellation scheme.
    n_runs : int
        Number of runs per subject.
    seed : int or None
        Specific seed to check. If None, check all seeds from 1 to n_runs.
    popsize : int
        CMA-ES population size.
    n_iter : int
        CMA-ES number of iterations.
    het_mode : str
        Heterogeneity mode. Choices: "homo", "yeo", "2maps".
    """
    try:
        all_subs = np.loadtxt(
            os.path.join(
                cubnm_paper.config.DATA_DIR, "hcp", "samples", subset + ".txt"
            ),
            dtype=str,
        )
    except FileNotFoundError:
        raise ValueError(f"Subset {subset} not found.")

    selected_subs = []
    for sub in all_subs:
        # check if input data exists
        sc_path, fc_path, fcd_path = cubnm_paper.data.load_input_data(
            sub=sub, ses=ses, parc=parc, paths_only=True
        )
        if not (
            os.path.exists(fc_path)
            and os.path.exists(fcd_path)
            and os.path.exists(sc_path)
        ):
            continue
        # check if all runs exist
        if seed is not None:
            seeds = [seed]
        else:
            seeds = range(1, n_runs + 1)
        seeds_exist = {}
        for _seed in seeds:
            run_dir = cubnm_paper.data.load_cmaes(
                sub=sub,
                ses=ses,
                parc=parc,
                het_mode=het_mode,
                seed=_seed,
                popsize=popsize,
                n_iter=n_iter,
                path_only=True,
            )
            seeds_exist[_seed] = run_dir is not None
        out_exist = all(list(seeds_exist.values()))
        if not out_exist:
            selected_subs.append(sub)
        if len(selected_subs) >= n_subs:
            break
    print(*selected_subs)


def run(
    subs,
    ses,
    parc="schaefer-100",
    popsize=128,
    n_iter=120,
    n_runs=2,
    seed=None,
    het_mode="yeo",
):
    """
    Run CMA-ES optimization for given subject(s), session and runs.

    Parameters
    ----------
    subs : list of str
        List of subject IDs.
    ses : str
        Session ID.
    parc : str
        Parcellation scheme.
    popsize : int
        CMA-ES population size.
    n_iter : int
        CMA-ES number of iterations.
    n_runs : int
        Number of runs per subject.
    seed : int or None
        Specific seed to run. If None, run all seeds from 1 to n_runs.
    het_mode : str
        Heterogeneity mode. Choices: "homo", "yeo", "2maps".
    """
    problems = []
    optimizers = []
    for sub in subs:
        # output directory
        out_dir = os.path.join(
            cubnm_paper.config.DATA_DIR,
            "hcp",
            "sim",
            het_mode,
            sub,
            ses,
            f"ctx_parc-{parc}",
        )
        # input data
        sc, emp_fc_tril, emp_fcd_tril = cubnm_paper.data.load_input_data(
            sub=sub, ses=ses, parc=parc
        )
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
        # other simulation options
        sim_options = copy.deepcopy(cubnm_paper.config.HCP_SIM_OPTIONS)
        sim_options.update(
            {
                "sim_verbose": False,
            }
        )
        # separate problems/optimizers for each seed
        if seed is not None:
            seeds = [seed]
        else:
            seeds = range(1, n_runs + 1)
        for _seed in seeds:
            # skip the CMAES run if it's already done
            run_dir = cubnm_paper.data.load_cmaes(
                sub=sub,
                ses=ses,
                parc=parc,
                het_mode=het_mode,
                seed=_seed,
                popsize=popsize,
                n_iter=n_iter,
                path_only=True,
            )
            if run_dir is not None:
                print(f"{sub} seed {_seed} already exists at {run_dir}, skipping.")
                continue
            print(f"Running {sub} seed {_seed}")
            # define the BNM problem
            # (this is the same for all optimization seeds,
            # but should be redefined to have separate problem and sim group
            # objects per optimization run)
            problem = cubnm.optimize.BNMProblem(
                model="rWW",
                params={
                    "G": (0.001, 10.0),
                    "w_p": (0, 2.0),
                    "J_N": (0.001, 0.5),
                },
                sc=sc,
                emp_fc_tril=emp_fc_tril,
                emp_fcd_tril=emp_fcd_tril,
                het_params=het_params,
                maps=maps,
                maps_coef_range=maps_coef_range,
                node_grouping=node_grouping,
                out_dir=out_dir,
                **sim_options,
            )
            problems.append(problem)
            # define the optimizer
            optimizer = cubnm.optimize.CMAESOptimizer(
                popsize=popsize,
                n_iter=n_iter,
                seed=_seed,
                algorithm_kws=dict(tolfun=5e-3),
            )
            optimizers.append(optimizer)
    start = time.time()
    if len(problems) > 1:
        # batch optimization
        cubnm.optimize.batch_optimize(optimizers, problems)
    else:
        # single optimization (single subject/seed)
        optimizer = optimizers[0]
        problem = problems[0]
        optimizer.setup_problem(problem)
        optimizer.optimize()
        optimizer.save()

    print(
        f"CMAES for {len(problems)} runs with {popsize} "
        f"particles and {n_iter} iterations took a total "
        f"walltime of {time.time() - start}s"
    )


def _add_shared_arguments(parser):
    parser.add_argument("--ses", type=str, default="", help="Session")
    parser.add_argument(
        "--parc", type=str, default="schaefer-100", help="Parcellation scheme"
    )
    parser.add_argument(
        "--het_mode",
        type=str,
        default="yeo",
        choices=["homo", "yeo", "2maps"],
        help="Regional parameters heterogeneity mode",
    )
    parser.add_argument("--popsize", type=int, default=128, help="Population size")
    parser.add_argument("--n_iter", type=int, default=120, help="Number of iterations")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=2,
        help="Number of runs. Ignored when seed is provided.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optimization seed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="cmd")

    # run command
    run_parser = subparsers.add_parser("run", help="Run CMA-ES optimization")
    run_parser.add_argument(
        "--subs",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of subject IDs",
    )
    _add_shared_arguments(run_parser)

    # print_subs command
    print_subs_parser = subparsers.add_parser(
        "print_subs", help="Run CMA-ES optimization"
    )
    print_subs_parser.add_argument(
        "--subset",
        type=str,
        default="twins",
        choices=["twins", "twins_unrelated", "twins_unrelated_96"],
        help="Subset of subjects",
    )
    print_subs_parser.add_argument(
        "--n_subs", type=int, default=10000, help="Number of subjects to print"
    )
    _add_shared_arguments(print_subs_parser)

    # discard extra args (run args may be passed to print_subs)
    args, _ = parser.parse_known_args()

    if args.cmd == "run":
        run(
            subs=args.subs,
            ses=args.ses,
            parc=args.parc,
            het_mode=args.het_mode,
            popsize=args.popsize,
            n_iter=args.n_iter,
            n_runs=args.n_runs,
            seed=args.seed,
        )
    elif args.cmd == "print_subs":
        print_subs(
            ses=args.ses,
            subset=args.subset,
            n_subs=args.n_subs,
            seed=args.seed,
            parc=args.parc,
            het_mode=args.het_mode,
            n_runs=args.n_runs,
            popsize=args.popsize,
            n_iter=args.n_iter,
        )
