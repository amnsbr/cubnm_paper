import os
import shutil
import subprocess
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats

import cubnm
import cubnm_paper

SOLAR_ROOT = os.path.join(
    cubnm_paper.config.DATA_DIR,
    "hcp", "solar",
)
PEDIGREE_DIR = os.path.join(
    cubnm_paper.config.DATA_DIR,
    "hcp", "pedi",
)

RUN_SOLAR_PATH = os.path.join(
    os.path.dirname(__file__),
    "run_solar.sh",
)

def run(
    sessions=['REST1_LR', 'REST2_LR'],
    inorm=True,
):
    """
    Calculate heritability (h2) of simulated and empirical features
    using SOLAR-Eclipse run on GPU. When multiple sessions are provided,
    the feature values are averaged across sessions before heritability
    estimation. Singleton subjects are removed based on family IDs.
    Output is saved in SOLAR_ROOT/<sessions_combined>/<feature>/output.csv directory.
    SOLAR-Eclipse dynamic version 9.0.0 is required. `solar` command should be
    available in the system PATH. 

    Parameters
    ----------
    sessions : list of str
        List of session names to include in the analysis.
    inorm : bool
        Whether to apply inverse normalization to the features.
    """
    sessions_str = "_".join(sessions)

    # load data
    data = {}
    opts = {}
    for ses in sessions:
        data[ses] = cubnm_paper.data.load_all_cmaes("twins", ses, "yeo", n_runs=2)
        # create a dataframe of optimal parameters and cost function + components
        opts[ses] = pd.DataFrame({sub: data[ses][sub]['opt'] for sub in data[ses].keys()}).T

    # filter to eligible subjects:
    # select subjects with data in all sessions
    if len(sessions) > 1:
        subs = sorted(list(
            set(data['REST1_LR'].keys()).intersection(set(data['REST2_LR'].keys())
        )))
    else:
        subs = sorted(list(data[sessions[0]].keys()))
    print("Subjects with data from selected sessions:", len(subs))
    # load demographic data
    unrestricted = pd.read_csv(os.path.join(cubnm_paper.config.DATA_DIR, 'hcp', 'pheno_unrestricted.csv'), index_col='Subject')
    restricted = pd.read_csv(os.path.join(cubnm_paper.config.DATA_DIR, 'hcp', 'pheno_restricted.csv'), index_col='Subject')
    demo = pd.DataFrame({
        'age': restricted.loc[:, 'Age_in_Yrs'],
        'sex': unrestricted.loc[:, 'Gender'],
        'Family_ID': restricted.loc[:, 'Family_ID'],
    })
    demo.index = demo.index.astype(str)
    demo = demo.loc[subs]
    # remove singletons
    family_counts = demo['Family_ID'].value_counts()
    singleton_families = family_counts.loc[family_counts == 1].index
    demo = demo.loc[~(demo['Family_ID'].isin(singleton_families))]
    demo = demo.drop(columns=['Family_ID'])
    subs = demo.index
    print("After removing singletons:", len(subs))

    # print demographics
    print("Demographics:")
    print("Age:")
    print(demo['age'].describe().round(1))
    print("Sex:")
    print((demo['sex'].value_counts(normalize=True)*100).round(2))

    # create an empty simgroup to use for converting tril to square matrices
    sg = cubnm.sim.rWWSimGroup(
        sc = np.zeros((100, 100)),
        duration=900,
        TR=0.72,
        exc_interhemispheric=True
    )
    sg.N = 1

    # calculate heritability for each parameter
    state_vars = ['I_E', 'r_E', 'S_E', 'I_I', 'r_I', 'S_I']
    params = ['G'] + [f'w_p{i}' for i in range(7)] + [f'J_N{i}' for i in range(7)]
    features = [
        'emp_fc_strength', 'sim_fc_strength', 'sc_strength',
        'emp_fc_edge', 'sim_fc_edge', 'sc_edge', 'params'
    ] + state_vars

    for feature in features:
        # prepare solar input
        solar_in = {}
        for ses in sessions:
            # determine columns and key in data
            if '_strength' in feature:
                k = feature.replace('_strength', '_tril')
                columns = [f'node_{i}' for i in range(100)]
            elif '_edge' in feature:
                k = feature.replace('_edge', '_tril')
                n_edges = data[ses][subs[0]][k].shape[0]
                columns = [f'edge_{i}' for i in range(n_edges)]
            elif feature in state_vars:
                columns = [f'node_{i}' for i in range(100)]
            elif feature == 'params':
                columns = params
            # load and curate input data
            solar_in[ses] = pd.DataFrame(
                index=subs, 
                columns=columns,
                dtype=float
            )
            for sub in tqdm(subs):
                # row-wise strengths
                if 'strength' in feature:
                    # convert tril to squared matrix
                    sg.sim_fc_trils = data[ses][sub][k][None, :]
                    mat = sg.get_sim_fc(0)
                    # set diagonal to nan (excluding self-connections)
                    np.fill_diagonal(mat, np.nan)
                    for i, col in enumerate(solar_in[ses].columns):
                        solar_in[ses].loc[sub, col] = np.nansum(mat[i, :])
                # edge strengths
                elif 'edge' in feature:
                    solar_in[ses].loc[sub, :] = data[ses][sub][k]
                # state variables
                elif feature in state_vars:
                    solar_in[ses].loc[sub, :] = data[ses][sub]['states'][feature]
                elif feature == 'params':
                    solar_in[ses].loc[sub, :] = opts[ses].loc[sub, params].values
            
        if len(sessions) > 1:
            # average across sessions
            solar_in = pd.DataFrame(
                        np.mean([solar_in[ses].values for ses in sessions], axis=0),
                        index=solar_in[sessions[0]].index,
                        columns=solar_in[sessions[0]].columns
                    )
        else:
            solar_in = solar_in[sessions[0]]
                
        # add age and sex and "id" (required by solar)
        solar_in = pd.concat([demo, solar_in], axis=1)
        solar_in.index = solar_in.index.rename('id')

        # inverse normalization (on columns excluding age and sex)
        if inorm:
            norm = scipy.stats.Normal()
            solar_in.iloc[:, 2:] = (solar_in.iloc[:, 2:].rank(axis=0) / (solar_in.shape[0]+1)).apply(norm.icdf)

        # save input csv
        solar_dir = os.path.join(SOLAR_ROOT, sessions_str, feature)
        os.makedirs(solar_dir, exist_ok=True)
        solar_in.to_csv(os.path.join(solar_dir, 'input.csv'))

        # create 'trait.header' file which includes
        # the list of columns line-separated
        np.savetxt(
            os.path.join(solar_dir, 'trait.header'),
            solar_in.columns[2:].tolist(),
            delimiter='\n',
            fmt="%s"
        )
        # copy pedigree data to analysis directory
        for file in os.listdir(PEDIGREE_DIR):
            if file.endswith('.csv'):
                continue
            shutil.copy(
                os.path.join(PEDIGREE_DIR, file),
                os.path.join(solar_dir, file)
            )

        # run solar
        result = subprocess.run(
            f"cd {solar_dir} && solar < {RUN_SOLAR_PATH}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running SOLAR for feature {feature} in {solar_dir}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        else:
            print(f"SOLAR completed for feature {feature} in {solar_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate heritability (h2) of simulated and empirical features using SOLAR-Eclipse."
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        default=['REST1_LR', 'REST2_LR'],
        help="List of session names to include in the analysis."
    )
    parser.add_argument(
        '--no_inorm',
        action='store_true',
        help="Do not apply inverse normalization to the features."
    )

    args = parser.parse_args()

    run(
        sessions=args.sessions,
        inorm=not args.no_inorm,
    )