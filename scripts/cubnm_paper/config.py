import os

# directories
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# common simulation options used in simulations with HCP data
HCP_SIM_OPTIONS = dict(
    duration=900,
    bold_remove_s=30,
    TR=0.72,
    sc_dist=None,
    dt='0.1',
    bw_dt='1.0',
    states_ts=False,
    states_sampling=None,
    noise_out=False,
    sim_seed=0,
    noise_segment_length=30,
    gof_terms=['+fc_corr', '-fcd_ks'],
    do_fc=True,
    do_fcd=True,
    window_size=30,
    window_step=5,
    fcd_drop_edges=True,
    exc_interhemispheric=True,
    bw_params='heinzle2016-3T',
    sim_verbose=True,
    do_fic=True,
    max_fic_trials=0,
    fic_penalty_scale=0.5,
)