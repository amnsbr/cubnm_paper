import os
import datetime
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netneurotools.datasets
import netneurotools.interface
import netneurotools.plotting
import neuromaps.stats
import neuromaps.images
import neuromaps.nulls

def plot_parc(
    parc_data, 
    parc='schaefer-100',
    cmap='viridis', 
    **kwargs
):
    """
    Plot parcellated data on fslr surface

    Parameters
    ----------
    parc_data : array-like, shape (n_parcels,)
    parc : str
        Parcellation scheme.
    cmap : str or matplotlib Colormap
    **kwargs
        Additional keyword arguments passed to `netneurotools.plotting.pv_plot_surface`.

    Returns
    -------
    plotter : pv.Plotter
        The plotter object.
    """
    if parc == 'schaefer-100':
        parc_file = netneurotools.datasets.fetch_schaefer2018('fslr32k')['100Parcels7Networks']
    else:
        raise ValueError(f"Parcellation '{parc}' not recognized.")
    # map parcel data to vertices
    surf_data, _, _ = netneurotools.interface.parcels_to_vertices(
        parc_data=parc_data.reshape(2, -1),
        parc_file=parc_file,
    )
    # plot on surface
    return netneurotools.plotting.pv_plot_surface(
        surf_data, 
        'fslr32k', 
        layout='row', 
        jupyter_backend='static',
        zoom_ratio=1.25,
        cmap=cmap,
        lighting_style='default',
        plotter_kws=dict(
            window_size=(2400, 600),
        ),
        show_colorbar=False,
        **kwargs
    )

def make_colorbar(
    vmin, 
    vmax, 
    cmap=None, 
    bins=None, 
    orientation='vertical', 
    figsize=None, 
    label=None
):
    """
    Plot a colorbar

    Parameters
    ---------
    vmin, vmax: (float)
    cmap: (str or `matplotlib.colors.Colormap`)
    bins: (int)
        if specified will plot a categorical cmap
    orientation: (str)
        - 'vertical'
        - 'horizontal'
    figsize: (tuple)

    Returns
    -------
    fig: (`matplotlib.figure.Figure`)
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.linspace(vmin, vmax, 100).reshape(10, 10), cmap=plt.cm.get_cmap(cmap, bins))
    fig.gca().set_visible(False)
    divider = make_axes_locatable(ax)
    if orientation == 'horizontal':
        cax = divider.append_axes("bottom", size="10%", pad=0.05)
    else:
        cax = divider.append_axes("left", size="10%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=np.array([vmin, vmax]), orientation=orientation)
    if label is not None:
        if orientation == 'horizontal':
            rotation = 0
            labelpad = -10
        else:
            rotation = 270
            labelpad = 10
        cbar.set_label(label, rotation=rotation, labelpad=labelpad)
    cax.yaxis.tick_left()
    cax.xaxis.tick_bottom()
    return fig

def icc(x, y):
    """
    Calculate intraclass correlation between two measurements
    using ICC(2,k) model.
    
    Parameters
    ----------
    x, y : array-like, shape (n_samples,)

    Returns
    -------
    icc_value : float
        Intraclass correlation coefficient.
    """
    icc_df = pd.DataFrame(index=range(x.shape[0]), columns=['x', 'y'], dtype=float)
    icc_df.loc[:, 'x'] = x.values
    icc_df.loc[:, 'y'] = y.values
    icc_df_long = icc_df.unstack().reset_index()
    icc_res = pg.intraclass_corr(
        data=icc_df_long, 
        targets='level_1', 
        raters='level_0',
        ratings=0
    ).set_index("Type")
    return icc_res.loc['ICC2k', 'ICC']

def spin_test(x, y, n_perm=1000):
    """
    Spin test (Pearson correlation) between two cortical maps parcellated
    based on Schaefer-100 parcellation.

    Parameters
    ----------
    x, y : array-like, shape (n_parcels,)
        Parcellated cortical maps to compare.
    n_perm : int
        Number of permutations for spin test.
    
    Returns
    -------
    r : float
        Pearson correlation coefficient between x and y.
    p_value : float
        Two-tailed p-value from spin test.
    """
    # load parcellation
    parc_annot = netneurotools.datasets.fetch_schaefer2018('fsaverage')['100Parcels7Networks']
    parcellation = neuromaps.images.annot_to_gifti(parc_annot)
    # rotate parcel indices
    indices = np.arange(100)
    rotated_indices = neuromaps.nulls.alexander_bloch(
        indices, 
        atlas='fsaverage', 
        density='164k', 
        n_perm=n_perm, 
        seed=1234, 
        parcellation=parcellation
    )
    rotated = x[rotated_indices]
    # compute correlation and p-value
    r, p = neuromaps.stats.compare_images(x, y, nulls=rotated, ignore_zero=False)
    return r, p

def convert_seconds(seconds):
    """
    Convert seconds to a human-readable string format,
    used for plotting scaling results.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    time_str : str
        Human-readable time string
    """
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} m"
    elif seconds < 86400:
        hours = seconds / (60 * 60)
        return f"{hours:.1f} h"
    else:
        days = seconds / (60 * 60 * 24)
        return f"{days:.1f} d"
