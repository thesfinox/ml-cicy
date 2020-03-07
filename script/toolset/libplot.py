# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# LIBPLOT:
#
#   Library of definitions and classes to plot, visualize and save results.
#
# AUTHOR: Riccardo Finotello
#

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt

assert np.__version__  >  '1.16' # to avoid issues with pytables

from os import path

# Set label sizes
mpl.rc('axes', labelsize=12)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Set working directories
ROOT_DIR = '.'      # root directory
IMG_DIR  = 'img'    # images directory
# Create directories
IMG_PATH = path.join(ROOT_DIR, IMG_DIR)
if path.isdir(IMG_PATH) is False:
    mkdir(IMG_PATH)

# Save the current figure
def save_fig(name, tight_layout=True, extension='png', resolution=600):

    filename = path.join(IMG_PATH, name + '.' + extension)
    if tight_layout:
        plt.tight_layout()

    print('    Saving {}...'.format(filename), flush=True)
    plt.savefig(filename, format=extension, dpi=resolution)
    print('    Saved {}!'.format(filename), flush=True)

# Get a generator to count the occurrencies
def get_counts(df, label, feature):

    for n in np.sort(df[feature].unique()):
        uniques, counts = np.unique(df[label].loc[df[feature] == n].values,
                                    return_counts=True
                                   )
        for u, c in np.c_[uniques, counts]:
            yield [ n, u, c ]

# Plot histogram of occurrencies
def count_plot(ax, data, title=None, xlabel=None, ylabel='N',
               legend=None, xlog=False, ylog=False, binstep=5,
               **kwargs
              ):

    min_tick = np.min(data) if np.min(data) > -100 else -100 # set a MIN cut
    max_tick = np.max(data) if np.max(data) < 100  else 100  # set a MAX cut

    ax.grid(alpha=0.2)                   # create a grid
    ax.set_title(title)                  # set title
    ax.set_xlabel(xlabel)                # set a label for the x axis
    ax.set_ylabel(ylabel)                # set a label for the y axis
    ax.set_xticks(np.arange(min_tick,    # set no. of ticks in the x axis
                            max_tick,
                            step=binstep
                           )
                 )

    if xlog:                             # use log scale in x axis if needed
        ax.set_xscale('log')
    if ylog:                             # use log scale in y axis if needed
        ax.set_yscale('log')

    ax.hist(data,                        # create histogram using 'step' funct.
            histtype='step',
            label=legend,
            **kwargs)

    if legend is not None:               # add legend
        ax.legend(loc='best')

    return ax

# Plot labeled features and their values
def label_plot(ax, data, title=None, xlabel=None, ylabel='values',
               legend=None, xlog=False, ylog=False, binstep=1,
               **kwargs
              ):

    labels      = [f[0] for f in data]   # labels vector
    importances = [f[1] for f in data]   # importances vector
    length      = len(labels)            # length of the labels vector
    
    ax.grid(alpha=0.2)                   # create a grid
    ax.set_title(title)                  # set title
    ax.set_xlabel(xlabel)                # set a label for the x axis
    ax.set_ylabel(ylabel)                # set a label for the x axis

    ax.set_xticks(np.arange(length,      # set no. of ticks in the x axis
                            step=binstep
                           )
                 )
    ax.set_xticklabels(labels,           # set name of labels of the x axis
                       ha='right',       # horizontal alignment
                       rotation=45       # rotation of the labels
                      )

    if xlog:                             # use log scale in x axis if needed
        ax.set_xscale('log')
    if ylog:                             # use log scale in y axis if needed
        ax.set_yscale('log')

    ax.plot(np.arange(length),           # plot data
            importances,
            label=legend,
            **kwargs)

    if legend is not None:               # add legend
        ax.legend(loc='best')

    return ax

# Plot the correlation matrix of a Pandas dataframe
def mat_plot(ax, df, label='correlation matrix', **kwargs):

    matrix = df.corr()                   # create correlation matrix
    labels = df.columns.tolist()         # extract the name of the labels

    ax.set_xticks(np.arange(len(labels), # set ticks for x axis
                  step=1)
                 )
    ax.set_xticklabels([''] + labels,    # set the name of the ticks
                       rotation=90
                      )

    ax.set_yticks(np.arange(len(labels), # set ticks for y axis
                  step=1)
                 )
    ax.set_yticklabels([''] + labels)    # set the name of the ticks
                    
    matshow = ax.matshow(matrix,         # show the matrix
                         vmin=-1.0,
                         vmax=1.0,
                         **kwargs
                        )
                                
    cbar = ax.figure.colorbar(matshow,   # create the colour bar
                              ax=ax,
                              fraction=0.05,
                              pad=0.05
                             )
    cbar.ax.set_ylabel(label,            # show the colour bar
                       va='bottom',      # vertical alignment
                       rotation=-90)     # rotation of the label

    return ax

# Plot a scatter plot with colours and sizes
def scatter_plot(ax, data, title=None, xlabel=None, ylabel=None,
                 legend=None, xlog=False, ylog=False,
                 colour=True, size=True, colour_label='N', size_leg=0,
                 **kwargs):

    ax.grid(alpha=0.2)                   # create  a grid
    ax.set_xlabel(xlabel)                # set labels for the x axis
    ax.set_ylabel(ylabel)                # set labels for the y axis
    ax.set_title(title)                  # set title

    if xlog:                             # use log scale in x axis if needed
        ax.set_xscale('log')
    if ylog:                             # use log scale in y axis if needed
        ax.set_yscale('log')

    if colour:                           # create the plot with size and colours
        if size:
            scat = ax.scatter(data[0], data[1], s=data[2], c=data[2], **kwargs)
        else:
            scat = ax.scatter(data[0], data[1], c=data[2], **kwargs)
        cbar = ax.figure.colorbar(scat, ax=ax)
        cbar.ax.set_ylabel(colour_label, rotation=-90, va='bottom')
    else:
        if size:
            scat = ax.scatter(data[0], data[1], s=data[2], **kwargs)
        else:
            scat = ax.scatter(data[0], data[1], **kwargs)

    scat.set_label(legend)               # set label of the plot
    if size_leg:                         # add the size legend if needed
        handles, labels = scat.legend_elements('sizes', num=size_leg)
        ax.legend(handles, labels, loc='lower center',
                  bbox_to_anchor=(0.5,-0.3), ncol=len(handles),
                  fontsize='medium', frameon=False)

    if legend:                           # show the legend
        ax.legend(loc='best')

    return ax

# Plot a series with trivial x label
def series_plot(ax, data, title=None, xlabel='series', ylabel=None,
                legend=None, xlog=False, ylog=False,
                step=False, std=False, **kwargs):

    ax.grid(alpha=0.2)                   # create the grid
    ax.set_title(title)                  # set the title
    ax.set_xlabel(xlabel)                # set labels for the x axis
    ax.set_ylabel(ylabel)                # set labels for the y axis

    if xlog:                             # use log scale in the x axis if needed
        ax.set_xscale('log')
    if ylog:                             # use log scale in the y axis if needed
        ax.set_yscale('log')

    series = np.arange(len(data))        # create trivial x axis data
    if step:                             # create the plot
        ax.step(series, data, label=legend, **kwargs)
    else:
        ax.plot(series, data, label=legend, **kwargs)

    if std:                              # show coloured strip with std
        ax.fill_between(series,
                        data + np.std(data),
                        data - np.std(data),
                        alpha=0.2)

    if legend is not None:               # show the legend
        ax.legend(loc='best')

    return ax
