# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# DATAVIEW:
#
#   Library of definitions and classes to visualize data inside a dataset.
#
# AUTHOR: Riccardo Finotello
#

import numpy  as np
import pandas as pd

assert np.__version__  >  '1.16' # to avoid issues with pytables

from toolset.libutilities import *
from toolset.libplot      import *

filter_dict = { 'h11': [1,16], 'h21': [1,86] }

def data_visualization(df):

    # Discard product spaces inside the dataframe
    df_noprod = df.loc[df['isprod'] == 0]
    # Remove outliers from the dataset
    df_noprod_noout = RemoveOutliers(filter_dict=filter_dict).\
                                     fit_transform(df_noprod)

    # Plot the occurrencies of h_11 and h_21
    print('\nCreating plots of the occurrencies of h_11 and h_21:')

    fig, plot = plt.subplots(1, 2, figsize=(12,5))
    fig.tight_layout()

    count_plot(plot[0],
               df_noprod_noout['h11'],
               title='Frequency of values of $h_{11}$',
               xlabel='$h_{11}$',
               legend='no products AND no outliers',
               ylog=True,
               binstep=2)
    count_plot(plot[0],
               df_noprod['h11'],
               title='Frequency of values of $h_{11}$',
               xlabel='$h_{11}$',
               legend='no products',
               ylog=True,
               binstep=2)
    count_plot(plot[0],
               df['h11'],
               title='Frequency of values of $h_{11}$',
               xlabel='$h_{11}$',
               legend='default',
               ylog=True,
               binstep=2)

    count_plot(plot[1],
               df_noprod_noout['h21'],
               title='Frequency of values of $h_{21}$',
               xlabel='$h_{21}$',
               legend='no products AND no outliers',
               ylog=True,
               binstep=10)
    count_plot(plot[1],
               df_noprod['h21'],
               title='Frequency of values of $h_{21}$',
               xlabel='$h_{21}$',
               legend='no products',
               ylog=True,
               binstep=10)
    count_plot(plot[1],
               df['h21'],
               title='Frequency of values of $h_{21}$',
               xlabel='$h_{21}$',
               legend='default',
               ylog=True,
               binstep=10)

    save_fig('h11_h21_occurrencies')
    # plt.show()
    plt.close(fig)

    # Show the occurrencies of h_11 and h_21 w.r.t. scalar features:
    scat_feat = [ 'num_cp', 'num_eqs', 'norm_matrix', 'rank_matrix' ]
    scat_labs = [ 'h11', 'h21', 'euler' ]

    fig, plot = plt.subplots(len(scat_feat),
                             len(scat_labs),
                             figsize=(6*len(scat_labs),5*len(scat_feat))
                            )
    fig.tight_layout()

    for n in range(len(scat_feat)):
        for m in range(len(scat_labs)):
            scatter_plot(plot[n,m],
                         list(np.asarray(\
                                 list(get_counts(df_noprod_noout,
                                                 scat_labs[m],
                                                 scat_feat[n]
                                                )
                                     )
                                 ).T
                             ),
                         title='Distribution of ${}$'.format(scat_labs[m]),
                         xlabel=scat_feat[n],
                         size=True,
                         colour=False,
                         legend='no products AND no outliers',
                         alpha=0.3
                        )
            scatter_plot(plot[n,m],
                         list(np.asarray(\
                                 list(get_counts(df_noprod,
                                                 scat_labs[m],
                                                 scat_feat[n]
                                                )
                                     )
                                 ).T
                             ),
                         title='Distribution of ${}$'.format(scat_labs[m]),
                         xlabel=scat_feat[n],
                         size=True,
                         colour=False,
                         legend='no products',
                         alpha=0.3
                        )
            scatter_plot(plot[n,m],
                         list(np.asarray(\
                                 list(get_counts(df,
                                                 scat_labs[m],
                                                 scat_feat[n]
                                                )
                                     )
                                 ).T
                             ),
                         title='Distribution of ${}$'.format(scat_labs[m]),
                         xlabel=scat_feat[n],
                         size=True,
                         colour=False,
                         legend='deafult',
                         alpha=0.3
                        )

    save_fig('h11_h21_distributions-comparison')
    # plt.show()
    plt.close(fig)

    fig, plot = plt.subplots(len(scat_feat),
                             len(scat_labs),
                             figsize=(6*len(scat_labs),5*len(scat_feat))
                            )
    fig.tight_layout()

    for n in range(len(scat_feat)):
        for m in range(len(scat_labs)):
            scatter_plot(plot[n,m],
                         list(np.asarray(\
                                 list(get_counts(df_noprod_noout,
                                                 scat_labs[m],
                                                 scat_feat[n]
                                                )
                                     )
                                 ).T
                             ),
                         title='Distribution of ${}$'.format(scat_labs[m]),
                         xlabel=scat_feat[n],
                         size=True,
                         colour=True,
                         colour_label='Number of occurrencies',
                         alpha=0.5
                        )

    save_fig('h11_h21_distributions')
    # plt.show()
    plt.close(fig)

    return df_noprod_noout
