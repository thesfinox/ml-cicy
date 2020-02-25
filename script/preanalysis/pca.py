# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# PCA:
#
#   Perform a PCA on the matrix components.
#
# AUTHOR: Riccardo Finotello
#

import numpy             as np
import sklearn           as sk

assert np.__version__ >  '1.16'   # to avoid issues with pytables
assert sk.__version__ >= '0.22.1' # for the recent implementation

from os                    import path
from sklearn.decomposition import PCA
from toolset.libutilities  import *
from toolset.libplot       import *

# Set working directories
ROOT_DIR = '.' # root directory

def pca(df_name, seed=42):

    # Read the database
    DB_PROD_NAME = df_name + '_features_analysis'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the database!')

    # Divide scalar, vector and tensor features
    scalar_features = list(df.\
            select_dtypes(include=['int8', 'int64', 'float64']).columns)
    vector_features = list(df.\
            drop(labels='matrix', axis=1).\
            select_dtypes(include=['object']).columns)
    tensor_features = list(df.\
            drop(labels=vector_features, axis=1).\
            select_dtypes(include=['object']).columns)

    # Set random seed
    RAND = seed
    np.random.seed(RAND)

    # Now extract the flattened matrix components:
    df['matrix_flat'] = ExtractTensor(flatten=True).fit_transform(df['matrix'])

    # Compute the PCA analysis with 2 components (plot-ready)
    print('\nComputing PCA with 2 components...')
    pca2             = PCA(n_components=2, random_state=RAND)
    matrix_pca_2     = pca2.fit_transform(np.stack(df['matrix_flat']))
    matrix_pca_2_var = pca2.explained_variance_ratio_
    print('    Variance for 2 components PCA: ',
          '{:.3f}% for the first component, '.format(matrix_pca_2_var[0]*100),
          '{:.3f}% for the second component'.format(matrix_pca_2_var[1]*100))

    # Compute the PCA analysis keeping 99% of variance
    print('\nComputing PCA with 99% variance...')
    df['matrix_pca99'] = list(PCA(n_components=0.99, random_state=RAND).\
                    fit_transform(np.stack(df['matrix_flat'])))
    print('    No. of components of the PCA with 99% variance preserved: {:d}'.\
                    format(df['matrix_pca99'].apply(np.shape).unique()[0][0]))

    df = df[scalar_features +
            vector_features +
            tensor_features +
            ['matrix_pca99']
           ] # simple reorder of the dataframe

    # Save dataframe to file
    print('\nSaving PCA to file...')
    DB_PROD_NAME = df_name + '_features_analysis'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    df.to_hdf(DB_PROD_PATH, key='df')
    print('PCA has been saved to file!')


    # Use the PCA with 2 components to plot the distribution of h_11 and h_21
    DB_PROD_NAME = df_name + '_labels_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df_labels = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the database!')

    fig, plot = plt.subplots(1, 2, figsize=(12,5))
    fig.tight_layout()

    scatter_plot(plot[0],
                 [ matrix_pca_2[:,0], matrix_pca_2[:,1], df_labels['h11'] ],
                 title='Distribution of $h_{11}$',
                 xlabel='PCA 1st component',
                 ylabel='PCA 2nd components',
                 size=False,
                 colour=True,
                 colour_label='Number of occurrencies',
                 alpha=0.5
                )
    scatter_plot(plot[1],
                 [ matrix_pca_2[:,0], matrix_pca_2[:,1], df_labels['h21'] ],
                 title='Distribution of $h_{21}$',
                 xlabel='PCA 1st component',
                 ylabel='PCA 2nd components',
                 size=False,
                 colour=True,
                 colour_label='Number of occurrencies',
                 alpha=0.5
                )

    save_fig('h11_h21_pca_distribution')
    # plt.show()
    plt.close(fig)

