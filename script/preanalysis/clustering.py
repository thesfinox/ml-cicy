# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# CLUSTERING:
#
#   Perform a clustering analysis on the matrix components.
#
# AUTHOR: Riccardo Finotello
#

import numpy             as np
import sklearn           as sk

assert np.__version__ >  '1.16'   # to avoid issues with pytables
assert sk.__version__ >= '0.22.1' # for the recent implementation

from os                   import path
from sklearn.cluster      import KMeans
from toolset.libutilities import *

# Set working directories
ROOT_DIR = '.' # root directory

def clustering(df_name, seed=42):

    # Read the database
    DB_PROD_NAME = df_name + '_features_production'
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


    cluster_range = range(2,20)                           # no. of clusters
    kmeans_labels = np.empty([df['matrix_flat'].shape[0], # empty vector
                                 len(cluster_range)
                             ],
                             dtype='int32'
                            )
    print('\nComputing clustering:')
    for n in cluster_range:
        print('    {:d} clusters...'.format(n))
        kmeans = KMeans(n_clusters=n, random_state=RAND, n_jobs=-1)
        kmeans.fit_transform(np.stack(df['matrix_flat']))
        kmeans_labels[:,n-min(cluster_range)] = kmeans.labels_

    df['clustering'] = list(kmeans_labels)

    df = df[scalar_features +
            ['clustering'] +
            vector_features +
            tensor_features] # simple reorder of the dataframe


    print('\nSaving dataset to file...')
    DB_PROD_NAME = df_name + '_features_analysis'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    df.to_hdf(DB_PROD_PATH, key='df')
    print('Clustering labels have been saved to file!')

    return cluster_range
