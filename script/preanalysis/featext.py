# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# FEATEXT:
#
#   Extract usable features from the dataset, based on previous analysis.
#
# AUTHOR: Riccardo Finotello
#

import numpy as np

assert np.__version__ >  '1.16'   # to avoid issues with pytables

from os                   import path
from joblib               import dump
from toolset.libutilities import *
from toolset.libplot      import *

# Set working directories
ROOT_DIR = '.' # root directory

def feature_extract(df_name):

    DB_PROD_NAME = df_name + '_features_analysis'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the features database!')
        
    DB_PROD_NAME = df_name + '_labels_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df_labels = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the labels database!')


    # For simplicity (and since disk space is not an issue, for now) we create
    # separate files for the dataset to be considered:
    print('\nExtracting features based on previous analysis...')
    df_matrix  = np.stack(ExtractTensor(flatten=True).\
            fit_transform(df['matrix']))
    df_num_cp  = np.stack(ExtractTensor(flatten=True).\
            fit_transform(df['num_cp']))
    df_dim_cp  = np.stack(ExtractTensor(flatten=True).\
            fit_transform(df['dim_cp']))
    df_eng_h11 = np.c_[df_num_cp,
                       df_dim_cp,
                       np.stack(ExtractTensor(flatten=True).\
                               fit_transform(df['matrix_pca99']))
                      ]
    df_eng_h21 = np.c_[df_num_cp,
                       df_dim_cp,
                       np.stack(ExtractTensor(flatten=True).\
                               fit_transform(df['dim_h0_amb'])),
                       np.stack(ExtractTensor(flatten=True).\
                               fit_transform(df['matrix_pca99']))
                      ]

    print('Saving and compressing datasets...')

    DB_PROD_NAME = df_name + '_matrix'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    dump(df_matrix, DB_PROD_PATH, compress=('xz',9))

    DB_PROD_NAME = df_name + '_num_cp'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    dump(df_num_cp, DB_PROD_PATH, compress=('xz',9))

    DB_PROD_NAME = df_name + '_dim_cp'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    dump(df_dim_cp, DB_PROD_PATH, compress=('xz',9))

    DB_PROD_NAME = df_name + '_eng_h11'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    dump(df_eng_h11, DB_PROD_PATH, compress=('xz',9))

    DB_PROD_NAME = df_name + '_eng_h21'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    dump(df_eng_h21, DB_PROD_PATH, compress=('xz',9))

    print('Datasets have been saved and compressed!')
