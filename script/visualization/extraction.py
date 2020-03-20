# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# EXTRACTION:
#
#   Library of definitions and classes to extract usable features
#
# AUTHOR: Riccardo Finotello
#

import pandas as pd

from os                   import path
from toolset.libutilities import *
from toolset.libplot      import *

# Set working directories
ROOT_DIR = '.'      # root directory

def data_extraction(df, df_name):

    # Select features to drop
    features_to_drop = list(df.filter(regex='min_|max_|median_|mean_').columns)\
                       + ['c2', 'isprod', 'favour', 'size']
    df               = df.drop(labels=features_to_drop, axis=1)

    labels      = ['h11', 'h21', 'euler']
    df_features = df.drop(labels=labels, axis=1)
    df_labels   = df[labels]

    # Separate into scalar, vector and tensor features
    scalar_features = list(df_features.\
            select_dtypes(include=['int8', 'int64', 'float64']).columns)
    vector_features = list(df_features.\
            drop(labels='matrix', axis=1).\
            select_dtypes(include=['object']).columns)
    tensor_features = list(df_features.\
            drop(labels=vector_features, axis=1).\
            select_dtypes(include=['object']).columns)

    # Extract tensor and vectors
    print('\nExtracting tensors...', flush=True)

    for feature in vector_features:
        df_features[feature] = ExtractTensor(flatten=False).\
                fit_transform(df_features[feature])
    for feature in tensor_features:
        df_features[feature] = ExtractTensor(flatten=False).\
                fit_transform(df_features[feature])
        
    df_features = df_features[scalar_features +
                              vector_features +
                              tensor_features] # simple reorder of the dataframe


    # Plot the correlation matrix as a reference
    print('\nPlotting the correlation matrix:', flush=True)

    fig, plot = plt.subplots(1, 3, figsize=(18,5))
    fig.tight_layout()

    mat_plot(plot[0], df_labels)
    mat_plot(plot[1], df_features[scalar_features])
    mat_plot(plot[2], df_features.join(df_labels)[labels +
                                                  [ 'num_cp',
                                                    'num_cp_1',
                                                    'num_cp_2',
                                                    'num_cp_neq1',
                                                    'rank_matrix',
                                                    'norm_matrix'
                                                  ]
                                                 ]
            )

    save_fig('correlation_matrix')
    # plt.show()
    plt.close(fig)


    # Save 'production-ready' dataframe to file
    print('\nSaving features and labels to file...', flush=True)
    DB_PROD_NAME = df_name + '_features_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    df_features.to_hdf(DB_PROD_PATH, key='df_features')
    print('    Features have been saved to file!', flush=True)

    DB_PROD_NAME = df_name + '_labels_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    df_labels.to_hdf(DB_PROD_PATH, key='df_labels')
    print('    Labels have been saved to file!', flush=True)
