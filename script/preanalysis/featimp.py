# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# FEATIMP:
#
#   Compute the feature importances using a forest of decision trees.
#
# AUTHOR: Riccardo Finotello
#

import numpy             as np
import sklearn           as sk

assert np.__version__ >  '1.16'   # to avoid issues with pytables
assert sk.__version__ >= '0.22.1' # for the recent implementation

from os                   import path, mkdir
from sklearn.ensemble     import RandomForestRegressor
from toolset.libutilities import *
from toolset.libplot      import *

# Set working directories
ROOT_DIR = '.' # root directory

def importances(df_name, cluster_range, seed=42):

    # Read the database
    DB_PROD_NAME = df_name + '_features_analysis'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the database!', flush=True)

    DB_PROD_NAME = df_name + '_labels_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df_labels = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the labels database!', flush=True)

    # Divide scalar, vector and tensor features
    scalar_features = list(df.\
            select_dtypes(include=['int8', 'int64', 'float64']).columns)
    vector_features = list(df.\
            drop(labels=['matrix', 'matrix_pca99'], axis=1).\
            select_dtypes(include=['object']).columns)
    tensor_features = list(df.\
            drop(labels=vector_features, axis=1).\
            select_dtypes(include=['object']).columns)

    # Set random seed
    RAND = seed
    np.random.seed(RAND)

    # Create a matrix to hold all the features to be fed to the Decision Tree:
    dtree_features = df[scalar_features].values
    for feature in vector_features:
        dtree_features = np.c_[dtree_features,
                               ExtractTensor(flatten=True).\
                                       fit_transform(df[feature])]
    for feature in tensor_features + ['matrix_pca99']:
        dtree_features = np.c_[dtree_features,
                               ExtractTensor(flatten=True).\
                                       fit_transform(df[feature])]


    # Then create the Decision Tree and train it:
    params = { 'criterion':    'mse',
               'n_estimators': 48,
               'n_jobs':       -1,
               'random_state': RAND
             }

    print('\nStudying feature importances:', flush=True)

    dtree_h11 = RandomForestRegressor(**params)
    print('    Fitting decision trees for h_11...', flush=True)
    dtree_h11.fit(dtree_features, df_labels['h11'])

    dtree_h21 = RandomForestRegressor(**params)
    print('    Fitting decision trees for h_21...', flush=True)
    dtree_h21.fit(dtree_features, df_labels['h21'])

    # check accuracy just "for fun"
    print('    Accuracy for h_11: {:.3f}%'.format(\
            accuracy_score(df_labels['h11'].values,
                           dtree_h11.predict(dtree_features))*100))
    print('    Accuracy for h_21: {:.3f}%'.format(\
            accuracy_score(df_labels['h21'].values,
                           dtree_h21.predict(dtree_features))*100))


    # We then plot the feature importances together with their labels:
    feature_extended_labels = []
    for column in df:
        length = np.prod(df[column].values[0].shape).astype(int)
        if column == 'clustering':
            for n in range(length):
                feature_extended_labels.append('kmeans_{:d}'.\
                        format(min(cluster_range)+n))
        else:
            feature_extended_labels.append(column)
            for n in range(length-1):
                feature_extended_labels.append('')

    importances_h11 = list(zip(feature_extended_labels,
                               dtree_h11.feature_importances_))
    importances_h21 = list(zip(feature_extended_labels,
                               dtree_h21.feature_importances_))

    fig, plot = plt.subplots(2, 2, figsize=(12, 10))
    fig.tight_layout()

    label_plot(plot[0,0],
               importances_h11[0:10],
               title='Scalar Features',
               ylabel='Feature Importances',
               legend='$h_{11}$'
              )
    label_plot(plot[0,0],
               importances_h21[0:10],
               title='Scalar Features',
               ylabel='Feature Importances',
               legend='$h_{21}$'
              )

    label_plot(plot[0,1],
               importances_h11[10:28],
               title='Clustering Features',
               ylabel='Feature Importances',
               legend='$h_{11}$'
              )
    label_plot(plot[0,1],
               importances_h21[10:28],
               title='Clustering Features',
               ylabel='Feature Importances',
               legend='$h_{21}$'
              )

    label_plot(plot[1,0],
               importances_h11[28:90],
               title='Vector Features',
               ylabel='Feature Importances',
               legend='$h_{11}$'
              )
    label_plot(plot[1,0],
               importances_h21[28:90],
               title='Vector Features',
               ylabel='Feature Importances',
               legend='$h_{21}$'
              )

    label_plot(plot[1,1],
               importances_h11[90:],
               title='Tensor Features',
               ylabel='Feature Importances',
               legend='$h_{11}$'
              )
    label_plot(plot[1,1],
               importances_h21[90:],
               title='Tensor Features',
               ylabel='Feature Importances',
               legend='$h_{21}$'
              )

    save_fig('feature_importances')
    # plt.show()
    plt.close(fig)

    vector_importances_h11_sum = [ ('dim_cp_sum',     np.sum([ f[1] for f in importances_h11[28:42] ]) ),
                                   ('num_dim_cp_sum',  np.sum([ f[1] for f in importances_h11[42:49] ]) ),
                                   ('deg_eqs_sum',     np.sum([ f[1] for f in importances_h11[49:67] ]) ),
                                   ('num_deg_eqs_sum', np.sum([ f[1] for f in importances_h11[67:75] ]) ),
                                   ('dim_h0_amb_sum',  np.sum([ f[1] for f in importances_h11[75:90] ]) )
                                 ]
    vector_importances_h21_sum = [ ('dim_cp_sum',      np.sum([ f[1] for f in importances_h21[28:42] ]) ),
                                   ('num_dim_cp_sum',  np.sum([ f[1] for f in importances_h21[42:49] ]) ),
                                   ('deg_eqs_sum',     np.sum([ f[1] for f in importances_h21[49:67] ]) ),
                                   ('num_deg_eqs_sum', np.sum([ f[1] for f in importances_h21[67:75] ]) ),
                                   ('dim_h0_amb_sum',  np.sum([ f[1] for f in importances_h21[75:90] ]) )
                                 ]

    tensor_importances_h11_sum = [ ('matrix_sum',      np.sum([ f[1] for f in importances_h11[90:360] ]) ),
                                   ('pca99_sum',       np.sum([ f[1] for f in importances_h11[360:] ]) )
                                 ]
    tensor_importances_h21_sum = [ ('matrix_sum',      np.sum([ f[1] for f in importances_h21[90:360] ]) ),
                                   ('pca99_sum',       np.sum([ f[1] for f in importances_h21[360:] ]) )
                                 ]

    fig, plot = plt.subplots(1, 2, figsize=(12,5))
    fig.tight_layout()

    label_plot(plot[0],
               vector_importances_h11_sum,
               title='Sum of Vector Features',
               ylabel='Feature Importances',
               legend='$h_{11}$'
              )
    label_plot(plot[0],
               vector_importances_h21_sum,
               title='Sum of Vector Features',
               ylabel='Feature Importances',
               legend='$h_{21}$'
              )

    label_plot(plot[1],
               tensor_importances_h11_sum,
               title='Sum of Tensor Features',
               ylabel='Feature Importances',
               legend='$h_{11}$'
              )
    label_plot(plot[1],
               tensor_importances_h21_sum,
               title='Sum of Tensor Features',
               ylabel='Feature Importances',
               legend='$h_{21}$'
              )

    save_fig('feature_importances_vector_tensor_sum')
    # plt.show()
    plt.close(fig)


    importances_h11_sum = [ ('scalar',  np.sum([ f[1] for f in importances_h11[0:10] ]) ),
                            ('vector',  np.sum([ f[1] for f in importances_h11[28:90] ]) ),
                            ('tensor',  np.sum([ f[1] for f in importances_h11[90:360] ]) ),
                            ('cluster', np.sum([ f[1] for f in importances_h11[10:28] ]) ),
                            ('pca',     np.sum([ f[1] for f in importances_h11[360:] ]) ),
                          ]
    importances_h21_sum = [ ('scalar',  np.sum([ f[1] for f in importances_h21[0:10] ]) ),
                            ('vector',  np.sum([ f[1] for f in importances_h21[28:90] ]) ),
                            ('tensor',  np.sum([ f[1] for f in importances_h21[90:360] ]) ),
                            ('cluster', np.sum([ f[1] for f in importances_h21[10:28] ]) ),
                            ('pca',     np.sum([ f[1] for f in importances_h21[364:] ]) ),
                          ]

    fig, plot = plt.subplots(figsize=(6,5))
    fig.tight_layout()

    label_plot(plot,
               importances_h11_sum,
               title='Sum of Features',
               ylabel='Feature Importances',
               legend='$h_{11}$'
              )
    label_plot(plot,
               importances_h21_sum,
               title='Sum of Features',
               ylabel='Feature Importances',
               legend='$h_{21}$'
              )

    save_fig('feature_importances_sum')
    # plt.show()
    plt.close(fig)
