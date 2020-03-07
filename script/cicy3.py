# Machine Learning for Complete Intersection Calabi-Yau Manifolds
# 
# We use machine learning techniques to predict the Hodge numbers of Calabi-Yau
# 3-folds. The relevant numbers are h_11 and h_21. We use several approaches.
# Specifically:
# 
# 1. evaluate different algorithms using Scikit-learn (and XGBoost, if needed).
# For each one of them build a baseline with only the configuration matrix for
# both h_11 and h_21, then use feature engineering to improve the results):
#     
#     - Linear Regression,
#     - Lasso,
#     - ElasticNet,
#     - Ridge,
#     - LinearSVR,
#     - SVR (rbf),
#     - Boosted Tree (Gradient Boosting),
#     - Random Forest,
#     
# 2. build a working Convolutional Neural Network using Tensorflow-Keras:
# 
#     - build a baseline with a sequential model,
#     - use the functional API to improve the net,
#     
# 3. build a fully connected Deep Neural Network using only dense layers using
# the same framework in order to have another usable model with a completely
# different architecture,
#     
# 4. use stacking to improve the overall result (keep in mind that no matter how
# bad an algorithm can be, its stacked version can significantly improve,
# provided that the stacked algorithms are sufficiently diverse amongst them).
#
# AUTHOR: Riccardo Finotello
#

import sys
import tarfile

assert sys.version_info >= (3, 6)  # require at least Python 3.6

from urllib                   import request as rq
from os                       import environ, path, mkdir
from toolset.libplot          import *
from toolset.libutilities     import *
from visualization.dataview   import *
from visualization.extraction import *
from preanalysis.clustering   import *
from preanalysis.pca          import *
from preanalysis.featimp      import *
from preanalysis.featext      import *
from algorithms               import linear_regression, \
                                     lasso, \
                                     elastic_net, \
                                     ridge, \
                                     linear_svr, \
                                     svr_rbf, \
                                     xgb, \
                                     xgbrf, \
                                     grd_boost, \
                                     rnd_for, \
                                     sequential_matrix, \
                                     functional_matrix, \
                                     functional_matrix_pca, \
                                     functional_matrix_pca_dense, \
                                     stacking

environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # do not print Tensorflow warnings

def main():

    ROOT_DIR = '.'      # root directory
    # LOG_DIR  = 'log'    # logs directory

    # # Create and/or define the directories
    # LOG_PATH = path.join(ROOT_DIR, LOG_DIR)
    # if path.isdir(LOG_PATH) is False:
    #     mkdir(LOG_PATH)

    # Name of the dataset
    if len(sys.argv) > 1:
        DB_NAME = sys.argv[1]
    else:
        print('Please, provide the name of the database!', flush=True)
        exit(1)

    # Random seed
    if len(sys.argv) > 2:
        RAND = int(sys.argv[2])
    else:
        print('Please, provide the random seed!', flush=True)
        exit(1)

    # Download the dataset
    URL_ROOT = 'http://www.lpthe.jussieu.fr/~erbin/files/data/'
    TAR      = DB_NAME + '_data.tar.gz'
    URL      = URL_ROOT + TAR

    # Request the file
    TAR_PATH = path.join(ROOT_DIR, TAR)
    if not path.isfile(TAR_PATH):
        print('Requesting database from source URL...', flush=True)
        rq.urlretrieve(URL, TAR_PATH)
        print('Receiving database from source URL...', flush=True)

    # Extract the database from the tarball
    DB_FILE = DB_NAME + '.h5'
    DB_PATH = path.join(ROOT_DIR, DB_FILE)

    # Extract the database
    if not path.isfile(DB_PATH):
        print('Extracting tarball...', flush=True)
        tar = tarfile.open(TAR_PATH, 'r:gz')
        tar.extract(DB_FILE)
        print('Extracted {}'.format(DB_PATH), flush=True)

    # Load the database
    df = load_dataset(DB_PATH)

    # Create visualizations and plots from the dataset (returns cleaned dataset)
    df = data_visualization(df)

    # Extract features and labels
    data_extraction(df, DB_NAME)

    # Now consider the clustering analysis
    cluster_range = clustering(DB_NAME, seed=RAND)

    # Now consider the PCA analysis
    pca(DB_NAME, seed=RAND)

    # Compute the feature importances
    importances(DB_NAME, cluster_range, seed=RAND)

    # Extract usable features
    feature_extract(DB_NAME)

    # Compute algorithms
    linear_regression.compute(DB_NAME, rounding=np.floor, seed=RAND)
    lasso.compute(DB_NAME, n_iter=50, rounding=np.floor, seed=RAND)
    elastic_net.compute(DB_NAME, n_iter=50, rounding=np.floor, seed=RAND)
    ridge.compute(DB_NAME, n_iter=50, rounding=np.floor, seed=RAND)
    linear_svr.compute(DB_NAME, n_iter=50, rounding=np.floor, seed=RAND)
    svr_rbf.compute(DB_NAME, n_iter=50, rounding=np.rint, seed=RAND)
    xgb.compute(DB_NAME, n_iter=15, rounding=np.floor, seed=RAND)
    xgbrf.compute(DB_NAME, n_iter=15, rounding=np.floor, seed=RAND)
    rnd_for.compute(DB_NAME, n_iter=15, rounding=np.floor, seed=RAND)
    grd_boost.compute(DB_NAME, n_iter=15, rounding=np.floor, seed=RAND)
    sequential_matrix.compute(DB_NAME, rounding=np.rint, seed=RAND)
    functional_matrix.compute(DB_NAME, rounding=np.rint, seed=RAND)
    functional_matrix_pca.compute(DB_NAME, rounding=np.rint, seed=RAND)
    functional_matrix_pca_dense.compute(DB_NAME, rounding=np.rint, seed=RAND)

    # Stack the algorithms
    stacking.compute(DB_NAME, n_iter=50, seed=RAND)

if __name__ == '__main__':
    main()
