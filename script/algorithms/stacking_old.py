# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# STACKING:
#
#   Stack the computed algorithms to improve accuracy.
#
# AUTHOR: Riccardo Finotello
#

import numpy      as np
import sklearn    as sk
import pandas     as pd
import tensorflow as tf

assert np.__version__ >  '1.16'   # to avoid issues with pytables
assert sk.__version__ >= '0.22.1' # for the recent implementation
assert tf.__version__ >= '2.0.0'  # newest version

from os                          import path
from joblib                      import load, dump
from sklearn.preprocessing       import StandardScaler, \
                                        MinMaxScaler, \
                                        RobustScaler, \
                                        MaxAbsScaler
from sklearn.metrics             import make_scorer
from sklearn.model_selection     import train_test_split, \
                                        cross_val_predict, \
                                        KFold, \
                                        GridSearchCV
from sklearn.linear_model        import LinearRegression
from sklearn.svm                 import SVR
from sklearn.ensemble            import RandomForestRegressor
from skopt                       import BayesSearchCV
from skopt.space                 import Categorical, Real, Integer
from tensorflow.keras            import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import ReduceLROnPlateau, \
                                        ModelCheckpoint, \
                                        EarlyStopping
from tensorflow.keras.utils      import plot_model
from toolset.libutilities        import *
from toolset.libplot             import *
from toolset.libnn               import *

# Set working directories
ROOT_DIR = '.' # root directory
MOD_DIR  = 'models' # models directory
MOD_PATH = path.join(ROOT_DIR, MOD_DIR)
if path.isdir(MOD_PATH) is False:
    mkdir(MOD_PATH)
    
def meta_fit(estimator,         # Fit the same estimator with scalers
             scalers,
             X_validation,
             y_validation,
             X_test,
             y_test,
             rounding=np.rint):

    print('  --> Fit without scalers:', flush=True)
    estimator.fit(X_validation, y_validation)
    gridcv_score(estimator, rounding=rounding)
    prediction_score(estimator, X_test, y_test, rounding=rounding)
    
    for scaler in scalers:
        print('  --> Fit with {}:'.format(scaler[0]), flush=True)
        X_validation = scaler[1].fit_transform(X_validation)
        X_test       = scaler[1].transform(X_test)
        estimator.fit(X_validation, y_validation)
        gridcv_score(estimator, rounding=rounding)
        prediction_score(estimator, X_test, y_test, rounding=rounding)

def compute(df_name, n_iter=30, seed=42):

    # Print banner
    print('\n----- STACKING -----', flush=True)

    # Set random seed
    RAND = seed
    np.random.seed(RAND)
    tf.random.set_seed(RAND)

    # Load models
    print('\nLoading models:', flush=True)
    print('    Loading LinearRegressor...', flush=True)
    lin_reg_h11 = load(path.join(MOD_PATH, 'lin_reg_h11.joblib.xz'))
    lin_reg_h21 = load(path.join(MOD_PATH, 'lin_reg_h21.joblib.xz'))

    print('    Loading Lasso...', flush=True)
    lasso_h11   = load(path.join(MOD_PATH, 'lasso_h11.joblib.xz'))
    lasso_h21   = load(path.join(MOD_PATH, 'lasso_h21.joblib.xz'))

    print('    Loading ElasticNet...', flush=True)
    el_net_h11  = load(path.join(MOD_PATH, 'el_net_h11.joblib.xz'))
    el_net_h21  = load(path.join(MOD_PATH, 'el_net_h21.joblib.xz'))

    print('    Loading Ridge...', flush=True)
    ridge_h11   = load(path.join(MOD_PATH, 'ridge_h11.joblib.xz'))
    ridge_h21   = load(path.join(MOD_PATH, 'ridge_h21.joblib.xz'))

    print('    Loading LinearSVR...', flush=True)
    lin_svr_h11 = load(path.join(MOD_PATH, 'lin_svr_h11.joblib.xz'))
    lin_svr_h21 = load(path.join(MOD_PATH, 'lin_svr_h21.joblib.xz'))

    print('    Loading SVR...', flush=True)
    svr_rbf_h11 = load(path.join(MOD_PATH, 'svr_rbf_h11.joblib.xz'))
    svr_rbf_h21 = load(path.join(MOD_PATH, 'svr_rbf_h21.joblib.xz'))

    print('    Loading XGBRegressor...', flush=True)
    xgb_h11     = load(path.join(MOD_PATH, 'xgb_h11.joblib.xz'))
    xgb_h21     = load(path.join(MOD_PATH, 'xgb_h21.joblib.xz'))

    print('    Loading XGBRFRegressor...', flush=True)
    xgbrf_h11   = load(path.join(MOD_PATH, 'xgbrf_h11.joblib.xz'))
    xgbrf_h21   = load(path.join(MOD_PATH, 'xgbrf_h21.joblib.xz'))

    print('    Loading the Sequential Keras model...', flush=True)
    seq_h11     = load_model(path.join(MOD_PATH, 'cnn_matrix_sequential_h11.h5'))
    seq_h21     = load_model(path.join(MOD_PATH, 'cnn_matrix_sequential_h21.h5'))

    print('    Loading the Functional Keras model...', flush=True)
    matrix_functional           = load_model(path.join(MOD_PATH,
                                                       'cnn_functional.h5'))
    print('    Loading the Functional Keras model with PCA...', flush=True)
    matrix_functional_pca       = load_model(path.join(MOD_PATH,
                                                       'cnn_functional_pca.h5'))
    print('    Loading the Functional Keras model with PCA and Dense layers...', flush=True)
    matrix_functional_pca_dense = load_model(path.join(MOD_PATH,
                                                       'cnn_functional_pca_dense.h5'))


    # Load database
    DB_PROD_NAME = df_name + '_matrix'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_matrix = load(DB_PROD_PATH)
    else:
        print('Cannot read the matrix database!', flush=True)
        
    DB_PROD_NAME = df_name + '_eng_h11'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_eng_h11 = load(DB_PROD_PATH)
    else:
        print('Cannot read the eng_h11 database!', flush=True)
        
    DB_PROD_NAME = df_name + '_eng_h21'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_eng_h21 = load(DB_PROD_PATH)
    else:
        print('Cannot read the eng_h21 database!', flush=True)
        
    DB_PROD_NAME = df_name + '_labels_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df_labels = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the labels database!', flush=True)

    # Split into training and test
    df_matrix_train, df_matrix_test, \
    df_eng_h11_train, df_eng_h11_test, \
    df_eng_h21_train, df_eng_h21_test, \
    df_labels_train, df_labels_test = train_test_split(df_matrix, 
                                                       df_eng_h11,
                                                       df_eng_h21,
                                                       df_labels,
                                                       test_size=1/10,
                                                       shuffle=True,
                                                       random_state=RAND)

    # Split the training set into two subsets
    df_matrix_train, df_matrix_val, \
    df_eng_h11_train, df_eng_h11_val, \
    df_eng_h21_train, df_eng_h21_val, \
    df_labels_train, df_labels_val = train_test_split(df_matrix_train,
                                                      df_eng_h11_train,
                                                      df_eng_h21_train,
                                                      df_labels_train,
                                                      test_size=1/2,
                                                      shuffle=True,
                                                      random_state=RAND)


    # # Train the algorithms:
    # print('\nTraining models:', flush=True)
    # print('    Fitting the LinearRegressor...', flush=True)
    # lin_reg_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # lin_reg_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the Lasso...', flush=True)
    # lasso_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # lasso_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the ElasticNet...', flush=True)
    # el_net_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # el_net_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the Ridge...', flush=True)
    # ridge_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # ridge_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the LinearSVR...', flush=True)
    # lin_svr_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # lin_svr_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the SVR...', flush=True)
    # svr_rbf_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # svr_rbf_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the XGBRegressor...', flush=True)
    # xgb_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # xgb_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the XGBRFRegressor...', flush=True)
    # xgbrf_h11.fit(df_eng_h11_train, df_labels_train['h11'])
    # xgbrf_h21.fit(df_eng_h21_train, df_labels_train['h21'])

    # print('    Fitting the Sequential Keras models...', flush=True)
    # seq_h11.fit(x=K.cast(df_matrix_train.reshape(-1,12,15,1), dtype='float64'),
    #             y=K.cast(df_labels_train['h11'], dtype='float64'),
    #             batch_size=32,
    #             epochs=300,
    #             verbose=0
    #            )
    # seq_h21.fit(x=K.cast(df_matrix_train.reshape(-1,12,15,1), dtype='float64'),
    #             y=K.cast(df_labels_train['h21'], dtype='float64'),
    #             batch_size=32,
    #             epochs=300,
    #             verbose=0
    #            )

    # print('    Fitting the Functional Conv2D Keras models...', flush=True)
    # matrix_functional.fit(x=[K.cast(df_eng_h21_train[:,0].reshape(-1,1),
    #                                 dtype='float64'),
    #                          K.cast(df_eng_h21_train[:,1:13].reshape(-1,12,1),
    #                                 dtype='float64'),
    #                          K.cast(df_eng_h21_train[:,13:28].reshape(-1,15,1),
    #                                 dtype='float64'),
    #                          K.cast(df_matrix_train.reshape(-1,12,15,1),
    #                                 dtype='float64')],
    #                       y=[K.cast(df_labels_train['h11'], dtype='float64'),
    #                          K.cast(df_labels_train['h21'], dtype='float64')],
    #                       batch_size=32,
    #                       epochs=300,
    #                       verbose=0
    #                      )

    # print('    Fitting the Functional Conv1D Keras models...', flush=True)
    # matrix_functional_pca.fit(x=[K.cast(df_eng_h21_train[:,0].reshape(-1,1),
    #                                     dtype='float64'),
    #                              K.cast(df_eng_h21_train[:,1:13].reshape(-1,12,1),
    #                                     dtype='float64'),
    #                              K.cast(df_eng_h21_train[:,13:28].reshape(-1,15,1),
    #                                     dtype='float64'),
    #                              K.cast(df_eng_h21_train[:,28:].reshape(-1,81,1),
    #                                     dtype='float64')],
    #                           y=[K.cast(df_labels_train['h11'], dtype='float64'),
    #                              K.cast(df_labels_train['h21'], dtype='float64')],
    #                           batch_size=32,
    #                           epochs=300,
    #                           verbose=0
    #                          )

    # print('    Fitting the Functional Dense Keras models...', flush=True)
    # matrix_functional_pca_dense.fit(x=[K.cast(df_eng_h21_train[:,0].reshape(-1,1), 
    #                                           dtype='float64'),
    #                                    K.cast(df_eng_h21_train[:,1:13],
    #                                           dtype='float64'),
    #                                    K.cast(df_eng_h21_train[:,13:28],
    #                                           dtype='float64'),
    #                                    K.cast(df_eng_h21_train[:,28:],
    #                                           dtype='float64')],
    #                                 y=[K.cast(df_labels_train['h11'],
    #                                           dtype='float64'),
    #                                    K.cast(df_labels_train['h21'],
    #                                           dtype='float64')],
    #                                 batch_size=32,
    #                                 epochs=300,
    #                                 verbose=0
    #                                )

    # print('End of the training procedure!', flush=True)


    # Use the trained models to build the second level predictions:
    print('\nPredictions of the LinearRegressor...', flush=True)
    lin_reg_h11_predictions_val = lin_reg_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   lin_reg_h11_predictions_val,
                                   rounding=np.floor)*100))
    lin_reg_h21_predictions_val = lin_reg_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   lin_reg_h21_predictions_val,
                                   rounding=np.floor)*100))

    lin_reg_h11_predictions_test = lin_reg_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   lin_reg_h11_predictions_test,
                                   rounding=np.floor)*100))
    lin_reg_h21_predictions_test = lin_reg_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   lin_reg_h21_predictions_test,
                                   rounding=np.floor)*100))

    print('Predictions of the Lasso...', flush=True)
    lasso_h11_predictions_val = lasso_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   lasso_h11_predictions_val,
                                   rounding=np.floor)*100))
    lasso_h21_predictions_val = lasso_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   lasso_h21_predictions_val,
                                   rounding=np.floor)*100))

    lasso_h11_predictions_test = lasso_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   lasso_h11_predictions_test,
                                   rounding=np.floor)*100))
    lasso_h21_predictions_test = lasso_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   lasso_h21_predictions_test,
                                   rounding=np.floor)*100))

    print('Predictions of the ElasticNet...', flush=True)
    el_net_h11_predictions_val = el_net_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   el_net_h11_predictions_val,
                                   rounding=np.floor)*100))
    el_net_h21_predictions_val = el_net_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   el_net_h21_predictions_val,
                                   rounding=np.floor)*100))

    el_net_h11_predictions_test = el_net_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   el_net_h11_predictions_test,
                                   rounding=np.floor)*100))
    el_net_h21_predictions_test = el_net_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   el_net_h21_predictions_test,
                                   rounding=np.floor)*100))

    print('Predictions of the Ridge...', flush=True)
    ridge_h11_predictions_val = ridge_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   ridge_h11_predictions_val,
                                   rounding=np.floor)*100))
    ridge_h21_predictions_val = ridge_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   ridge_h21_predictions_val,
                                   rounding=np.floor)*100))

    ridge_h11_predictions_test = ridge_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   ridge_h11_predictions_test,
                                   rounding=np.floor)*100))
    ridge_h21_predictions_test = ridge_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   ridge_h21_predictions_test,
                                   rounding=np.floor)*100))

    print('Predictions of the LinearSVR...', flush=True)
    lin_svr_h11_predictions_val = lin_svr_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   lin_svr_h11_predictions_val,
                                   rounding=np.floor)*100))
    lin_svr_h21_predictions_val = lin_svr_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   lin_svr_h21_predictions_val,
                                   rounding=np.floor)*100))

    lin_svr_h11_predictions_test = lin_svr_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   lin_svr_h11_predictions_test,
                                   rounding=np.floor)*100))
    lin_svr_h21_predictions_test = lin_svr_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   lin_svr_h21_predictions_test,
                                   rounding=np.floor)*100))

    print('Predictions of the SVR...', flush=True)
    svr_rbf_h11_predictions_val = svr_rbf_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   svr_rbf_h11_predictions_val,
                                   rounding=np.rint)*100))
    svr_rbf_h21_predictions_val = svr_rbf_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   svr_rbf_h21_predictions_val,
                                   rounding=np.rint)*100))

    svr_rbf_h11_predictions_test = svr_rbf_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   svr_rbf_h11_predictions_test,
                                   rounding=np.rint)*100))
    svr_rbf_h21_predictions_test = svr_rbf_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   svr_rbf_h21_predictions_test,
                                   rounding=np.rint)*100))

    print('Predictions of the XGBRegressor...', flush=True)
    xgb_h11_predictions_val = xgb_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   xgb_h11_predictions_val,
                                   rounding=np.floor)*100))
    xgb_h21_predictions_val = xgb_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   xgb_h21_predictions_val,
                                   rounding=np.floor)*100))

    xgb_h11_predictions_test = xgb_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   xgb_h11_predictions_test,
                                   rounding=np.floor)*100))
    xgb_h21_predictions_test = xgb_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   xgb_h21_predictions_test,
                                   rounding=np.floor)*100))

    print('Predictions of the XGBRFRegressor...', flush=True)
    xgbrf_h11_predictions_val = xgbrf_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   xgbrf_h11_predictions_val,
                                   rounding=np.floor)*100))
    xgbrf_h21_predictions_val = xgbrf_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   xgbrf_h21_predictions_val,
                                   rounding=np.floor)*100))

    xgbrf_h11_predictions_test = xgbrf_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   xgbrf_h11_predictions_test,
                                   rounding=np.floor)*100))
    xgbrf_h21_predictions_test = xgbrf_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   xgbrf_h21_predictions_test,
                                   rounding=np.floor)*100))


    print('Predictions of the Sequential Keras models...', flush=True)
    seq_h11_predictions_val = seq_h11.predict(K.cast(df_matrix_val.reshape(-1,12,15,1), dtype='float64'))
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   seq_h11_predictions_val,
                                   rounding=np.rint)*100))
    seq_h21_predictions_val = seq_h21.predict(K.cast(df_matrix_val.reshape(-1,12,15,1), dtype='float64'))
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   seq_h21_predictions_val,
                                   rounding=np.rint)*100))

    seq_h11_predictions_test = seq_h11.predict(K.cast(df_matrix_test.reshape(-1,12,15,1), dtype='float64'))
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   seq_h11_predictions_test,
                                   rounding=np.rint)*100))
    seq_h21_predictions_test = seq_h21.predict(K.cast(df_matrix_test.reshape(-1,12,15,1), dtype='float64'))
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   seq_h21_predictions_test,
                                   rounding=np.rint)*100))

    print('Predictions of the Functional Conv2D Keras models...', flush=True)
    matrix_functional_predictions_val = \
    matrix_functional.predict([K.cast(df_eng_h21_val[:,0].reshape(-1,1),
                                      dtype='float64'),
                               K.cast(df_eng_h21_val[:,1:13].reshape(-1,12,1),
                                      dtype='float64'),
                               K.cast(df_eng_h21_val[:,13:28].reshape(-1,15,1),
                                      dtype='float64'),
                               K.cast(df_matrix_val.reshape(-1,12,15,1),
                                      dtype='float64')])
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   matrix_functional_predictions_val[0],
                                   rounding=np.rint)*100))
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   matrix_functional_predictions_val[1],
                                   rounding=np.rint)*100))

    matrix_functional_predictions_test = \
    matrix_functional.predict([K.cast(df_eng_h21_test[:,0].reshape(-1,1),
                                      dtype='float64'),
                               K.cast(df_eng_h21_test[:,1:13].reshape(-1,12,1),
                                      dtype='float64'),
                               K.cast(df_eng_h21_test[:,13:28].reshape(-1,15,1),
                                   dtype='float64'),
                               K.cast(df_matrix_test.reshape(-1,12,15,1),
                                   dtype='float64')])
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   matrix_functional_predictions_test[0],
                                   rounding=np.rint)*100))
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   matrix_functional_predictions_test[1],
                                   rounding=np.rint)*100))

    print('Predictions of the Functional Conv1D Keras models...', flush=True)
    matrix_functional_pca_predictions_val = \
    matrix_functional_pca.predict([K.cast(df_eng_h21_val[:,0].reshape(-1,1),
                                          dtype='float64'),
                                   K.cast(df_eng_h21_val[:,1:13].reshape(-1,12,1),
                                          dtype='float64'),
                                   K.cast(df_eng_h21_val[:,13:28].reshape(-1,15,1),
                                          dtype='float64'),
                                   K.cast(df_eng_h21_val[:,28:].reshape(-1,81,1),
                                          dtype='float64')])
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   matrix_functional_pca_predictions_val[0],
                                   rounding=np.rint)*100))
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   matrix_functional_pca_predictions_val[1],
                                   rounding=np.rint)*100))

    matrix_functional_pca_predictions_test = \
    matrix_functional_pca.predict([K.cast(df_eng_h21_test[:,0].reshape(-1,1),
                                          dtype='float64'),
                                   K.cast(df_eng_h21_test[:,1:13].reshape(-1,12,1),
                                          dtype='float64'),
                                   K.cast(df_eng_h21_test[:,13:28].reshape(-1,15,1),
                                          dtype='float64'),
                                   K.cast(df_eng_h21_test[:,28:].reshape(-1,81,1),
                                          dtype='float64')])
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   matrix_functional_pca_predictions_test[0],
                                   rounding=np.rint)*100))
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   matrix_functional_pca_predictions_test[1],
                                   rounding=np.rint)*100))


    print('Predictions of the Functional Dense Keras models...', flush=True)
    matrix_functional_pca_dense_predictions_val = \
    matrix_functional_pca_dense.predict([K.cast(df_eng_h21_val[:,0].reshape(-1,1),
                                                dtype='float64'),
                                         K.cast(df_eng_h21_val[:,1:13],
                                                dtype='float64'),
                                         K.cast(df_eng_h21_val[:,13:28],
                                                dtype='float64'),
                                         K.cast(df_eng_h21_val[:,28:],
                                                dtype='float64')])
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   matrix_functional_pca_dense_predictions_val[0],
                                   rounding=np.rint)*100))
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   matrix_functional_pca_dense_predictions_val[1],
                                   rounding=np.rint)*100))

    matrix_functional_pca_dense_predictions_test = \
    matrix_functional_pca_dense.predict([K.cast(df_eng_h21_test[:,0].reshape(-1,1),
                                                dtype='float64'),
                                         K.cast(df_eng_h21_test[:,1:13],
                                                dtype='float64'),
                                         K.cast(df_eng_h21_test[:,13:28],
                                                dtype='float64'),
                                         K.cast(df_eng_h21_test[:,28:],
                                                dtype='float64')])
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   matrix_functional_pca_dense_predictions_test[0],
                                   rounding=np.rint)*100))
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   matrix_functional_pca_dense_predictions_test[1],
                                   rounding=np.rint)*100))

    print('Concatenatig prediction vectors...', flush=True)
    h11_predictions_val = np.c_[lin_reg_h11_predictions_val,
                                lasso_h11_predictions_val,
                                el_net_h11_predictions_val,
                                ridge_h11_predictions_val,
                                lin_svr_h11_predictions_val,
                                svr_rbf_h11_predictions_val,
                                xgb_h11_predictions_val,
                                xgbrf_h11_predictions_val,
                                seq_h11_predictions_val,
                                matrix_functional_predictions_val[0],
                                matrix_functional_pca_predictions_val[0],
                                matrix_functional_pca_dense_predictions_val[0]
                               ]
    h21_predictions_val = np.c_[lin_reg_h11_predictions_val,
                                lasso_h11_predictions_val,
                                el_net_h11_predictions_val,
                                ridge_h11_predictions_val,
                                lin_svr_h11_predictions_val,
                                svr_rbf_h11_predictions_val,
                                xgb_h11_predictions_val,
                                xgbrf_h11_predictions_val,
                                seq_h11_predictions_val,
                                matrix_functional_predictions_val[1],
                                matrix_functional_pca_predictions_val[1],
                                matrix_functional_pca_dense_predictions_val[1]
                               ]

    h11_predictions_test = np.c_[lin_reg_h11_predictions_test,
                                lasso_h11_predictions_test,
                                el_net_h11_predictions_test,
                                ridge_h11_predictions_test,
                                lin_svr_h11_predictions_test,
                                svr_rbf_h11_predictions_test,
                                xgb_h11_predictions_test,
                                xgbrf_h11_predictions_test,
                                seq_h11_predictions_test,
                                matrix_functional_predictions_test[0],
                                matrix_functional_pca_predictions_test[0],
                                matrix_functional_pca_dense_predictions_test[0]
                               ]
    h21_predictions_test = np.c_[lin_reg_h11_predictions_test,
                                lasso_h11_predictions_test,
                                el_net_h11_predictions_test,
                                ridge_h11_predictions_test,
                                lin_svr_h11_predictions_test,
                                svr_rbf_h11_predictions_test,
                                xgb_h11_predictions_test,
                                xgbrf_h11_predictions_test,
                                seq_h11_predictions_test,
                                matrix_functional_predictions_test[1],
                                matrix_functional_pca_predictions_test[1],
                                matrix_functional_pca_dense_predictions_test[1]
                               ]

    print('End of the predictions!', flush=True)


    # Train a second level meta estimator, such as
    #
    # - LinearRegression,
    # 
    # - SVR (Gaussian kernel),
    # 
    # - SVR (polynomial kernel),
    #
    # - Random Forest
    # 

    # Apply scaler to data
    scalers = [('StandardScaler', StandardScaler()),
               ('MinMaxScaler',   MinMaxScaler()),
               ('RobustScaler',   RobustScaler()),
               ('MaxAbsScaler',   MaxAbsScaler())
              ]

    # Define a new cross-validation strategy
    meta_cv  = KFold(n_splits=8, shuffle=True, random_state=RAND)

    # Define a LinearRegressor
    search_params_lin_reg = {'fit_intercept': [ True, False ],
                             'normalize':     [ True, False ]
                            }
    rounding_lin_reg = np.rint

    meta_lin_reg_h11 = GridSearchCV(estimator=LinearRegression(),
                                    param_grid=search_params_lin_reg,
                                    scoring=make_scorer(accuracy_score,
                                                        greater_is_better=True,
                                                        rounding=rounding_lin_reg),
                                    cv=meta_cv,
                                    n_jobs=-1,
                                    verbose=0
                                   )
    meta_lin_reg_h21 = GridSearchCV(estimator=LinearRegression(),
                                    param_grid=search_params_lin_reg,
                                    scoring=make_scorer(accuracy_score,
                                                        greater_is_better=True,
                                                        rounding=rounding_lin_reg),
                                    cv=meta_cv,
                                    n_jobs=-1,
                                    verbose=0
                                   )

    print('\nFitting the LinearRegressor for h_11...', flush=True)
    meta_fit(meta_lin_reg_h11,
             scalers,
             h11_predictions_val,
             df_labels_val['h11'].values,
             h11_predictions_test,
             df_labels_test['h11'].values,
             rounding=rounding_lin_reg)

    print('\nFitting the LinearRegressor for h_21...', flush=True)
    meta_fit(meta_lin_reg_h21,
             scalers,
             h21_predictions_val,
             df_labels_val['h21'].values,
             h21_predictions_test,
             df_labels_test['h21'].values,
             rounding=rounding_lin_reg)

    # Define a SVR with Gaussian kernel
    search_params_svr_rbf = {'C':         Real(1e-4, 1e4,
                                               base=10,
                                               prior='log-uniform'),
                             'gamma':     Real(1e-6, 1e2,
                                               base=10,
                                               prior='log-uniform'),
                             'epsilon':   Real(1e-5, 1e1,
                                               base=10,
                                               prior='log-uniform'),
                             'shrinking': Integer(False, True)
                            }
    rounding_svr_rbf = np.rint

    meta_svr_rbf_h11 = BayesSearchCV(estimator=SVR(kernel='rbf', max_iter=15000),
                                     search_spaces=search_params_svr_rbf,
                                     n_iter=n_iter,
                                     scoring=make_scorer(accuracy_score,
                                                         greater_is_better=True,
                                                         rounding=rounding_svr_rbf),
                                     cv=meta_cv,
                                     n_jobs=-1,
                                     verbose=0
                                    )
    meta_svr_rbf_h21 = BayesSearchCV(estimator=SVR(kernel='rbf', max_iter=15000),
                                     search_spaces=search_params_svr_rbf,
                                     n_iter=n_iter,
                                     scoring=make_scorer(accuracy_score, 
                                                         greater_is_better=True,
                                                         rounding=rounding_svr_rbf),
                                     cv=meta_cv,
                                     n_jobs=-1,
                                     verbose=0
                                    )

    print('\n\nFitting the SVR(kernel=\'rbf\') for h_11...', flush=True)
    meta_fit(meta_svr_rbf_h11,
             scalers,
             h11_predictions_val,
             df_labels_val['h11'].values,
             h11_predictions_test,
             df_labels_test['h11'].values,
             rounding=rounding_svr_rbf)

    print('\nFitting the SVR(kernel=\'rbf\') for h_21...', flush=True)
    meta_fit(meta_svr_rbf_h21,
             scalers,
             h21_predictions_val,
             df_labels_val['h21'].values,
             h21_predictions_test,
             df_labels_test['h21'].values,
             rounding=rounding_svr_rbf)

    # Define a SVR with polynomial kernel
    search_params_svr_poly = {'degree':    Integer(2, 4),
                              'coef0':     Real(0.0, 10.0,
                                                prior='uniform'),
                              'C':         Real(1e-4, 1e5,
                                                base=10,
                                                prior='log-uniform'),
                              'gamma':     Real(1e-6, 1e2,
                                                base=10,
                                                prior='log-uniform'),
                              'epsilon':   Real(1e-5, 1e1,
                                                base=10,
                                                prior='log-uniform'),
                              'shrinking': Integer(False, True)
                             }
    rounding_svr_poly = np.rint

    meta_svr_poly_h11 = BayesSearchCV(estimator=SVR(kernel='poly', max_iter=15000),
                                      search_spaces=search_params_svr_poly,
                                      n_iter=n_iter,
                                      scoring=make_scorer(accuracy_score,
                                                          greater_is_better=True,
                                                          rounding=rounding_svr_poly),
                                      cv=meta_cv,
                                      n_jobs=-1,
                                      verbose=0
                                     )
    meta_svr_poly_h21 = BayesSearchCV(estimator=SVR(kernel='poly', max_iter=15000),
                                      search_spaces=search_params_svr_poly,
                                      n_iter=n_iter,
                                      scoring=make_scorer(accuracy_score,
                                                          greater_is_better=True,
                                                          rounding=rounding_svr_poly),
                                      cv=meta_cv,
                                      n_jobs=-1,
                                      verbose=0
                                     )

    print('\n\nFitting the SVR(kernel=\'poly\') for h_11...', flush=True)
    meta_fit(meta_svr_poly_h11,
             scalers,
             h11_predictions_val,
             df_labels_val['h11'].values,
             h11_predictions_test,
             df_labels_test['h11'].values,
             rounding=rounding_svr_poly)

    print('\nFitting the SVR(kernel=\'poly\') for h_21...', flush=True)
    meta_fit(meta_svr_poly_h21,
             scalers,
             h21_predictions_val,
             df_labels_val['h21'].values,
             h21_predictions_test,
             df_labels_test['h21'].values,
             rounding=rounding_svr_poly)

    # Define a RandomForest
    search_params_rnd_for = {'n_estimators':      Integer(2, 75,
                                                          prior='uniform'),
                             'criterion':         Categorical(['friedman_mse',
                                                               'mae'
                                                              ]),
                             'min_samples_split': Integer(2, 10,
                                                          prior='uniform'),
                             'min_samples_leaf':  Integer(1, 50,
                                                          prior='uniform'),
                             'max_depth':         Integer(2, 20,
                                                          prior='uniform')
                            }
    rounding_rnd_for = np.rint

    meta_rnd_for_h11 = BayesSearchCV(estimator=RandomForestRegressor(n_jobs=-1,
                                                            random_state=RAND),
                                      search_spaces=search_params_rnd_for,
                                      n_iter=int(n_iter/3),
                                      scoring=make_scorer(accuracy_score,
                                                          greater_is_better=True,
                                                          rounding=rounding_rnd_for),
                                      cv=meta_cv,
                                      n_jobs=1,
                                      verbose=0
                                     )
    meta_rnd_for_h21 = BayesSearchCV(estimator=RandomForestRegressor(n_jobs=-1,
                                                            random_state=RAND),
                                      search_spaces=search_params_rnd_for,
                                      n_iter=int(n_iter/3),
                                      scoring=make_scorer(accuracy_score,
                                                          greater_is_better=True,
                                                          rounding=rounding_rnd_for),
                                      cv=meta_cv,
                                      n_jobs=1,
                                      verbose=0
                                     )

    print('\n\nFitting the RandomForest for h_11...', flush=True)
    meta_fit(meta_rnd_for_h11,
             scalers,
             h11_predictions_val,
             df_labels_val['h11'].values,
             h11_predictions_test,
             df_labels_test['h11'].values,
             rounding=rounding_rnd_for)

    print('\nFitting the RandomForest for h_21...', flush=True)
    meta_fit(meta_rnd_for_h21,
             scalers,
             h21_predictions_val,
             df_labels_val['h21'].values,
             h21_predictions_test,
             df_labels_test['h21'].values,
             rounding=rounding_rnd_for)
