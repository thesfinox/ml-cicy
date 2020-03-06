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

from os                            import path
from tensorflow                    import keras
from joblib                        import load, dump
from sklearn.preprocessing         import StandardScaler, \
                                          MinMaxScaler, \
                                          RobustScaler, \
                                          MaxAbsScaler
from sklearn.metrics               import make_scorer
from sklearn.model_selection       import train_test_split, \
                                          cross_val_predict, \
                                          KFold, \
                                          GridSearchCV, \
                                          KFold
from sklearn.linear_model          import LinearRegression, \
                                          Lasso, \
                                          Ridge, \
                                          ElasticNet
from sklearn.svm                   import SVR, LinearSVR
from sklearn.ensemble              import RandomForestRegressor, \
                                          GradientBoostingRegressor
from xgboost                       import XGBRegressor, XGBRFRegressor
from skopt                         import BayesSearchCV
from skopt.space                   import Categorical, Real, Integer
from tensorflow.keras              import backend as K
from tensorflow.keras.layers       import Conv1D, \
                                          Dense, \
                                          Flatten, \
                                          BatchNormalization, \
                                          Dropout, \
                                          LeakyReLU, \
                                          Activation, \
                                          Input
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers   import Adam
from tensorflow.keras.callbacks    import ReduceLROnPlateau, \
                                          ModelCheckpoint, \
                                          EarlyStopping
from tensorflow.keras.utils        import plot_model
from toolset.libutilities          import *
from toolset.libplot               import *
from toolset.libnn                 import *

# Set working directories
ROOT_DIR = '.' # root directory
MOD_DIR  = 'models' # models directory
MOD_PATH = path.join(ROOT_DIR, MOD_DIR)
if path.isdir(MOD_PATH) is False:
    mkdir(MOD_PATH)

def meta_nn(layers,
            activation='relu',
            kernel_size=3,
            dropout=0.2,
            batch_normalization=True,
            dense=10,
            l1_regularization=0.0,
            l2_regularization=0.0):
    
    # kernel regularizations
    reg = l1_l2(l1=l1_regularization, l2=l2_regularization)

    # buld the model
    model = Sequential()
    model.add(Input(shape=(12,1)))
    
    for n in range(len(layers)):
        model.add(Conv1D(filters=layers[n],
                         kernel_size=kernel_size,
                         padding='valid',
                         kernel_regularizer=reg))
        if activation == 'relu':
            model.add(Activation('relu'))
        else:
            model.add(LeakyReLU(alpha=activation))

    if batch_normalization:
        model.add(BatchNormalization())

    if dropout > 0.0:
        model.add(Dropout(rate=dropout))

    model.add(Flatten())

    if dense > 0:
        model.add(Dense(units=dense, kernel_regularizer=reg))
        if activation == 'relu':
            model.add(Activation('relu'))
        else:
            model.add(LeakyReLU(alpha=activation))

        if batch_normalization:
            model.add(BatchNormalization())

    mode.add(Dense(1))

    return model
    
def meta_fit(estimator,         # Fit the same estimator with scalers
             scalers,
             X_validation,
             y_validation,
             X_test,
             y_test,
             rounding=np.rint):

    print('  --> Fit without scalers:')
    estimator.fit(X_validation, y_validation)
    gridcv_score(estimator, rounding=rounding)
    prediction_score(estimator, X_test, y_test, rounding=rounding)
    
    for scaler in scalers:
        print('  --> Fit with {}:'.format(scaler[0]))
        X_validation = scaler[1].fit_transform(X_validation)
        X_test       = scaler[1].transform(X_test)
        estimator.fit(X_validation, y_validation)
        gridcv_score(estimator, rounding=rounding)
        prediction_score(estimator, X_test, y_test, rounding=rounding)

def compute(df_name, n_iter=30, seed=42):

    # Print banner
    print('\n----- STACKING -----')

    # Set random seed
    RAND = seed
    np.random.seed(RAND)
    tf.random.set_seed(RAND)

    # Set cross-validation strategy
    cv = KFold(n_splits=6, shuffle=True, random_state=RAND)

    # Load models
    print('\nDefining models:')
    print('    Define LinearRegression...')
    search_params = {'fit_intercept': [ True, False ],
                     'normalize':     [ True, False ]
                    }

    lin_reg_h11 = GridSearchCV(estimator=LinearRegression(),
                               param_grid=search_params,
                               scoring=make_scorer(accuracy_score,
                                                   greater_is_better=True,
                                                   rounding=np.floor),
                               cv=cv,
                               n_jobs=-1,
                               verbose=0
                              )
    lin_reg_h21 = GridSearchCV(estimator=LinearRegression(),
                               param_grid=search_params,
                               scoring=make_scorer(accuracy_score,
                                                   greater_is_better=True,
                                                   rounding=np.floor),
                               cv=cv,
                               n_jobs=-1,
                               verbose=0
                              )

    print('    Define Lasso...')
    search_params = {'alpha':         Real(1e-6, 1e2,
                                           base=10,
                                           prior='log-uniform'),
                     'fit_intercept': Integer(False, True),
                     'normalize':     Integer(False, True),
                     'positive':      Integer(False, True)
                    }

    lasso_h11 = BayesSearchCV(estimator=Lasso(max_iter=15000,
                                              random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )
    lasso_h21 = BayesSearchCV(estimator=Lasso(max_iter=15000,
                                              random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )

    print('    Define ElasticNet...')
    search_params = {'alpha':         Real(1e-6, 1e2,
                                           base=10,
                                           prior='log-uniform'),
                     'l1_ratio':      Real(0.0, 1.0,
                                           prior='uniform'),
                     'fit_intercept': Integer(False, True),
                     'normalize':     Integer(False, True),
                     'positive':      Integer(False, True)
                    }

    el_net_h11 = BayesSearchCV(estimator=ElasticNet(max_iter=15000,
                                                    random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )
    el_net_h21 = BayesSearchCV(estimator=ElasticNet(max_iter=15000,
                                                    random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )

    print('    Define Ridge...')
    search_params = {'alpha':         Real(1e-6, 1e2,
                                           base=10,
                                           prior='log-uniform'),
                     'fit_intercept': Integer(False, True),
                     'normalize':     Integer(False, True)
                    }

    ridge_h11 = BayesSearchCV(estimator=Ridge(max_iter=15000,
                                              random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )
    ridge_h21 = BayesSearchCV(estimator=Ridge(max_iter=15000,
                                              random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )

    print('    Define LinearSVR...')
    search_params = {'C':                 Real(1e-4, 1e4,
                                               base=10,
                                               prior='log-uniform'),
                     'intercept_scaling': Real(1e-2, 1e2,
                                               base=10,
                                               prior='log-uniform'),
                     'fit_intercept':     Integer(False, True),
                     'loss':              Categorical(\
                                            ['epsilon_insensitive',
                                             'squared_epsilon_insensitive'])
                    }

    lin_svr_h11 = BayesSearchCV(estimator=LinearSVR(max_iter=15000,
                                                    random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )
    lin_svr_h21 = BayesSearchCV(estimator=LinearSVR(max_iter=15000,
                                                    random_state=RAND),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )

    print('    Define SVR...')
    search_params = {'C':         Real(1e-4, 1e4,
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

    svr_rbf_h11 = BayesSearchCV(estimator=SVR(kernel='rbf',
                                              max_iter=15000),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.rint),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )
    svr_rbf_h21 = BayesSearchCV(estimator=SVR(kernel='rbf',
                                              max_iter=15000),
                              search_spaces=search_params,
                              n_iter=n_iter,
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.rint),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )

    print('    Define GradientBoostingRegressor...')
    search_params = {'learning_rate':     Real(1e-5, 1e-1,
                                               base=10,
                                               prior='log-uniform'),
                     'n_estimators':      Integer(100, 300,
                                                  prior='uniform'),
                     'subsample':         Real(0.6, 1.0,
                                               prior='uniform'),
                     'min_samples_split': Integer(2, 10,
                                                  prior='uniform'),
                     'max_depth':         Integer(2, 20,
                                          prior='uniform')
                    }

    grd_boost_h11 = BayesSearchCV(estimator=GradientBoostingRegressor(\
                                                    random_state=RAND),
                              search_spaces=search_params,
                              n_iter=int(n_iter/5),
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )
    grd_boost_h21 = BayesSearchCV(estimator=GradientBoostingRegressor(\
                                                    random_state=RAND),
                              search_spaces=search_params,
                              n_iter=int(n_iter/5),
                              scoring=make_scorer(accuracy_score,
                                                  greater_is_better=True,
                                                  rounding=np.floor),
                              cv=cv,
                              n_jobs=-1,
                              verbose=0
                             )

    print('    Define RandomForestRegressor...')
    search_params = {'n_estimators':      Integer(2, 75,
                                                  prior='uniform'),
                     'min_samples_split': Integer(2, 10,
                                                  prior='uniform'),
                     'min_samples_leaf':  Integer(1, 50,
                                                  prior='uniform'),
                     'max_depth':         Integer(2, 20,
                                                  prior='uniform')
                    }
    rnd_for_h11 = BayesSearchCV(estimator=RandomForestRegressor(n_jobs=-1,
                                                            random_state=RAND),
                                      search_spaces=search_params,
                                      n_iter=int(n_iter/5),
                                      scoring=make_scorer(accuracy_score,
                                                          greater_is_better=True,
                                                          rounding=np.floor),
                                      cv=cv,
                                      n_jobs=1,
                                      verbose=0
                                     )
    rnd_for_h21 = BayesSearchCV(estimator=RandomForestRegressor(n_jobs=-1,
                                                            random_state=RAND),
                                      search_spaces=search_params,
                                      n_iter=int(n_iter/5),
                                      scoring=make_scorer(accuracy_score,
                                                          greater_is_better=True,
                                                          rounding=np.floor),
                                      cv=cv,
                                      n_jobs=1,
                                      verbose=0
                                     )

    print('    Define the Sequential Keras model...')
    seq_h11     = load_model(path.join(MOD_PATH,
                                       'cnn_matrix_sequential_h11.h5'))
    seq_h21     = load_model(path.join(MOD_PATH, 'cnn_matrix_sequential_h21.h5'))

    print('    Define the Functional Keras model...')
    matrix_functional           = load_model(path.join(MOD_PATH,
                                                       'cnn_functional.h5'))
    print('    Define the Functional Keras model with PCA...')
    matrix_functional_pca       = load_model(path.join(MOD_PATH,
                                                       'cnn_functional_pca.h5'))
    print('    Define the Functional Keras model with PCA and Dense layers...')
    matrix_functional_pca_dense = load_model(path.join(MOD_PATH,
                                                       'cnn_functional_pca_dense.h5'))


    # Load database
    DB_PROD_NAME = df_name + '_matrix'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_matrix = load(DB_PROD_PATH)
    else:
        print('Cannot read the matrix database!')
        
    DB_PROD_NAME = df_name + '_eng_h11'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_eng_h11 = load(DB_PROD_PATH)
    else:
        print('Cannot read the eng_h11 database!')
        
    DB_PROD_NAME = df_name + '_eng_h21'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_eng_h21 = load(DB_PROD_PATH)
    else:
        print('Cannot read the eng_h21 database!')
        
    DB_PROD_NAME = df_name + '_labels_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df_labels = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the labels database!')

    # Split into training and test
    df_matrix_train, df_matrix_test, \
    df_eng_h11_train, df_eng_h11_test, \
    df_eng_h21_train, df_eng_h21_test, \
    df_labels_train, df_labels_test = train_test_split(df_matrix, 
                                                       df_eng_h11,
                                                       df_eng_h21,
                                                       df_labels,
                                                       test_size=0.1,
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
                                                      test_size=3.0/9.0,
                                                      shuffle=True,
                                                      random_state=RAND)


    # Train the algorithms:
    print('\nTraining models:')
    print('    Fitting the LinearRegressor...')
    lin_reg_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    lin_reg_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    lin_reg_h11 = lin_reg_h11.best_estimator_
    lin_reg_h21 = lin_reg_h21.best_estimator_

    print('    Fitting the Lasso...')
    lasso_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    lasso_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    lasso_h11 = lasso_h11.best_estimator_
    lasso_h21 = lasso_h21.best_estimator_

    print('    Fitting the ElasticNet...')
    el_net_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    el_net_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    el_net_h11 = el_net_h11.best_estimator_
    el_net_h21 = el_net_h21.best_estimator_

    print('    Fitting the Ridge...')
    ridge_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    ridge_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    ridge_h11 = ridge_h11.best_estimator_
    ridge_h21 = ridge_h21.best_estimator_

    print('    Fitting the LinearSVR...')
    lin_svr_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    lin_svr_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    lin_svr_h11 = lin_svr_h11.best_estimator_
    lin_svr_h21 = lin_svr_h21.best_estimator_

    print('    Fitting the SVR...')
    svr_rbf_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    svr_rbf_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    svr_rbf_h11 = svr_rbf_h11.best_estimator_
    svr_rbf_h21 = svr_rbf_h21.best_estimator_

    print('    Fitting the GradientBoostingRegressor...')
    grd_boost_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    grd_boost_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    grd_boost_h11 = grd_boost_h11.best_estimator_
    grd_boost_h21 = grd_boost_h21.best_estimator_

    print('    Fitting the RandomForestRegressor...')
    rnd_for_h11.fit(df_eng_h11_train, df_labels_train['h11'].values)
    rnd_for_h21.fit(df_eng_h21_train, df_labels_train['h21'].values)
    rnd_for_h11 = rnd_for_h11.best_estimator_
    rnd_for_h21 = rnd_for_h21.best_estimator_

    print('    Fitting the Sequential Keras models...')
    seq_h11.fit(x=K.cast(df_matrix_train.reshape(-1,12,15,1), dtype='float64'),
                y=K.cast(df_labels_train['h11'], dtype='float64'),
                validation_split=0.2,
                batch_size=32,
                epochs=300,
                verbose=1
               )
    seq_h21.fit(x=K.cast(df_matrix_train.reshape(-1,12,15,1), dtype='float64'),
                y=K.cast(df_labels_train['h21'], dtype='float64'),
                validation_split=0.2,
                batch_size=32,
                epochs=300,
                verbose=1
               )
    seq_h11 = load_model(path.join(MOD_PATH, 'cnn_matrix_sequential_h11.h5'))
    seq_h21 = load_model(path.join(MOD_PATH, 'cnn_matrix_sequential_h21.h5'))

    K.clear_session()

    print('    Fitting the Functional Conv2D Keras models...')
    matrix_functional.fit(x=[K.cast(df_eng_h21_train[:,0].reshape(-1,1),
                                    dtype='float64'),
                             K.cast(df_eng_h21_train[:,1:13].reshape(-1,12,1),
                                    dtype='float64'),
                             K.cast(df_eng_h21_train[:,13:28].reshape(-1,15,1),
                                    dtype='float64'),
                             K.cast(df_matrix_train.reshape(-1,12,15,1),
                                    dtype='float64')],
                          y=[K.cast(df_labels_train['h11'], dtype='float64'),
                             K.cast(df_labels_train['h21'], dtype='float64')],
                          validation_split=0.2,
                          batch_size=32,
                          epochs=300,
                          verbose=1
                         )
    matrix_functional = load_model(path.join(MOD_PATH, 'cnn_functional.h5'))

    K.clear_session()

    print('    Fitting the Functional Conv1D Keras models...')
    matrix_functional_pca.fit(x=[K.cast(df_eng_h21_train[:,0].reshape(-1,1),
                                        dtype='float64'),
                                 K.cast(df_eng_h21_train[:,1:13].reshape(-1,12,1),
                                        dtype='float64'),
                                 K.cast(df_eng_h21_train[:,13:28].reshape(-1,15,1),
                                        dtype='float64'),
                                 K.cast(df_eng_h21_train[:,28:].reshape(-1,81,1),
                                        dtype='float64')],
                              y=[K.cast(df_labels_train['h11'], dtype='float64'),
                                 K.cast(df_labels_train['h21'], dtype='float64')],
                              validation_split=0.2,
                              batch_size=32,
                              epochs=300,
                              verbose=1
                             )
    matrix_functional_pca = load_model(path.join(MOD_PATH, 'cnn_functional_pca.h5'))

    K.clear_session()

    print('    Fitting the Functional Dense Keras models...')
    matrix_functional_pca_dense.fit(x=[K.cast(df_eng_h21_train[:,0].reshape(-1,1), 
                                              dtype='float64'),
                                       K.cast(df_eng_h21_train[:,1:13],
                                              dtype='float64'),
                                       K.cast(df_eng_h21_train[:,13:28],
                                              dtype='float64'),
                                       K.cast(df_eng_h21_train[:,28:],
                                              dtype='float64')],
                                    y=[K.cast(df_labels_train['h11'],
                                              dtype='float64'),
                                       K.cast(df_labels_train['h21'],
                                              dtype='float64')],
                                    validation_split=0.2,
                                    batch_size=32,
                                    epochs=300,
                                    verbose=1
                                   )
    matrix_functional_pca_dense = load_model(path.join(MOD_PATH,
                                                'cnn_functional_pca_dense.h5'))

    K.clear_session()

    print('End of the training procedure!')


    # Use the trained models to build the second level predictions:
    print('\nPredictions of the LinearRegressor...')
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

    print('Predictions of the Lasso...')
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

    print('Predictions of the ElasticNet...')
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

    print('Predictions of the Ridge...')
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

    print('Predictions of the LinearSVR...')
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

    print('Predictions of the SVR...')
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

    print('Predictions of the GradientBoostingRegressor...')
    grd_boost_h11_predictions_val = grd_boost_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   grd_boost_h11_predictions_val,
                                   rounding=np.floor)*100))
    grd_boost_h21_predictions_val = grd_boost_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   grd_boost_h21_predictions_val,
                                   rounding=np.floor)*100))

    grd_boost_h11_predictions_test = grd_boost_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   grd_boost_h11_predictions_test,
                                   rounding=np.floor)*100))
    grd_boost_h21_predictions_test = grd_boost_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   grd_boost_h21_predictions_test,
                                   rounding=np.floor)*100))

    print('Predictions of the RandomForestRegressor...')
    rnd_for_h11_predictions_val = rnd_for_h11.predict(df_eng_h11_val)
    print('    Accuracy of the validation predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h11'].values,
                                   rnd_for_h11_predictions_val,
                                   rounding=np.floor)*100))
    rnd_for_h21_predictions_val = rnd_for_h21.predict(df_eng_h21_val)
    print('    Accuracy of the validation predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_val['h21'].values,
                                   rnd_for_h21_predictions_val,
                                   rounding=np.floor)*100))

    rnd_for_h11_predictions_test = rnd_for_h11.predict(df_eng_h11_test)
    print('    Accuracy of the test predictions for h_11: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h11'].values,
                                   rnd_for_h11_predictions_test,
                                   rounding=np.floor)*100))
    rnd_for_h21_predictions_test = rnd_for_h21.predict(df_eng_h21_test)
    print('    Accuracy of the test predictions for h_21: {:.3f}%'.format(\
                    accuracy_score(df_labels_test['h21'].values,
                                   rnd_for_h21_predictions_test,
                                   rounding=np.floor)*100))


    print('Predictions of the Sequential Keras models...')
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

    print('Predictions of the Functional Conv2D Keras models...')
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

    print('Predictions of the Functional Conv1D Keras models...')
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


    print('Predictions of the Functional Dense Keras models...')
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

    print('Concatenatig prediction vectors...')
    h11_predictions_val = np.c_[lin_reg_h11_predictions_val,
                                lasso_h11_predictions_val,
                                el_net_h11_predictions_val,
                                ridge_h11_predictions_val,
                                lin_svr_h11_predictions_val,
                                svr_rbf_h11_predictions_val,
                                grd_boost_h11_predictions_val,
                                rnd_for_h11_predictions_val,
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
                                grd_boost_h11_predictions_val,
                                rnd_for_h11_predictions_val,
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
                                grd_boost_h11_predictions_test,
                                rnd_for_h11_predictions_test,
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
                                grd_boost_h11_predictions_test,
                                rnd_for_h11_predictions_test,
                                seq_h11_predictions_test,
                                matrix_functional_predictions_test[1],
                                matrix_functional_pca_predictions_test[1],
                                matrix_functional_pca_dense_predictions_test[1]
                               ]

    print('End of the predictions!')


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
    meta_cv  = KFold(n_splits=3, shuffle=True, random_state=RAND)

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

    print('\nFitting the LinearRegressor for h_11...')
    meta_fit(meta_lin_reg_h11,
             scalers,
             h11_predictions_val,
             df_labels_val['h11'].values,
             h11_predictions_test,
             df_labels_test['h11'].values,
             rounding=rounding_lin_reg)

    print('\nFitting the LinearRegressor for h_21...')
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

    print('\n\nFitting the SVR(kernel=\'rbf\') for h_11...')
    meta_fit(meta_svr_rbf_h11,
             scalers,
             h11_predictions_val,
             df_labels_val['h11'].values,
             h11_predictions_test,
             df_labels_test['h11'].values,
             rounding=rounding_svr_rbf)

    print('\nFitting the SVR(kernel=\'rbf\') for h_21...')
    meta_fit(meta_svr_rbf_h21,
             scalers,
             h21_predictions_val,
             df_labels_val['h21'].values,
             h21_predictions_test,
             df_labels_test['h21'].values,
             rounding=rounding_svr_rbf)

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
                                      n_iter=int(n_iter/5),
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
                                      n_iter=int(n_iter/5),
                                      scoring=make_scorer(accuracy_score,
                                                          greater_is_better=True,
                                                          rounding=rounding_rnd_for),
                                      cv=meta_cv,
                                      n_jobs=1,
                                      verbose=0
                                     )

    print('\n\nFitting the RandomForest for h_11...')
    meta_fit(meta_rnd_for_h11,
             scalers,
             h11_predictions_val,
             df_labels_val['h11'].values,
             h11_predictions_test,
             df_labels_test['h11'].values,
             rounding=rounding_rnd_for)

    print('\nFitting the RandomForest for h_21...')
    meta_fit(meta_rnd_for_h21,
             scalers,
             h21_predictions_val,
             df_labels_val['h21'].values,
             h21_predictions_test,
             df_labels_test['h21'].values,
             rounding=rounding_rnd_for)

    # Use a neural network as meta learner for h_11
    meta_nn_h11 = meta_nn(layers=[10, 5, 1],
                          activation=0.3,
                          kernel_size=3,
                          dropout=0.2,
                          batch_normalization=True,
                          dense=0,
                          l1_regularization=0.0,
                          l2_regularization=0.0)                          
    meta_nn_h11.summary()
    meta_nn_h11.compile(optimizer=Adam(learning_rate=0.001),
                        loss=keras.losses.MeanSquaredError(),
                        metrics=[keras.metrics.MeanSquaredError()]
                       )
    callbacks_h11 = [EarlyStopping(monitor='val_mean_squared_error',
                                   patience=50,
                                   verbose=0),
                     ReduceLROnPlateau(monitor='val_mean_squared_error',
                                       factor=0.3,
                                       patience=30,
                                       verbose=0),
                     ModelCheckpoint(path.join(MOD_PATH,
                                               'cnn_meta_matrix_sequential_h11.h5'),
                                     monitor='val_mean_squared_error',
                                     save_best_only=True,
                                    verbose=0)
                    ]

    print('\nFitting Neural Network for h_11...')
    meta_nn_h11.fit(x=K.cast(h11_predictions_val, dtype='float64'),
                    y=K.cast(h11_labels_nn_train, dtype='float64'),
                    batch_size=32,
                    epochs=1000,
                    verbose=1,
                    callbacks=callbacks_h11,
                    validation_split=0.1
                   )
    if path.isfile(path.join(MOD_PATH, 'cnn_meta_matrix_sequential_h11.h5')):
        meta_nn_h11  = load_model(path.join(MOD_PATH,
                                  'cnn_meta_matrix_sequential_h11.h5'))
    else:
        print('\nCannot load best model!')

    prediction_score(meta_nn_h11,
                     h11_predictions_test,
                     h11_labels_nn_test,
                     rounding=np.rint)

    # Use a neural network as meta learner for h_21
    meta_nn_h21 = meta_nn(layers=[10, 5, 1],
                          activation=0.3,
                          kernel_size=3,
                          dropout=0.3,
                          batch_normalization=True,
                          dense=0,
                          l1_regularization=0.0,
                          l2_regularization=0.0)                          
    meta_nn_h21.summary()
    meta_nn_h21.compile(optimizer=Adam(learning_rate=0.001),
                        loss=keras.losses.MeanSquaredError(),
                        metrics=[keras.metrics.MeanSquaredError()]
                       )
    callbacks_h21 = [EarlyStopping(monitor='val_mean_squared_error',
                                   patience=50,
                                   verbose=0),
                     ReduceLROnPlateau(monitor='val_mean_squared_error',
                                       factor=0.3,
                                       patience=30,
                                       verbose=0),
                     ModelCheckpoint(path.join(MOD_PATH,
                                               'cnn_meta_matrix_sequential_h21.h5'),
                                     monitor='val_mean_squared_error',
                                     save_best_only=True,
                                     verbose=0)
                    ]

    print('\nFitting Neural Network for h_21...')
    meta_nn_h21.fit(x=K.cast(h21_predictions_val, dtype='float64'),
                    y=K.cast(h21_labels_nn_train, dtype='float64'),
                    batch_size=32,
                    epochs=1000,
                    verbose=1,
                    callbacks=callbacks_h21,
                    validation_split=0.1
                   )
    if path.isfile(path.join(MOD_PATH, 'cnn_meta_matrix_sequential_h21.h5')):
        meta_nn_h21  = load_model(path.join(MOD_PATH,
                                  'cnn_meta_matrix_sequential_h21.h5'))
    else:
        print('\nCannot load best model!')

    prediction_score(meta_nn_h21,
                     h21_predictions_test,
                     h21_labels_nn_test,
                     rounding=np.rint)
