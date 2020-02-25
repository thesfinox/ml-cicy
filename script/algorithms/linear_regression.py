# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# LINEAR_REGRESSION:
#
#   Compute the LinearRegression model for the data.
#
# AUTHOR: Riccardo Finotello
#

import numpy   as np
import sklearn as sk
import pandas  as pd

assert np.__version__ >  '1.16'   # to avoid issues with pytables
assert sk.__version__ >= '0.22.1' # for the recent implementation

from os                      import path
from joblib                  import load, dump
# from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import make_scorer
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model    import LinearRegression
from toolset.libutilities    import *
from toolset.libplot         import *

# Set working directories
ROOT_DIR = '.' # root directory
MOD_DIR  = 'models' # models directory
MOD_PATH = path.join(ROOT_DIR, MOD_DIR)
if path.isdir(MOD_PATH) is False:
    mkdir(MOD_PATH)


def compute(df_name, rounding=np.floor, seed=42):

    # Print banner
    print('\n----- LINEAR REGRESSION -----')

    # Set random seed
    RAND = seed
    np.random.seed(RAND)

    # Define the cross-validation strategy
    cv = KFold(n_splits=9, shuffle=True, random_state=RAND)

    # Load datasets
    DB_PROD_NAME = df_name + '_matrix'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_matrix = load(DB_PROD_PATH)
    else:
        print('Cannot read the matrix database!')
        
    DB_PROD_NAME = df_name + '_num_cp'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_num_cp = load(DB_PROD_PATH)
    else:
        print('Cannot read the num_cp database!')
        
    DB_PROD_NAME = df_name + '_dim_cp'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_dim_cp = load(DB_PROD_PATH)
    else:
        print('Cannot read the dim_cp database!')
        
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
        
    h11_labels   = ExtractTensor(flatten=True).fit_transform(df_labels['h11'])
    h21_labels   = ExtractTensor(flatten=True).fit_transform(df_labels['h21'])
    euler_labels = ExtractTensor(flatten=True).fit_transform(df_labels['euler'])
        
    # Split into training and test sets
    df_matrix_train, df_matrix_test, \
    df_num_cp_train, df_num_cp_test, \
    df_dim_cp_train, df_dim_cp_test, \
    df_eng_h11_train, df_eng_h11_test, \
    df_eng_h21_train, df_eng_h21_test, \
    h11_labels_train, h11_labels_test, \
    h21_labels_train, h21_labels_test, \
    euler_labels_train, euler_labels_test = train_test_split(df_matrix,
                                                             df_num_cp,
                                                             df_dim_cp,
                                                             df_eng_h11,
                                                             df_eng_h21,
                                                             h11_labels,
                                                             h21_labels,
                                                             euler_labels,
                                                             test_size=0.1,
                                                             random_state=RAND,
                                                             shuffle=True
                                                            )

    # Reshape the single feature input
    df_num_cp_train = df_num_cp_train.reshape(-1,1)
    df_num_cp_test  = df_num_cp_test.reshape(-1,1)


    # Apply StandardScaler to the input
    # std_scal = StandardScaler()

    # df_matrix_train  = std_scal.fit_transform(df_matrix_train)
    # df_matrix_test   = std_scal.transform(df_matrix_test)

    # df_num_cp_train  = std_scal.fit_transform(df_num_cp_train)
    # df_num_cp_test   = std_scal.transform(df_num_cp_test)

    # df_dim_cp_train  = std_scal.fit_transform(df_dim_cp_train)
    # df_dim_cp_test   = std_scal.transform(df_dim_cp_test)

    # df_eng_h11_train = std_scal.fit_transform(df_eng_h11_train)
    # df_eng_h11_test  = std_scal.transform(df_eng_h11_test)

    # df_eng_h21_train = std_scal.fit_transform(df_eng_h21_train)
    # df_eng_h21_test  = std_scal.transform(df_eng_h21_test)

    # Compute the algorithm
    search_params = {'fit_intercept': [ True, False ],
                     'normalize':     [ True, False ]
                    }

    lin_reg_h11 = GridSearchCV(estimator=LinearRegression(),
                               param_grid=search_params,
                               scoring=make_scorer(accuracy_score,
                                                   greater_is_better=True,
                                                   rounding=rounding),
                               cv=cv,
                               n_jobs=-1,
                               verbose=0
                              )
    lin_reg_h21 = GridSearchCV(estimator=LinearRegression(),
                               param_grid=search_params,
                               scoring=make_scorer(accuracy_score,
                                                   greater_is_better=True,
                                                   rounding=rounding),
                               cv=cv,
                               n_jobs=-1,
                               verbose=0
                              )

    #####################################
    # MATRIX BASELINE                   #
    #####################################
    print('\nFitting the matrix baseline on h_11...')
    lin_reg_h11.fit(df_matrix_train, h11_labels_train)
    gridcv_score(lin_reg_h11, rounding=rounding)
    prediction_score(lin_reg_h11,
                     df_matrix_test,
                     h11_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Fitting the matrix baseline on h_21...')
    lin_reg_h21.fit(df_matrix_train, h21_labels_train)
    gridcv_score(lin_reg_h21, rounding=rounding)
    prediction_score(lin_reg_h21,
                     df_matrix_test,
                     h21_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Plotting error differences...')
    fig, plot = plt.subplots(figsize=(6, 5))
    fig.tight_layout()

    count_plot(plot,
               error_diff(h11_labels_test,
                          lin_reg_h11.best_estimator_.predict(df_matrix_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{11}$')
    count_plot(plot,
               error_diff(h21_labels_test,
                          lin_reg_h21.best_estimator_.predict(df_matrix_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{21}$')

    save_fig('lin_reg_error_matrix')
    # # plt.show()
    plt.close(fig)

    #####################################
    # NUM_CP BASELINE                   #
    #####################################
    print('\nFitting the num_cp baseline on h_11...')
    lin_reg_h11.fit(df_num_cp_train, h11_labels_train)
    gridcv_score(lin_reg_h11, rounding=rounding)
    prediction_score(lin_reg_h11,
                     df_num_cp_test,
                     h11_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Fitting the num_cp baseline on h_21...')
    lin_reg_h21.fit(df_num_cp_train, h21_labels_train)
    gridcv_score(lin_reg_h21, rounding=rounding)
    prediction_score(lin_reg_h21,
                     df_num_cp_test,
                     h21_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Plotting error differences...')
    fig, plot = plt.subplots(figsize=(6, 5))
    fig.tight_layout()

    count_plot(plot,
               error_diff(h11_labels_test,
                          lin_reg_h11.best_estimator_.predict(df_num_cp_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{11}$')
    count_plot(plot,
               error_diff(h21_labels_test,
                          lin_reg_h21.best_estimator_.predict(df_num_cp_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{21}$')

    save_fig('lin_reg_error_num_cp')
    # plt.show()
    plt.close(fig)

    #####################################
    # DIM_CP BASELINE                   #
    #####################################
    print('\nFitting the dim_cp baseline on h_11...')
    lin_reg_h11.fit(df_dim_cp_train, h11_labels_train)
    gridcv_score(lin_reg_h11, rounding=rounding)
    prediction_score(lin_reg_h11,
                     df_dim_cp_test,
                     h11_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Fitting the dim_cp baseline on h_21...')
    lin_reg_h21.fit(df_dim_cp_train, h21_labels_train)
    gridcv_score(lin_reg_h21, rounding=rounding)
    prediction_score(lin_reg_h21,
                     df_dim_cp_test,
                     h21_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Plotting error differences...')
    fig, plot = plt.subplots(figsize=(6, 5))
    fig.tight_layout()

    count_plot(plot,
               error_diff(h11_labels_test,
                          lin_reg_h11.best_estimator_.predict(df_dim_cp_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{11}$')
    count_plot(plot,
               error_diff(h21_labels_test,
                          lin_reg_h21.best_estimator_.predict(df_dim_cp_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{21}$')

    save_fig('lin_reg_error_dim_cp')
    # plt.show()
    plt.close(fig)

    #####################################
    # FEATURE ENGINEERING               #
    #####################################
    print('\nFitting the feature engineered dataset on h_11...')
    lin_reg_h11.fit(df_eng_h11_train, h11_labels_train)
    gridcv_score(lin_reg_h11, rounding=rounding)
    prediction_score(lin_reg_h11,
                     df_eng_h11_test,
                     h11_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Fitting the feature engineered dataset on h_21...')
    lin_reg_h21.fit(df_eng_h21_train, h21_labels_train)
    gridcv_score(lin_reg_h21, rounding=rounding)
    prediction_score(lin_reg_h21,
                     df_eng_h21_test,
                     h21_labels_test,
                     use_best_estimator=True,
                     rounding=rounding)

    print('Plotting error differences...')
    fig, plot = plt.subplots(figsize=(6, 5))
    fig.tight_layout()

    count_plot(plot,
               error_diff(h11_labels_test,
                          lin_reg_h11.best_estimator_.predict(df_eng_h11_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{11}$')
    count_plot(plot,
               error_diff(h21_labels_test,
                          lin_reg_h21.best_estimator_.predict(df_eng_h21_test),
                          rounding=rounding),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{21}$')

    save_fig('lin_reg_error_eng')
    # plt.show()
    plt.close(fig)

    # Saving the feature engineered models
    print('\nSaving models...')
    MOD_FILE = path.join(MOD_PATH, 'lin_reg_h11.joblib.xz')
    dump(lin_reg_h11.best_estimator_, MOD_FILE, compress=('xz',9))
    MOD_FILE = path.join(MOD_PATH, 'lin_reg_h21.joblib.xz')
    dump(lin_reg_h21.best_estimator_, MOD_FILE, compress=('xz',9))
    print('Models saved!')
