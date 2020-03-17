# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# SEQUENTIAL_MATRIX:
#
#   Compute a Sequential model for the data using only the matrix.
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

from os                          import path, mkdir
from joblib                      import load, dump
from sklearn.pipeline            import Pipeline
from sklearn.preprocessing       import MinMaxScaler, StandardScaler
from sklearn.model_selection     import train_test_split
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

def compute(df_name, rounding=np.floor, seed=42):

    # Print banner
    print('\n----- KERAS SEQUENTIAL MODEL -----', flush=True)

    # Set random seed
    RAND = seed
    np.random.seed(RAND)
    tf.random.set_seed(RAND)


    # Load datasets
    DB_PROD_NAME = df_name + '_matrix'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5.xz')
    if path.isfile(DB_PROD_PATH):
        df_matrix = load(DB_PROD_PATH)
    else:
        print('Cannot read the matrix database!', flush=True)
        
    DB_PROD_NAME = df_name + '_labels_production'
    DB_PROD_PATH = path.join(ROOT_DIR, DB_PROD_NAME + '.h5')
    if path.isfile(DB_PROD_PATH):
        df_labels = pd.read_hdf(DB_PROD_PATH)
    else:
        print('Cannot read the labels database!', flush=True)
        
    h11_labels   = ExtractTensor(flatten=True).fit_transform(df_labels['h11'])
    h21_labels   = ExtractTensor(flatten=True).fit_transform(df_labels['h21'])
    euler_labels = ExtractTensor(flatten=True).fit_transform(df_labels['euler'])
        
    # Split into training and test sets
    df_matrix_train, df_matrix_test, \
    h11_labels_train, h11_labels_test, \
    h21_labels_train, h21_labels_test, \
    euler_labels_train, euler_labels_test = train_test_split(df_matrix,
                                                             h11_labels,
                                                             h21_labels,
                                                             euler_labels,
                                                             test_size=0.1,
                                                             random_state=RAND,
                                                             shuffle=True
                                                            )

    # # Apply scaler to the input
    # scal = Pipeline([('normalization', MinMaxScaler()),
    #                  ('standardization', StandardScaler())])

    # df_matrix_train = scal.fit_transform(df_matrix_train)
    # df_matrix_test  = scal.transform(df_matrix_test)

    # Reshape the matrix
    df_matrix_nn_train = df_matrix_train.reshape(-1,12,15,1)
    df_matrix_nn_test  = df_matrix_test.reshape(-1,12,15,1)

    # Split into training and validation set
    df_matrix_nn_train, df_matrix_nn_val, \
    h11_labels_nn_train, h11_labels_nn_val, \
    h21_labels_nn_train, h21_labels_nn_val, \
    euler_labels_nn_train, euler_labels_nn_val = train_test_split(df_matrix_nn_train,
                                                                  h11_labels_train,
                                                                  h21_labels_train,
                                                                  euler_labels_train,
                                                                  test_size=1/9,
                                                                  random_state=RAND
                                                                 )

    # Cast into Tensorflow vectors
    df_matrix_nn_train    = K.cast(df_matrix_nn_train, dtype='float64')
    df_matrix_nn_val      = K.cast(df_matrix_nn_val, dtype='float64')
    df_matrix_nn_test     = K.cast(df_matrix_nn_test, dtype='float64')

    h11_labels_nn_train   = K.cast(h11_labels_nn_train, dtype='float64')
    h21_labels_nn_train   = K.cast(h21_labels_nn_train, dtype='float64')
    euler_labels_nn_train = K.cast(euler_labels_nn_train, dtype='float64')

    h11_labels_nn_val     = K.cast(h11_labels_nn_val, dtype='float64')
    h21_labels_nn_val     = K.cast(h21_labels_nn_val, dtype='float64')
    euler_labels_nn_val   = K.cast(euler_labels_nn_val, dtype='float64')

    # Compile and fit the model for h_11:
    cnn_h11_params = {'conv2d_layers':       [ 80, 40, 20 ],
                      'conv_activation':     'relu',
                      'dense_activation':    'relu',
                      'kernel_size':         5,
                      'max_pool':            None,
                      'dropout':             0.2,
                      'batch_normalization': True,
                      'dense':               800,
                      'out_activation':      True,
                      'l1_regularization':   1e-5,
                      'l2_regularization':   0.0
                     }

    model_h11_cnn = build_cnn_sequential(**cnn_h11_params)

    # Plot the model
    model_h11_cnn_plot   = path.join(IMG_PATH, 'cnn_matrix_sequential_h11.png')
    plot_model(model_h11_cnn,
               to_file=model_h11_cnn_plot,
               rankdir='LR',
               show_layer_names=False,
               dpi=300)

    # Show a summary
    print('\nSequential model for h_11...', flush=True)
    model_h11_cnn.summary()

    # Compile the model
    model_h11_cnn.compile(optimizer=Adam(learning_rate=0.001),
                          loss=keras.losses.MeanSquaredError(),
                          metrics=[keras.metrics.MeanSquaredError()]
                         )

    # Create callbacks
    callbacks_h11 = [EarlyStopping(monitor='val_mean_squared_error',
                                   patience=50,
                                   verbose=1),
                     ReduceLROnPlateau(monitor='val_mean_squared_error',
                                       factor=0.3,
                                       patience=30,
                                       verbose=1),
                     ModelCheckpoint(path.join(MOD_PATH,
                                               'cnn_matrix_sequential_h11.h5'),
                                     monitor='val_mean_squared_error',
                                     save_best_only=True,
                                     verbose=1)
                    ]

    # Fit the model
    model_h11_history = model_h11_cnn.fit(x=df_matrix_nn_train,
                                          y=h11_labels_nn_train,
                                          batch_size=16,
                                          epochs=1000,
                                          verbose=1,
                                          callbacks=callbacks_h11,
                                          validation_data=(df_matrix_nn_val,
                                                           h11_labels_nn_val)
                                         )


    # Plot validation loss and mean squared error
    print('Plotting validation loss and metric...', flush=True)
    fig, plot = plt.subplots(1, 2, figsize=(12,5))
    fig.tight_layout()

    series_plot(plot[0],
                model_h11_history.history['loss'], 
                title='Validation and Training Loss',
                xlabel='Epoch',
                legend='Training loss')
    series_plot(plot[0],
                model_h11_history.history['val_loss'],
                title='Validation and Training Loss',
                xlabel='Epoch',
                legend='Validation loss')

    series_plot(plot[1],
                np.sqrt(model_h11_history.history['mean_squared_error']),
                title='Validation and Training Metric',
                xlabel='Epoch',
                legend='Training RMSE')
    series_plot(plot[1],
                np.sqrt(model_h11_history.history['val_mean_squared_error']),
                title='Validation and Training Metric',
                xlabel='Epoch',
                legend='Validation RMSE')

    save_fig('cnn_matrix_sequential_h11')
    # plt.show()
    plt.close(fig)

    # Evaluate the model on the test set:

    if path.isfile(path.join(MOD_PATH, 'cnn_matrix_sequential_h11.h5')):
        model_h11_cnn = load_model(path.join(MOD_PATH,
                                             'cnn_matrix_sequential_h11.h5'))
    else:
        print('\nCannot load best model!', flush=True)

    print('    Accuracy (rint) on the training set: {:.3f}%'.format(\
                    accuracy_score(h11_labels_nn_train,
                                   model_h11_cnn.predict(df_matrix_nn_train),
                                   rounding=np.rint)*100))
    print('    Accuracy (rint) on the validation set: {:.3f}%'.format(\
                    accuracy_score(h11_labels_nn_val,
                                   model_h11_cnn.predict(df_matrix_nn_val),
                                   rounding=np.rint)*100))
    prediction_score(model_h11_cnn,
                     df_matrix_nn_test,
                     h11_labels_test,
                     rounding=np.rint)

    print('Plotting error distribution...', flush=True)
    fig, plot = plt.subplots(figsize=(6, 5))
    fig.tight_layout()

    count_plot(plot,
               error_diff(h11_labels_test,
                          model_h11_cnn.predict(df_matrix_nn_test).reshape(-1,),
                          rounding=np.rint),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{11}$')

    save_fig('cnn_matrix_sequential_h11_error_eng')
    # plt.show()
    plt.close(fig)

    # Compile and fit the model for h_21:
    cnn_h21_params = {'conv2d_layers':       [ 180, 150, 150, 100, 50, 20 ],
                      'conv_activation':     'relu',
                      'dense_activation':    'relu',
                      'kernel_size':         6,
                      'max_pool':            None,
                      'dropout':             0.4,
                      'batch_normalization': True,
                      'dense':               30,
                      'out_activation':      True,
                      'l1_regularization':   0.0,
                      'l2_regularization':   0.0
                     }

    model_h21_cnn = build_cnn_sequential(**cnn_h21_params)

    # Plot the model
    model_h21_cnn_plot   = path.join(IMG_PATH, 'cnn_matrix_sequential_h21.png')
    plot_model(model_h21_cnn,
               to_file=model_h21_cnn_plot,
               rankdir='LR',
               show_layer_names=False,
               dpi=300)

    # Show a summary
    print('\nSequential model for h_21...', flush=True)
    model_h21_cnn.summary()

    # Compile the model
    model_h21_cnn.compile(optimizer=Adam(learning_rate=0.001),
                          loss=keras.losses.MeanSquaredError(),
                          metrics=[keras.metrics.MeanSquaredError()]
                         )

    # Create callbacks
    callbacks_h21 = [EarlyStopping(monitor='val_mean_squared_error',
                                   patience=50,
                                   verbose=1),
                     ReduceLROnPlateau(monitor='val_mean_squared_error',
                                       factor=0.3,
                                       patience=30,
                                       verbose=1),
                     ModelCheckpoint(path.join(MOD_PATH,
                                               'cnn_matrix_sequential_h21.h5'),
                                     monitor='val_mean_squared_error',
                                     save_best_only=True,
                                     verbose=1)
                    ]

    # Fit the model
    model_h21_history = model_h21_cnn.fit(x=df_matrix_nn_train,
                                          y=h21_labels_nn_train,
                                          batch_size=32,
                                          epochs=1000,
                                          verbose=1,
                                          callbacks=callbacks_h21,
                                          validation_data=(df_matrix_nn_val,
                                                           h21_labels_nn_val)
                                         )


    # Plot validation loss and metrics:
    print('Plotting validation loss and metric...', flush=True)
    fig, plot = plt.subplots(1, 2, figsize=(12,5))
    fig.tight_layout()

    series_plot(plot[0],
                model_h21_history.history['loss'],
                title='Validation and Training Loss',
                xlabel='Epoch',
                legend='Training loss')
    series_plot(plot[0],
                model_h21_history.history['val_loss'],
                title='Validation and Training Loss',
                xlabel='Epoch',
                legend='Validation loss')

    series_plot(plot[1],
                np.sqrt(model_h21_history.history['mean_squared_error']),
                title='Validation and Training Metric',
                xlabel='Epoch',
                legend='Training RMSE')
    series_plot(plot[1],
                np.sqrt(model_h21_history.history['val_mean_squared_error']),
                title='Validation and Training Metric',
                xlabel='Epoch',
                legend='Validation RMSE')

    save_fig('cnn_matrix_sequential_h21')
    # plt.show()
    plt.close(fig)


    # Evaluate the model on the test set:
    if path.isfile(path.join(MOD_PATH, 'cnn_matrix_sequential_h21.h5')):
        model_h21_cnn = load_model(path.join(MOD_PATH,
                                             'cnn_matrix_sequential_h21.h5')) 
    else:
        print('\nCannot load the best model!', flush=True)
        
    print('    Accuracy (rint) on the training set: {:.3f}%'.format(\
                    accuracy_score(h21_labels_nn_train,
                                   model_h21_cnn.predict(df_matrix_nn_train),
                                   rounding=np.rint)*100))
    print('    Accuracy (rint) on the validation set: {:.3f}%'.format(\
                    accuracy_score(h21_labels_nn_val,
                                   model_h21_cnn.predict(df_matrix_nn_val),
                                   rounding=np.rint)*100))
    prediction_score(model_h21_cnn,
                     df_matrix_nn_test,
                     h21_labels_test,
                     rounding=np.rint)

    print('Plotting error distribution...', flush=True)
    fig, plot = plt.subplots(figsize=(6, 5))
    fig.tight_layout()

    count_plot(plot,
               error_diff(h21_labels_test,
                          model_h21_cnn.predict(df_matrix_nn_test).reshape(-1,),
                          rounding=np.rint),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{21}$')

    save_fig('cnn_matrix_sequential_h21_error_eng')
    # plt.show()
    plt.close(fig)

    # Clear Tensorflow session
    K.clear_session()
