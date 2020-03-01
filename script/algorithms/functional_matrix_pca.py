# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# FUNCTIONAL_MATRIX_PCA:
#
#   Compute a Functional model for the data using the PCA of the matrix.
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
# from sklearn.preprocessing       import StandardScaler
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
    print('\n----- KERAS FUNCTIONAL MODEL (PCA) -----')

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

    # Apply StandardScaler to the input
    # std_scal = StandardScaler()

    # df_matrix_train  = std_scal.fit_transform(df_matrix_train)
    # df_matrix_test   = std_scal.transform(df_matrix_test)

    # Reshape the matrix
    df_matrix_nn_train = df_matrix_train.reshape(-1,12,15,1)
    df_matrix_nn_test  = df_matrix_test.reshape(-1,12,15,1)

    # Split into training and validation set
    df_matrix_nn_train, df_matrix_nn_val, \
    df_num_cp_nn_train, df_num_cp_nn_val, \
    df_dim_cp_nn_train, df_dim_cp_nn_val, \
    df_eng_h11_nn_train, df_eng_h11_nn_val, \
    df_eng_h21_nn_train, df_eng_h21_nn_val, \
    h11_labels_nn_train, h11_labels_nn_val, \
    h21_labels_nn_train, h21_labels_nn_val, \
    euler_labels_nn_train, euler_labels_nn_val = train_test_split(df_matrix_nn_train,
                                                                  df_num_cp_train,
                                                                  df_dim_cp_train,
                                                                  df_eng_h11_train,
                                                                  df_eng_h21_train,
                                                                  h11_labels_train,
                                                                  h21_labels_train,
                                                                  euler_labels_train,
                                                                  test_size=1/9,
                                                                  random_state=RAND
                                                                 )

    # Cast into Tensorflow vectors
    df_matrix_nn_train    = K.cast(df_matrix_nn_train, dtype='float64')
    df_num_cp_nn_train    = K.cast(df_num_cp_nn_train, dtype='float64')
    df_dim_cp_nn_train    = K.cast(df_dim_cp_nn_train, dtype='float64')
    df_eng_h11_nn_train   = K.cast(df_eng_h11_nn_train, dtype='float64')
    df_eng_h21_nn_train   = K.cast(df_eng_h21_nn_train, dtype='float64')

    df_matrix_nn_val      = K.cast(df_matrix_nn_val, dtype='float64')
    df_num_cp_nn_val      = K.cast(df_num_cp_nn_val, dtype='float64')
    df_dim_cp_nn_val      = K.cast(df_dim_cp_nn_val, dtype='float64')
    df_eng_h11_nn_val     = K.cast(df_eng_h11_nn_val, dtype='float64')
    df_eng_h21_nn_val     = K.cast(df_eng_h21_nn_val, dtype='float64')

    df_matrix_nn_test     = K.cast(df_matrix_nn_test, dtype='float64')
    df_num_cp_nn_test     = K.cast(df_num_cp_test, dtype='float64')
    df_dim_cp_nn_test     = K.cast(df_dim_cp_test, dtype='float64')
    df_eng_h11_nn_test    = K.cast(df_eng_h11_test, dtype='float64')
    df_eng_h21_nn_test    = K.cast(df_eng_h21_test, dtype='float64')

    h11_labels_nn_train   = K.cast(h11_labels_nn_train, dtype='float64')
    h21_labels_nn_train   = K.cast(h21_labels_nn_train, dtype='float64')
    euler_labels_nn_train = K.cast(euler_labels_nn_train, dtype='float64')

    h11_labels_nn_val     = K.cast(h11_labels_nn_val, dtype='float64')
    h21_labels_nn_val     = K.cast(h21_labels_nn_val, dtype='float64')
    euler_labels_nn_val   = K.cast(euler_labels_nn_val, dtype='float64')

    num_cp_input_train     = df_eng_h21_nn_train[:,0]
    num_cp_input_val       = df_eng_h21_nn_val[:,0]
    num_cp_input_test      = df_eng_h21_nn_test[:,0]

    dim_cp_input_train     = df_eng_h21_nn_train[:,1:13]
    dim_cp_input_val       = df_eng_h21_nn_val[:,1:13]
    dim_cp_input_test      = df_eng_h21_nn_test[:,1:13]

    dim_h0_amb_input_train = df_eng_h21_nn_train[:,13:28]
    dim_h0_amb_input_val   = df_eng_h21_nn_val[:,13:28]
    dim_h0_amb_input_test  = df_eng_h21_nn_test[:,13:28]

    dim_cp_input_train_conv = K.reshape(df_eng_h21_nn_train[:,1:13], (-1,12,1))
    dim_cp_input_val_conv   = K.reshape(df_eng_h21_nn_val[:,1:13], (-1,12,1))
    dim_cp_input_test_conv  = K.reshape(df_eng_h21_nn_test[:,1:13], (-1,12,1))

    dim_h0_amb_input_train_conv = K.reshape(df_eng_h21_nn_train[:,13:28], (-1,15,1))
    dim_h0_amb_input_val_conv   = K.reshape(df_eng_h21_nn_val[:,13:28], (-1,15,1))
    dim_h0_amb_input_test_conv  = K.reshape(df_eng_h21_nn_test[:,13:28], (-1,15,1))

    matrix_pca_input_train = df_eng_h21_nn_train[:,28:]
    matrix_pca_input_val   = df_eng_h21_nn_val[:,28:]
    matrix_pca_input_test  = df_eng_h21_nn_test[:,28:]

    matrix_pca_input_train_conv = K.reshape(df_eng_h21_nn_train[:,28:], (-1,81,1))
    matrix_pca_input_val_conv   = K.reshape(df_eng_h21_nn_val[:,28:], (-1,81,1))
    matrix_pca_input_test_conv  = K.reshape(df_eng_h21_nn_test[:,28:], (-1,81,1))

    # Build, compile and fit the model:
    model_cnn = build_conv_model_2(num_cp_dense_layers=([],[]),
                                   dim_cp_conv1d_layers=([],[15,30,30,15]),
                                   dim_h0_amb_conv1d_layers=([20],[20,20,10]),
                                   matrix_conv1d_layers=([200, 200, 100],
                                                         [300, 300, 200, 200, 100]),
                                   activation='relu',
                                   kernel_size=3,
                                   max_pool=[0,0],
                                   dropout=[0.5,0.4],
                                   batch_normalization=True,
                                   dense=[10],
                                   out_activation=True,
                                   l1_regularization=(0.0,0.0),
                                   l2_regularization=(0.0,0.0)
                                  )

    # Plot the model
    model_cnn_plot = path.join(IMG_PATH, 'cnn_functional_pca.png')
    plot_model(model_cnn,
               to_file=model_cnn_plot,
               rankdir='TF',
               show_layer_names=True,
               dpi=300)

    # Show a summary
    model_cnn.summary()

    # Compile the model
    model_cnn.compile(optimizer=Adam(learning_rate=0.001),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()]
                     )

    # Create callbacks
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=80,
                               verbose=0),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.3,
                                   patience=50,
                                   verbose=0),
                 ModelCheckpoint(path.join(MOD_PATH,
                                           'cnn_functional_pca.h5'),
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=0)
                ]

    # Fit the model
    model_cnn_history = model_cnn.fit(x=[num_cp_input_train,
                                         dim_cp_input_train_conv,
                                         dim_h0_amb_input_train_conv,
                                         matrix_pca_input_train_conv],
                                      y=[h11_labels_nn_train,
                                         h21_labels_nn_train],
                                      batch_size=32,
                                      epochs=1000,
                                      verbose=0,
                                      callbacks=callbacks,
                                      validation_data=([num_cp_input_val,
                                                        dim_cp_input_val_conv,
                                                        dim_h0_amb_input_val_conv,
                                                        matrix_pca_input_val_conv],
                                                       [h11_labels_nn_val,
                                                        h21_labels_nn_val]
                                                      )
                                     )


    # Evaluate the model on the test set:
    if path.isfile(path.join(MOD_PATH, 'cnn_functional_pca.h5')):
        model_cnn = load_model(path.join(MOD_PATH,
                                         'cnn_functional_pca.h5'))
    else:
        print('\nCannot load the best model!')
        
    print('\n    Accuracy (rint) on the training set for h_11: {:.3f}%'.format(\
            accuracy_score(h11_labels_nn_train,
                           model_cnn.predict([num_cp_input_train,
                                              dim_cp_input_train_conv,
                                              dim_h0_amb_input_train_conv,
                                              matrix_pca_input_train_conv])[0].\
                                                      reshape(-1,),
                           rounding=np.rint)*100))
    print('    Accuracy (rint) on the validation set for h_11: {:.3f}%'.format(\
            accuracy_score(h11_labels_nn_val,
                           model_cnn.predict([num_cp_input_val,
                                              dim_cp_input_val_conv,
                                              dim_h0_amb_input_val_conv,
                                              matrix_pca_input_val_conv])[0].\
                                                      reshape(-1,),
                           rounding=np.rint)*100))
    print('    Accuracy (rint) of the predictions for h_11: {:.3f}%'.format(\
            accuracy_score(h11_labels_test,
                           model_cnn.predict([num_cp_input_test,
                                              dim_cp_input_test_conv,
                                              dim_h0_amb_input_test_conv,
                                              matrix_pca_input_test_conv])[0].\
                                                      reshape(-1,),
                           rounding=np.rint)*100))

    print('\n    Accuracy (rint) on the training set for h_21: {:.3f}%'.format(\
            accuracy_score(h21_labels_nn_train,
                           model_cnn.predict([num_cp_input_train,
                                              dim_cp_input_train_conv,
                                              dim_h0_amb_input_train_conv,
                                              matrix_pca_input_train_conv])[1].\
                                                      reshape(-1,), \
                           rounding=np.rint)*100))
    print('    Accuracy (rint) on the validation set for h_21: {:.3f}%'.format(\
            accuracy_score(h21_labels_nn_val,
                           model_cnn.predict([num_cp_input_val,
                                              dim_cp_input_val_conv,
                                              dim_h0_amb_input_val_conv,
                                              matrix_pca_input_val_conv])[1].\
                                                      reshape(-1,),
                           rounding=np.rint)*100))
    print('    Accuracy (rint) of the predictions for h_21: {:.3f}%'.format(\
            accuracy_score(h21_labels_test,
                           model_cnn.predict([num_cp_input_test,
                                              dim_cp_input_test_conv,
                                              dim_h0_amb_input_test_conv,
                                              matrix_pca_input_test_conv])[1].\
                                                      reshape(-1,),
                           rounding=np.rint)*100))

    print('Plotting the error distribution...')
    fig, plot = plt.subplots(figsize=(6, 5))
    fig.tight_layout()

    count_plot(plot,
               error_diff(h11_labels_test,
                          model_cnn.predict([num_cp_input_test,
                                             dim_cp_input_test_conv,
                                             dim_h0_amb_input_test_conv,
                                             matrix_pca_input_test_conv])[0].\
                                                     reshape(-1,),
                          rounding=np.rint),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{11}$')
    count_plot(plot,
               error_diff(h21_labels_test,
                          model_cnn.predict([num_cp_input_test,
                                             dim_cp_input_test_conv,
                                             dim_h0_amb_input_test_conv,
                                             matrix_pca_input_test_conv])[1].\
                                                     reshape(-1,),
                          rounding=np.rint),
               title='Error distribution on the test set',
               xlabel='Difference from real value',
               legend='$h_{21}$')

    save_fig('cnn_functional_pca_error_eng')
    # plt.show()
    plt.close(fig)
