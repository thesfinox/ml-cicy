# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# LIBNN:
#
#   Library of definitions and classes to train Neural Network models using
#   Tensorflow-Keras.
#
# AUTHOR: Riccardo Finotello
#

import numpy      as np
import sklearn    as sk
import tensorflow as tf

assert np.__version__  >  '1.16'   # to avoid issues with pytables
assert sk.__version__  >= '0.22.1' # for the recent implementation
assert tf.__version__  >= '2.0.0'  # newest version

from os                            import path
from tensorflow                    import keras
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models       import Model, Sequential, save_model, \
                                          load_model
from tensorflow.keras.layers       import Input, Lambda, \
                                          Dense, Conv1D, Conv2D, concatenate, \
                                          Dropout, MaxPool1D, MaxPool2D, \
                                          Flatten, BatchNormalization, \
                                          Activation, LeakyReLU

gpus = tf.config.experimental.list_physical_devices('GPU')   # set memory growth to avoid taking the entire GPU RAM
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print('\nGPU setup: ',
              '{:d} physical GPUs, '.format(len(gpus)),
              '{:d} logical GPUs.'.format(len(logical_gpus)))
    except RuntimeError as e:
        print(e, flush=True)
else:
    print('\nNo GPUs in the setup!', flush=True)

# Define a Sequential Keras model
def build_cnn_sequential(conv2d_layers,
                         activation='relu',
                         kernel_size=3,
                         max_pool=2,
                         dropout=0.4,
                         batch_normalization=True,
                         dense=10,
                         out_activation=True,
                         l1_regularization=0.0,
                         l2_regularization=0.0
                        ):

    # kernel regularizer
    reg   = l1_l2(l1=l1_regularization, l2=l2_regularization)
    
    model = Sequential() # create the model
    
    model.add(Input(shape=(12,15,1)))              # take just the input shape
    
    for n in range(len(conv2d_layers)):            # add convolutional layers
        model.add(Conv2D(filters=conv2d_layers[n],
                         kernel_size=kernel_size,
                         padding='same',
                         kernel_regularizer=reg
                        )
                 )
        if activation == 'relu':                   # add their activation
            model.add(Activation('relu'))
        else:
            model.add(LeakyReLU(alpha=activation))
            
        if max_pool > 0:                           # reduce feature space if needed
            model.add(MaxPool2D(pool_size=max_pool,
                                padding='same'
                               )
                     )
        
        if batch_normalization:                    # works best with batch_normalization
            model.add(BatchNormalization())
    
    if dropout > 0.0:                              # add dropout to avoid overfitting
        model.add(Dropout(rate=dropout))
        
    model.add(Flatten())                           # add flattener
    
    if dense > 0:                                  # add dense layer (if requested)
        model.add(Dense(units=dense, kernel_regularizer=reg))
        if activation == 'relu':                   # add their activation
            model.add(Activation('relu'))
        else:
            model.add(LeakyReLU(alpha=activation))
        
        if batch_normalization:                    # works best with batch_normalization
            model.add(BatchNormalization())
    
    model.add(Dense(1))                            # output layer
    if out_activation:
        model.add(Activation('relu'))              # force positive output (if needed)
    
    return model

def build_conv_model(num_cp_dense_layers=(None,None),
                     dim_cp_conv1d_layers=(None,None),
                     dim_h0_amb_conv1d_layers=(None,None),
                     matrix_conv2d_layers=(None,None),
                     activation='relu',
                     kernel_size=(3,3),
                     matrix_kernel_size=(5,5),
                     max_pool=(2,2),
                     dropout=(0.4,0.2),
                     batch_normalization=True,
                     dense=[10],
                     out_activation=True,
                     l1_regularization=(0.0,0.0),
                     l2_regularization=(0.0,0.0)
                    ):
    
    # kernel regularizers
    reg_h11 = l1_l2(l1=l1_regularization[0], l2=l2_regularization[0])
    reg_h21 = l1_l2(l1=l1_regularization[1], l2=l2_regularization[1])
    
   # connections for num_cp
    num_cp_layer_in = Input(shape=(1,), name='num_cp')
    num_cp_layer_h11 = Lambda(lambda x: x, name='num_cp_h11')(num_cp_layer_in)
    num_cp_layer_h21 = Lambda(lambda x: x, name='num_cp_h21')(num_cp_layer_in)
    
    for n in range(len(num_cp_dense_layers[0])):
        num_cp_layer_h11 = Dense(num_cp_dense_layers[0][n],
                                 kernel_regularizer=reg_h11
                                )(num_cp_layer_h11)
        if activation == 'relu':
            num_cp_layer_h11 = Activation('relu')(num_cp_layer_h11)
        else:
            num_cp_layer_h11 = LeakyReLU(alpha=activation)(num_cp_layer_h11)
        if batch_normalization:
            num_cp_layer_h11 = BatchNormalization()(num_cp_layer_h11)    
    if dropout[0] > 0:
        num_cp_layer_h11 = Dropout(rate=dropout[0])(num_cp_layer_h11)
        
    for n in range(len(num_cp_dense_layers[1])):
        num_cp_layer_h21 = Dense(num_cp_dense_layers[1][n],
                                 kernel_regularizer=reg_h21
                                )(num_cp_layer_h21)
        if activation == 'relu':
            num_cp_layer_h21 = Activation('relu')(num_cp_layer_h21)
        else:
            num_cp_layer_h21 = LeakyReLU(alpha=activation)(num_cp_layer_h21)
        if batch_normalization:
            num_cp_layer_h21 = BatchNormalization()(num_cp_layer_h21)
    if dropout[1] > 0:
        num_cp_layer_h21 = Dropout(rate=dropout[1])(num_cp_layer_h21)
    
    # connections for dim_cp
    dim_cp_layer_in = Input(shape=(12,1), name='dim_cp')
    dim_cp_layer_h11 = Lambda(lambda x: x, name='dim_cp_h11')(dim_cp_layer_in)
    dim_cp_layer_h21 = Lambda(lambda x: x, name='dim_cp_h21')(dim_cp_layer_in)
    
    for n in range(len(dim_cp_conv1d_layers[0])):
        dim_cp_layer_h11 = Conv1D(dim_cp_conv1d_layers[0][n],
                                  kernel_size=kernel_size[0],
                                  padding='same')(dim_cp_layer_h11)
        if activation == 'relu':
            dim_cp_layer_h11 = Activation('relu')(dim_cp_layer_h11)
        else:
            dim_cp_layer_h11 = LeakyReLU(alpha=activation)(dim_cp_layer_h11)
        if max_pool[0] > 0:
            dim_cp_layer_h11 = MaxPool1D(pool_size=max_pool[0],
                                         padding='same')(dim_cp_layer_h11)
        if batch_normalization:
            dim_cp_layer_h11 = BatchNormalization()(dim_cp_layer_h11)    
    if dropout[0] > 0:
        dim_cp_layer_h11 = Dropout(rate=dropout[0])(dim_cp_layer_h11)
        
    for n in range(len(dim_cp_conv1d_layers[1])):
        dim_cp_layer_h21 = Conv1D(dim_cp_conv1d_layers[1][n],
                                  kernel_size=kernel_size[1],
                                  kernel_regularizer=reg_h21,
                                  padding='same')(dim_cp_layer_h21)
        if activation == 'relu':
            dim_cp_layer_h21 = Activation('relu')(dim_cp_layer_h21)
        else:
            dim_cp_layer_h21 = LeakyReLU(alpha=activation)(dim_cp_layer_h21)
        if max_pool[1] > 0:
            dim_cp_layer_h21 = MaxPool1D(pool_size=max_pool[1],
                                         padding='same')(dim_cp_layer_h21)
        if batch_normalization:
            dim_cp_layer_h21 = BatchNormalization()(dim_cp_layer_h21)
    if dropout[1] > 0:
        dim_cp_layer_h21 = Dropout(rate=dropout[1])(dim_cp_layer_h21)
            
    dim_cp_layer_h11 = Flatten()(dim_cp_layer_h11)
    dim_cp_layer_h21 = Flatten()(dim_cp_layer_h21)
    
    # connections for dim_h0_amb
    dim_h0_layer_in = Input(shape=(15,1), name='dim_h0_amb')
    dim_h0_layer_h11 = Lambda(lambda x: x, name='dim_h0_amb_h11')(dim_h0_layer_in)
    dim_h0_layer_h21 = Lambda(lambda x: x, name='dim_h0_amb_h21')(dim_h0_layer_in)
    
    for n in range(len(dim_h0_amb_conv1d_layers[0])):
        dim_h0_layer_h11 = Conv1D(dim_h0_amb_conv1d_layers[0][n],
                                  kernel_size=kernel_size[0],
                                  kernel_regularizer=reg_h11,
                                  padding='same')(dim_h0_layer_h11)
        if activation == 'relu':
            dim_h0_layer_h11 = Activation('relu')(dim_h0_layer_h11)
        else:
            dim_h0_layer_h11 = LeakyReLU(alpha=activation)(dim_h0_layer_h11)
        if max_pool[0] > 0:
            dim_h0_layer_h11 = MaxPool1D(pool_size=max_pool[0],
                                         padding='same')(dim_h0_layer_h11)
        if batch_normalization:
            dim_h0_layer_h11 = BatchNormalization()(dim_h0_layer_h11)    
    if dropout[0] > 0:
        dim_h0_layer_h11 = Dropout(rate=dropout[0])(dim_h0_layer_h11)
        
    for n in range(len(dim_h0_amb_conv1d_layers[1])):
        dim_h0_layer_h21 = Conv1D(dim_h0_amb_conv1d_layers[1][n],
                                  kernel_size=kernel_size[1],
                                  kernel_regularizer=reg_h21,
                                  padding='same')(dim_h0_layer_h21)
        if activation == 'relu':
            dim_h0_layer_h21 = Activation('relu')(dim_h0_layer_h21)
        else:
            dim_h0_layer_h21 = LeakyReLU(alpha=activation)(dim_h0_layer_h21)
        if max_pool[1] > 0:
            dim_h0_layer_h21 = MaxPool1D(pool_size=max_pool[1],
                                         padding='same')(dim_h0_layer_h21)
        if batch_normalization:
            dim_h0_layer_h21 = BatchNormalization()(dim_h0_layer_h21)
    if dropout[1] > 0:
        dim_h0_layer_h21 = Dropout(rate=dropout[1])(dim_h0_layer_h21)
            
    dim_h0_layer_h11 = Flatten()(dim_h0_layer_h11)
    dim_h0_layer_h21 = Flatten()(dim_h0_layer_h21)
    
    # connections for matrix
    matrix_layer_in = Input(shape=(12,15,1), name='matrix')
    matrix_layer_h11 = Lambda(lambda x: x, name='matrix_h11')(matrix_layer_in)
    matrix_layer_h21 = Lambda(lambda x: x, name='matrix_h21')(matrix_layer_in)
    
    for n in range(len(matrix_conv2d_layers[0])):
        matrix_layer_h11 = Conv2D(matrix_conv2d_layers[0][n],
                                  kernel_size=matrix_kernel_size[0],
                                  kernel_regularizer=reg_h11,
                                  padding='same')(matrix_layer_h11)
        if activation == 'relu':
            matrix_layer_h11 = Activation('relu')(matrix_layer_h11)
        else:
            matrix_layer_h11 = LeakyReLU(alpha=activation)(matrix_layer_h11)
        if max_pool[0] > 0:
            matrix_layer_h11 = MaxPool2D(pool_size=max_pool[0],
                                         padding='same')(matrix_layer_h11)
        if batch_normalization:
            matrix_layer_h11 = BatchNormalization()(matrix_layer_h11)    
    if dropout[0] > 0:
        matrix_layer_h11 = Dropout(rate=dropout[0])(matrix_layer_h11)
        
    for n in range(len(matrix_conv2d_layers[1])):
        matrix_layer_h21 = Conv2D(matrix_conv2d_layers[1][n],
                                  kernel_size=matrix_kernel_size[1],
                                  kernel_regularizer=reg_h21,
                                  padding='same')(matrix_layer_h21)
        if activation == 'relu':
            matrix_layer_h21 = Activation('relu')(matrix_layer_h21)
        else:
            matrix_layer_h21 = LeakyReLU(alpha=activation)(matrix_layer_h21)
        if max_pool[1] > 0:
            matrix_layer_h21 = MaxPool2D(pool_size=max_pool[1],
                                         padding='same')(matrix_layer_h21)
        if batch_normalization:
            matrix_layer_h21 = BatchNormalization()(matrix_layer_h21)
    if dropout[1] > 0:
        matrix_layer_h21 = Dropout(rate=dropout[1])(matrix_layer_h21)
            
    matrix_layer_h11 = Flatten()(matrix_layer_h11)
    matrix_layer_h21 = Flatten()(matrix_layer_h21)
            
    # outputs
    
    h11 = concatenate([num_cp_layer_h11,
                       dim_cp_layer_h11,
                       matrix_layer_h11])
    h11 = Dense(1, name='h_11')(h11)
    
    intermediate = Lambda(lambda x: x)(h11) # return correlation instead of anti-correlation
    intermediate = concatenate([intermediate,
                                dim_cp_layer_h21,
                                dim_h0_layer_h21])
    if len(dense) > 0:
        for n in range(len(dense)):
            intermediate = Dense(dense[n])(intermediate)
            if activation == 'relu':
                intermediate = Activation('relu')(intermediate)
            else:
                intermediate = LeakyReLU(alpha=activation)(intermediate)
    
    h21 = concatenate([num_cp_layer_h21,
                       dim_cp_layer_h21,
                       dim_h0_layer_h21,
                       matrix_layer_h21,
                       intermediate])
    h21 = Dense(1, name='h_21')(h21)
            
    model = Model(inputs=[num_cp_layer_in,
                          dim_cp_layer_in,
                          dim_h0_layer_in,
                          matrix_layer_in],
                  outputs=[h11, h21]
                 )
    
    return model

def build_conv_model_2(num_cp_dense_layers=(None,None),
                       dim_cp_conv1d_layers=(None,None),
                       dim_h0_amb_conv1d_layers=(None,None),
                       matrix_conv1d_layers=(None,None),
                       activation='relu',
                       kernel_size=(3,3),
                       pca_kernel_size=(5,5),
                       max_pool=(2,2),
                       dropout=(0.4,0.2),
                       batch_normalization=True,
                       dense=[1],
                       out_activation=True,
                       l1_regularization=(0.0,0.0),
                       l2_regularization=(0.0,0.0)
                      ):
    
    # kernel regularizers
    reg_h11 = l1_l2(l1=l1_regularization[0], l2=l2_regularization[0])
    reg_h21 = l1_l2(l1=l1_regularization[1], l2=l2_regularization[1])
    
   # connections for num_cp
    num_cp_layer_in = Input(shape=(1,), name='num_cp')
    num_cp_layer_h11 = Lambda(lambda x: x, name='num_cp_h11')(num_cp_layer_in)
    num_cp_layer_h21 = Lambda(lambda x: x, name='num_cp_h21')(num_cp_layer_in)
    
    for n in range(len(num_cp_dense_layers[0])):
        num_cp_layer_h11 = Dense(num_cp_dense_layers[0][n],
                                 kernel_regularizer=reg_h11
                                )(num_cp_layer_h11)
        if activation == 'relu':
            num_cp_layer_h11 = Activation('relu')(num_cp_layer_h11)
        else:
            num_cp_layer_h11 = LeakyReLU(alpha=activation)(num_cp_layer_h11)
        if batch_normalization:
            num_cp_layer_h11 = BatchNormalization()(num_cp_layer_h11)    
    if dropout[0] > 0:
        num_cp_layer_h11 = Dropout(rate=dropout[0])(num_cp_layer_h11)
        
    for n in range(len(num_cp_dense_layers[1])):
        num_cp_layer_h21 = Dense(num_cp_dense_layers[1][n],
                                 kernel_regularizer=reg_h21
                                )(num_cp_layer_h21)
        if activation == 'relu':
            num_cp_layer_h21 = Activation('relu')(num_cp_layer_h21)
        else:
            num_cp_layer_h21 = LeakyReLU(alpha=activation)(num_cp_layer_h21)
        if batch_normalization:
            num_cp_layer_h21 = BatchNormalization()(num_cp_layer_h21)
    if dropout[1] > 0:
        num_cp_layer_h21 = Dropout(rate=dropout[1])(num_cp_layer_h21)
    
    # connections for dim_cp
    dim_cp_layer_in = Input(shape=(12,1), name='dim_cp')
    dim_cp_layer_h11 = Lambda(lambda x: x, name='dim_cp_h11')(dim_cp_layer_in)
    dim_cp_layer_h21 = Lambda(lambda x: x, name='dim_cp_h21')(dim_cp_layer_in)
    
    for n in range(len(dim_cp_conv1d_layers[0])):
        dim_cp_layer_h11 = Conv1D(dim_cp_conv1d_layers[0][n],
                                  kernel_size=kernel_size[0],
                                  kernel_regularizer=reg_h11,
                                  padding='same')(dim_cp_layer_h11)
        if activation == 'relu':
            dim_cp_layer_h11 = Activation('relu')(dim_cp_layer_h11)
        else:
            dim_cp_layer_h11 = LeakyReLU(alpha=activation)(dim_cp_layer_h11)
        if max_pool[0] > 0:
            dim_cp_layer_h11 = MaxPool1D(pool_size=max_pool[0],
                                         padding='same')(dim_cp_layer_h11)
        if batch_normalization:
            dim_cp_layer_h11 = BatchNormalization()(dim_cp_layer_h11)    
    if dropout[0] > 0:
        dim_cp_layer_h11 = Dropout(rate=dropout[0])(dim_cp_layer_h11)
        
    for n in range(len(dim_cp_conv1d_layers[1])):
        dim_cp_layer_h21 = Conv1D(dim_cp_conv1d_layers[1][n],
                                  kernel_size=kernel_size[1],
                                  kernel_regularizer=reg_h21,
                                  padding='same')(dim_cp_layer_h21)
        if activation == 'relu':
            dim_cp_layer_h21 = Activation('relu')(dim_cp_layer_h21)
        else:
            dim_cp_layer_h21 = LeakyReLU(alpha=activation)(dim_cp_layer_h21)
        if max_pool[1] > 0:
            dim_cp_layer_h21 = MaxPool1D(pool_size=max_pool[1],
                                         padding='same')(dim_cp_layer_h21)
        if batch_normalization:
            dim_cp_layer_h21 = BatchNormalization()(dim_cp_layer_h21)
    if dropout[1] > 0:
        dim_cp_layer_h21 = Dropout(rate=dropout[1])(dim_cp_layer_h21)
            
    dim_cp_layer_h11 = Flatten()(dim_cp_layer_h11)
    dim_cp_layer_h21 = Flatten()(dim_cp_layer_h21)
    
    # connections for dim_h0_amb
    dim_h0_layer_in = Input(shape=(15,1), name='dim_h0_amb')
    dim_h0_layer_h11 = Lambda(lambda x: x, name='dim_h0_amb_h11')(dim_h0_layer_in)
    dim_h0_layer_h21 = Lambda(lambda x: x, name='dim_h0_amb_h21')(dim_h0_layer_in)
    
    for n in range(len(dim_h0_amb_conv1d_layers[0])):
        dim_h0_layer_h11 = Conv1D(dim_h0_amb_conv1d_layers[0][n],
                                  kernel_size=kernel_size[0],
                                  kernel_regularizer=reg_h11,
                                  padding='same')(dim_h0_layer_h11)
        if activation == 'relu':
            dim_h0_layer_h11 = Activation('relu')(dim_h0_layer_h11)
        else:
            dim_h0_layer_h11 = LeakyReLU(alpha=activation)(dim_h0_layer_h11)
        if max_pool[0] > 0:
            dim_h0_layer_h11 = MaxPool1D(pool_size=max_pool[0],
                                         padding='same')(dim_h0_layer_h11)
        if batch_normalization:
            dim_h0_layer_h11 = BatchNormalization()(dim_h0_layer_h11)    
    if dropout[0] > 0:
        dim_h0_layer_h11 = Dropout(rate=dropout[0])(dim_h0_layer_h11)
        
    for n in range(len(dim_h0_amb_conv1d_layers[1])):
        dim_h0_layer_h21 = Conv1D(dim_h0_amb_conv1d_layers[1][n],
                                  kernel_size=kernel_size[1],
                                  kernel_regularizer=reg_h21,
                                  padding='same')(dim_h0_layer_h21)
        if activation == 'relu':
            dim_h0_layer_h21 = Activation('relu')(dim_h0_layer_h21)
        else:
            dim_h0_layer_h21 = LeakyReLU(alpha=activation)(dim_h0_layer_h21)
        if max_pool[1] > 0:
            dim_h0_layer_h21 = MaxPool1D(pool_size=max_pool[1],
                                         padding='same')(dim_h0_layer_h21)
        if batch_normalization:
            dim_h0_layer_h21 = BatchNormalization()(dim_h0_layer_h21)
    if dropout[1] > 0:
        dim_h0_layer_h21 = Dropout(rate=dropout[1])(dim_h0_layer_h21)
            
    dim_h0_layer_h11 = Flatten()(dim_h0_layer_h11)
    dim_h0_layer_h21 = Flatten()(dim_h0_layer_h21)
    
    # connections for matrix PCA
    matrix_layer_in = Input(shape=(81,1), name='matrix')
    matrix_layer_h11 = Lambda(lambda x: x, name='matrix_h11')(matrix_layer_in)
    matrix_layer_h21 = Lambda(lambda x: x, name='matrix_h21')(matrix_layer_in)
    
    for n in range(len(matrix_conv1d_layers[0])):
        matrix_layer_h11 = Conv1D(matrix_conv1d_layers[0][n],
                                  kernel_size=pca_kernel_size[0],
                                  kernel_regularizer=reg_h11,
                                  padding='same')(matrix_layer_h11)
        if activation == 'relu':
            matrix_layer_h11 = Activation('relu')(matrix_layer_h11)
        else:
            matrix_layer_h11 = LeakyReLU(alpha=activation)(matrix_layer_h11)
        if max_pool[0] > 0:
            matrix_layer_h11 = MaxPool1D(pool_size=max_pool[0],
                                         padding='same')(matrix_layer_h11)
        if batch_normalization:
            matrix_layer_h11 = BatchNormalization()(matrix_layer_h11)    
    if dropout[0] > 0:
        matrix_layer_h11 = Dropout(rate=dropout[0])(matrix_layer_h11)
        
    for n in range(len(matrix_conv1d_layers[1])):
        matrix_layer_h21 = Conv1D(matrix_conv1d_layers[1][n],
                                  kernel_size=pca_kernel_size[1],
                                  kernel_regularizer=reg_h21,
                                  padding='same')(matrix_layer_h21)
        if activation == 'relu':
            matrix_layer_h21 = Activation('relu')(matrix_layer_h21)
        else:
            matrix_layer_h21 = LeakyReLU(alpha=activation)(matrix_layer_h21)
        if max_pool[1] > 0:
            matrix_layer_h21 = MaxPool1D(pool_size=max_pool[1],
                                         padding='same')(matrix_layer_h21)
        if batch_normalization:
            matrix_layer_h21 = BatchNormalization()(matrix_layer_h21)
    if dropout[1] > 0:
        matrix_layer_h21 = Dropout(rate=dropout[1])(matrix_layer_h21)
            
    matrix_layer_h11 = Flatten()(matrix_layer_h11)
    matrix_layer_h21 = Flatten()(matrix_layer_h21)
            
    # outputs
    
    h11 = concatenate([num_cp_layer_h11, dim_cp_layer_h11, matrix_layer_h11])
    h11 = Dense(1, name='h_11')(h11)
    
    intermediate = Lambda(lambda x: x)(h11) # return correlation instead of anti-correlation
    intermediate = concatenate([intermediate,
                                dim_cp_layer_h21,
                                dim_h0_layer_h21])
    if len(dense) > 0:
        for n in range(len(dense)):
            intermediate = Dense(dense[n])(intermediate)
            if activation == 'relu':
                intermediate = Activation('relu')(intermediate)
            else:
                intermediate = LeakyReLU(alpha=activation)(intermediate)
    
    h21 = concatenate([num_cp_layer_h21,
                       dim_cp_layer_h21,
                       dim_h0_layer_h21,
                       matrix_layer_h21,
                       intermediate])
    h21 = Dense(1, name='h_21')(h21)
            
    model = Model(inputs=[num_cp_layer_in,
                          dim_cp_layer_in,
                          dim_h0_layer_in,
                          matrix_layer_in],
                  outputs=[h11, h21]
                 )
    
    return model

def build_dense_model(num_cp_dense_layers=(None,None),
                      dim_cp_dense_layers=(None,None),
                      dim_h0_amb_dense_layers=(None,None),
                      matrix_dense_layers=(None,None),
                      activation='relu',
                      dropout=(0.4,0.2),
                      batch_normalization=True,
                      dense=[1],
                      out_activation=True,
                      l1_regularization=(0.0,0.0),
                      l2_regularization=(0.0,0.0)
                     ):
    
    # kernel regularizers
    reg_h11 = l1_l2(l1=l1_regularization[0], l2=l2_regularization[0])
    reg_h21 = l1_l2(l1=l1_regularization[1], l2=l2_regularization[1])
    
    # connections for num_cp
    num_cp_layer_in = Input(shape=(1,), name='num_cp')
    num_cp_layer_h11 = Lambda(lambda x: x, name='num_cp_h11')(num_cp_layer_in)
    num_cp_layer_h21 = Lambda(lambda x: x, name='num_cp_h21')(num_cp_layer_in)
    
    for n in range(len(num_cp_dense_layers[0])):
        num_cp_layer_h11 = Dense(num_cp_dense_layers[0][n],
                                 kernel_regularizer=reg_h11
                                )(num_cp_layer_h11)
        if activation == 'relu':
            num_cp_layer_h11 = Activation('relu')(num_cp_layer_h11)
        else:
            num_cp_layer_h11 = LeakyReLU(alpha=activation)(num_cp_layer_h11)
        if batch_normalization:
            num_cp_layer_h11 = BatchNormalization()(num_cp_layer_h11)    
    if dropout[0] > 0:
        num_cp_layer_h11 = Dropout(rate=dropout[0])(num_cp_layer_h11)
        
    for n in range(len(num_cp_dense_layers[1])):
        num_cp_layer_h21 = Dense(num_cp_dense_layers[1][n],
                                 kernel_regularizer=reg_h21
                                )(num_cp_layer_h21)
        if activation == 'relu':
            num_cp_layer_h21 = Activation('relu')(num_cp_layer_h21)
        else:
            num_cp_layer_h21 = LeakyReLU(alpha=activation)(num_cp_layer_h21)
        if batch_normalization:
            num_cp_layer_h21 = BatchNormalization()(num_cp_layer_h21)
    if dropout[1] > 0:
        num_cp_layer_h21 = Dropout(rate=dropout[1])(num_cp_layer_h21)
    
    # connections for dim_cp
    dim_cp_layer_in = Input(shape=(12,), name='dim_cp')
    dim_cp_layer_h11 = Lambda(lambda x: x, name='dim_cp_h11')(dim_cp_layer_in)
    dim_cp_layer_h21 = Lambda(lambda x: x, name='dim_cp_h21')(dim_cp_layer_in)
    
    for n in range(len(dim_cp_dense_layers[0])):
        dim_cp_layer_h11 = Dense(dim_cp_dense_layers[0][n],
                                 kernel_regularizer=reg_h11
                                )(dim_cp_layer_h11)
        if activation == 'relu':
            dim_cp_layer_h11 = Activation('relu')(dim_cp_layer_h11)
        else:
            dim_cp_layer_h11 = LeakyReLU(alpha=activation)(dim_cp_layer_h11)
        if batch_normalization:
            dim_cp_layer_h11 = BatchNormalization()(dim_cp_layer_h11)    
    if dropout[0] > 0:
        dim_cp_layer_h11 = Dropout(rate=dropout[0])(dim_cp_layer_h11)
        
    for n in range(len(dim_cp_dense_layers[1])):
        dim_cp_layer_h21 = Dense(dim_cp_dense_layers[1][n],
                                 kernel_regularizer=reg_h21
                                )(dim_cp_layer_h21)
        if activation == 'relu':
            dim_cp_layer_h21 = Activation('relu')(dim_cp_layer_h21)
        else:
            dim_cp_layer_h21 = LeakyReLU(alpha=activation)(dim_cp_layer_h21)
        if batch_normalization:
            dim_cp_layer_h21 = BatchNormalization()(dim_cp_layer_h21)
    if dropout[1] > 0:
        dim_cp_layer_h21 = Dropout(rate=dropout[1])(dim_cp_layer_h21)
    
    # connections for dim_h0_amb
    dim_h0_layer_in = Input(shape=(15,), name='dim_h0_amb')
    dim_h0_layer_h11 = Lambda(lambda x: x, name='dim_h0_amb_h11')(dim_h0_layer_in)
    dim_h0_layer_h21 = Lambda(lambda x: x, name='dim_h0_amb_h21')(dim_h0_layer_in)
    
    for n in range(len(dim_h0_amb_dense_layers[0])):
        dim_h0_layer_h11 = Dense(dim_h0_amb_dense_layers[0][n],
                                 kernel_regularizer=reg_h11
                                )(dim_h0_layer_h11)
        if activation == 'relu':
            dim_h0_layer_h11 = Activation('relu')(dim_h0_layer_h11)
        else:
            dim_h0_layer_h11 = LeakyReLU(alpha=activation)(dim_h0_layer_h11)
        if batch_normalization:
            dim_h0_layer_h11 = BatchNormalization()(dim_h0_layer_h11)    
    if dropout[0] > 0:
        dim_h0_layer_h11 = Dropout(rate=dropout[0])(dim_h0_layer_h11)
        
    for n in range(len(dim_h0_amb_dense_layers[1])):
        dim_h0_layer_h21 = Dense(dim_h0_amb_dense_layers[1][n],
                                 kernel_regularizer=reg_h21
                                )(dim_h0_layer_h21)
        if activation == 'relu':
            dim_h0_layer_h21 = Activation('relu')(dim_h0_layer_h21)
        else:
            dim_h0_layer_h21 = LeakyReLU(alpha=activation)(dim_h0_layer_h21)
        if batch_normalization:
            dim_h0_layer_h21 = BatchNormalization()(dim_h0_layer_h21)
    if dropout[1] > 0:
        dim_h0_layer_h21 = Dropout(rate=dropout[1])(dim_h0_layer_h21)
    
    # connections for matrix PCA
    matrix_layer_in = Input(shape=(81,), name='matrix')
    matrix_layer_h11 = Lambda(lambda x: x, name='matrix_h11')(matrix_layer_in)
    matrix_layer_h21 = Lambda(lambda x: x, name='matrix_h21')(matrix_layer_in)
    
    for n in range(len(matrix_dense_layers[0])):
        matrix_layer_h11 = Dense(matrix_dense_layers[0][n],
                                 kernel_regularizer=reg_h11
                                )(matrix_layer_h11)
        if activation == 'relu':
            matrix_layer_h11 = Activation('relu')(matrix_layer_h11)
        else:
            matrix_layer_h11 = LeakyReLU(alpha=activation)(matrix_layer_h11)
        if batch_normalization:
            matrix_layer_h11 = BatchNormalization()(matrix_layer_h11)    
    if dropout[0] > 0:
        matrix_layer_h11 = Dropout(rate=dropout[0])(matrix_layer_h11)
        
    for n in range(len(matrix_dense_layers[1])):
        matrix_layer_h21 = Dense(matrix_dense_layers[1][n],
                                 kernel_regularizer=reg_h21
                                )(matrix_layer_h21)
        if activation == 'relu':
            matrix_layer_h21 = Activation('relu')(matrix_layer_h21)
        else:
            matrix_layer_h21 = LeakyReLU(alpha=activation)(matrix_layer_h21)
        if batch_normalization:
            matrix_layer_h21 = BatchNormalization()(matrix_layer_h21)
    if dropout[1] > 0:
        matrix_layer_h21 = Dropout(rate=dropout[1])(matrix_layer_h21)
            
    # outputs
    
    h11 = concatenate([num_cp_layer_h11, dim_cp_layer_h11, matrix_layer_h11])
    h11 = Dense(1)(h11)
    h11 = concatenate([h11, num_cp_layer_h11])
    h11 = Dense(1, name='h_11')(h11)
    
    intermediate = Lambda(lambda x: x)(h11) # return correlation instead of anti-correlation
    intermediate = concatenate([intermediate,
                                dim_cp_layer_h21,
                                dim_h0_layer_h21])
    if len(dense) > 0:
        for n in range(len(dense)):
            intermediate = Dense(dense[n])(intermediate)
            if activation == 'relu':
                intermediate = Activation('relu')(intermediate)
            else:
                intermediate = LeakyReLU(alpha=activation)(intermediate)
    
    h21 = concatenate([num_cp_layer_h21,
                       dim_cp_layer_h21,
                       dim_h0_layer_h21,
                       matrix_layer_h21,
                       intermediate])
    h21 = Dense(1, name='h_21')(h21)
            
    model = Model(inputs=[num_cp_layer_in,
                          dim_cp_layer_in,
                          dim_h0_layer_in,
                          matrix_layer_in],
                  outputs=[h11, h21]
                 )
    
    return model
