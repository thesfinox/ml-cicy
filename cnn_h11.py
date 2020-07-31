#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks
# 
# We study the performance of CNN architectures on the configuration matrices of CICY 3-folds.

# In[1]:


# set memory growth (necessary for training)
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# ## Download the Dataset
# 
# We first download and unzip the dataset.

# In[2]:


import urllib, tarfile, os

file_url = 'http://www.lpthe.jussieu.fr/~erbin/files/data/cicy3o_data.tar.gz'
file_out = './cicy3o.tar.gz'
file_dat = 'cicy3o.h5'

if not os.path.isfile(file_out):
    urllib.request.urlretrieve(file_url, file_out)
    
if not os.path.isfile(file_dat):
    with tarfile.open(file_out, 'r') as tar:
        tar.extract(file_dat)


# ## Load the Dataset
# 
# We then load the dataset:

# In[3]:


import pandas as pd

dat = pd.read_hdf(os.path.join('.', file_dat))


# Remove the outliers (keep $h^{1,1} \in [1, 16]$ and $h^{2,1} \in [15, 86]$):

# In[4]:


dat_out   = dat
dat_noout = dat.loc[(dat['h11'] > 0) &
                    (dat['h11'] < 17) &
                    (dat['h21'] > 14) &
                    (dat['h21'] < 87)
                   ]

dat_out   = dat_out[['h11', 'matrix']]
dat_noout = dat_noout[['h11', 'matrix']]


# Then extract the `matrix` column into its dense format:

# In[5]:


import numpy as np

def extract_series(series: pd.Series) -> pd.Series:
    '''
    Extract a Pandas series into its dense format.
    
    Required arguments:
        series: the pandas series.
        
    Returns:
        the pandas series in dense format.
    '''
    # avoid direct overwriting
    series = series.copy()
    
    # cget the maximum size of each axis
    max_shape = series.apply(np.shape).max()
    
    # return the transformed series
    if np.prod(max_shape) > 1:
        # compute the necessary shift and apply it
        offset = lambda s: [(0, max_shape[i] - np.shape(s)[i])
                            for i in range(len(max_shape))
                           ]
        return series.apply(lambda s: np.pad(s, offset(s), mode='constant'))
    else:
        return series
    
# apply it to the matrix
dat_out   = dat_out.apply(extract_series)
dat_noout = dat_noout.apply(extract_series)


# ## Training and Validation Strategy
# 
# We then subsample the set into training, validation and test sets for evaluation.

# In[6]:


from sklearn.model_selection import train_test_split

# set random state
RAND = 42
np.random.seed(RAND)
tf.random.set_seed(RAND)

# split training set
dat_out_train_80,   dat_out_test_80   = train_test_split(dat_out, train_size=0.8, shuffle=True, random_state=RAND)
dat_out_train_30,   dat_out_test_30   = train_test_split(dat_out, train_size=0.3, shuffle=True, random_state=RAND)
dat_noout_train_80, dat_noout_test_80 = train_test_split(dat_noout, train_size=0.8, shuffle=True, random_state=RAND)
dat_noout_train_30, dat_noout_test_30 = train_test_split(dat_noout, train_size=0.3, shuffle=True, random_state=RAND)

# split validation set
dat_out_val_80,   dat_out_test_80   = train_test_split(dat_out_test_80, train_size=0.5, shuffle=True, random_state=RAND)
dat_out_val_30,   dat_out_test_30   = train_test_split(dat_out_test_30, train_size=1/7, shuffle=True, random_state=RAND)
dat_noout_val_80, dat_noout_test_80 = train_test_split(dat_noout_test_80, train_size=0.5, shuffle=True, random_state=RAND)
dat_noout_val_30, dat_noout_test_30 = train_test_split(dat_noout_test_30, train_size=1/7, shuffle=True, random_state=RAND)

# check sizes
print('80% training data:')
print('    Training set w/ outliers:   {:.2f}%'.format(100 * dat_out_train_80.shape[0] / dat_out.shape[0]))
print('    Validation set w/ outliers: {:.2f}%'.format(100 * dat_out_val_80.shape[0] / dat_out.shape[0]))
print('    Test set w/ outliers:       {:.2f}%'.format(100 * dat_out_test_80.shape[0] / dat_out.shape[0]))
print('')
print('    Training set w/o outliers:   {:.2f}%'.format(100 * dat_noout_train_80.shape[0] / dat_noout.shape[0]))
print('    Validation set w/o outliers: {:.2f}%'.format(100 * dat_noout_val_80.shape[0] / dat_noout.shape[0]))
print('    Test set w/o outliers:       {:.2f}%'.format(100 * dat_noout_test_80.shape[0] / dat_noout.shape[0]))
print('')
print('30% training data:')
print('    Training set w/ outliers:   {:.2f}%'.format(100 * dat_out_train_30.shape[0] / dat_out.shape[0]))
print('    Validation set w/ outliers: {:.2f}%'.format(100 * dat_out_val_30.shape[0] / dat_out.shape[0]))
print('    Test set w/ outliers:       {:.2f}%'.format(100 * dat_out_test_30.shape[0] / dat_out.shape[0]))
print('')
print('    Training set w/o outliers:   {:.2f}%'.format(100 * dat_noout_train_30.shape[0] / dat_noout.shape[0]))
print('    Validation set w/o outliers: {:.2f}%'.format(100 * dat_noout_val_30.shape[0] / dat_noout.shape[0]))
print('    Test set w/o outliers:       {:.2f}%'.format(100 * dat_noout_test_30.shape[0] / dat_noout.shape[0]))


# # Input Visualisation (Training Set)
# 
# We then want to visualise the input matrices for comparison with the output.

# In[7]:


os.makedirs('img', exist_ok=True)


# ## Random Samples
# 
# We first visualise random samples in the training set (they must be scaled in the interval $[0,1]$):

# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# normalisation factor (use the matrix w/o outliers to avoid getting an outlier)
max_train_entry = dat_noout_train_80['matrix'].apply(np.max).max()

ncols = 4
nrows = 4
_, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))

for i in range(nrows):
    for j in range(ncols):
        
        matrix = dat_noout_train_80['matrix'].                    iloc[np.random.randint(low=0,
                                           high=dat_noout_train_80.shape[0] - 1
                                          )
                        ]
        sns.heatmap(data=matrix / max_train_entry,
                    vmin=0.0,
                    vmax=1.0,
                    ax=ax[i,j],
                    cmap='Blues',
                    xticklabels=False,
                    yticklabels=False
                   )
        
plt.tight_layout()
plt.savefig('./img/train_samples_rnd.pdf', dpi=150, format='pdf')


# ## Average Entries
# 
# We then compute the average of all entries as a representative of the training set (first of all we scale the input and then take the mean value of each entry).

# In[9]:


avg_mat = dat_noout_train_80['matrix'] / max_train_entry
avg_mat = avg_mat.mean()

_, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.heatmap(data=avg_mat,
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            cmap='Blues',
            xticklabels=False,
            yticklabels=False
           )
        
plt.tight_layout()
plt.savefig('./img/train_samples_avg.pdf', dpi=150, format='pdf')


# # CNN Architectures
# 
# We then study both a sequential CNN and the Inception-like CNN to analyse their feature maps.

# In[10]:


# matrix
mat_out_train_80   = np.array(dat_out_train_80['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_noout_train_80 = np.array(dat_noout_train_80['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_out_train_30   = np.array(dat_out_train_30['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_noout_train_30 = np.array(dat_noout_train_30['matrix'].tolist()).reshape(-1, 12, 15, 1)

mat_out_val_80     = np.array(dat_out_val_80['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_noout_val_80   = np.array(dat_noout_val_80['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_out_val_30     = np.array(dat_out_val_30['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_noout_val_30   = np.array(dat_noout_val_30['matrix'].tolist()).reshape(-1, 12, 15, 1)

mat_out_test_80    = np.array(dat_out_test_80['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_noout_test_80  = np.array(dat_noout_test_80['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_out_test_30    = np.array(dat_out_test_30['matrix'].tolist()).reshape(-1, 12, 15, 1)
mat_noout_test_30  = np.array(dat_noout_test_30['matrix'].tolist()).reshape(-1, 12, 15, 1)

input_shape = (12, 15, 1)

# labels
lab_out_train_80 = dat_out_train_80['h11'].values.reshape(-1,1)
lab_noout_train_80 = dat_noout_train_80['h11'].values.reshape(-1,1)
lab_out_train_30 = dat_out_train_30['h11'].values.reshape(-1,1)
lab_noout_train_30 = dat_noout_train_30['h11'].values.reshape(-1,1)

lab_out_val_80   = dat_out_val_80['h11'].values.reshape(-1,1)
lab_noout_val_80   = dat_noout_val_80['h11'].values.reshape(-1,1)
lab_out_val_30   = dat_out_val_30['h11'].values.reshape(-1,1)
lab_noout_val_30   = dat_noout_val_30['h11'].values.reshape(-1,1)


lab_out_test_80   = dat_out_test_80['h11'].values.reshape(-1,1)
lab_noout_test_80   = dat_noout_test_80['h11'].values.reshape(-1,1)
lab_out_test_30   = dat_out_test_30['h11'].values.reshape(-1,1)
lab_noout_test_30   = dat_noout_test_30['h11'].values.reshape(-1,1)


# In[11]:


# outdir
os.makedirs('./mod', exist_ok=True)


# ## Inception CNN

# In[12]:


from tensorflow import keras

def inc_model(input_shape,
              model_name='inception',
              learning_rate=0.001,
              filters=[32],
              kernel_size=[(3,3), (5,5)],
              dropout=0.2,
              momentum=0.99,
              l1_reg=0.0,
              l2_reg=0.0
             ):
    
    # reset session
    keras.backend.clear_session()
    
    # define the regularisation factor
    kernel_regularizer = keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    
    # input of the model
    I = keras.layers.Input(shape=input_shape, name=model_name + '_input')
    x = I
    
    # convolutional layers
    for n in range(len(filters)):
        a = keras.layers.Conv2D(filters=filters[n],
                                kernel_size=kernel_size[0],
                                padding='same',
                                kernel_regularizer=kernel_regularizer,
                                activation='relu',
                                name=model_name + '_convA_' + str(n+1)
                               )(x)
        b = keras.layers.Conv2D(filters=filters[n],
                                kernel_size=kernel_size[1],
                                padding='same',
                                kernel_regularizer=kernel_regularizer,
                                activation='relu',
                                name=model_name + '_convB_' + str(n+1)
                               )(x)
        x = keras.layers.concatenate([a, b], name=model_name + '_conc_' + str(n+1))
        if momentum > 0.0:
            x = keras.layers.BatchNormalization(momentum=momentum,
                                                name=model_name + '_bnorm_' + str(n+1)
                                               )(x)
    
    # dropout layer
    if dropout > 0.0:
        x = keras.layers.Dropout(rate=dropout, name=model_name + '_drop')(x)
        
    # flatten
    x = keras.layers.Flatten(name=model_name + '_flat')(x)
    
    # output
    h11 = keras.layers.Dense(units=1,
                             activation='relu',
                             name='h11_output'
                            )(x)
    
    # define the model
    model = keras.models.Model(inputs=I, outputs=h11, name=model_name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mse', 'mae']
                 )
    
    return model


# We then define the model.

# In[13]:


inception_line = inc_model(input_shape=input_shape,
                           model_name='inc_line',
                           learning_rate=0.001,
                           filters=[32, 64, 32],
                           kernel_size=[(12,1), (1,15)],
                           dropout=0.2,
                           l1_reg=1.0e-4,
                           l2_reg=1.0e-4
                          )
inception_square = inc_model(input_shape=input_shape,
                             model_name='inc_square',
                             learning_rate=0.001,
                             filters=[32, 64, 32],
                             kernel_size=[(3,3), (5,5)],
                             dropout=0.2,
                             l1_reg=1.0e-4,
                             l2_reg=1.0e-4
                            )


# The model can be visualised using `keras` utility functions:

# In[14]:


#from IPython.display import Image

inception_line_dot = keras.utils.model_to_dot(inception_line,
                                              show_shapes=True,
                                              dpi=150
                                             )
inception_line_dot.write_pdf('./img/inc_arch_h11.pdf')
#Image(inception_line_dot.create_png(), width=480)


# We then train the model.

# In[15]:


os.makedirs('./dat', exist_ok=True)


# ### 80% Training Data w/ Outliers (line kernel)

# In[16]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_line_out_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

print('Training 80% data w/ outliers (line kernel)')
inception_line.summary()

# define the model
inception_line = inc_model(input_shape=input_shape,
                           model_name='inc_line',
                           learning_rate=0.001,
                           filters=[32, 64, 32],
                           kernel_size=[(12,1), (1,15)],
                           dropout=0.2,
                           l1_reg=1.0e-4,
                           l2_reg=1.0e-4
                          )
inc_line_out_80 = inception_line.fit(x=mat_out_train_80,
                                     y=lab_out_train_80,
                                     batch_size=32,
                                     epochs=2000,
                                     verbose=0,
                                     callbacks=callbacks,
                                     validation_data=(mat_out_val_80, lab_out_val_80)
                                    )


# We the take a look at the accuracy of the output:

# In[17]:


import joblib

# predictions
h11_pred = np.rint(inception_line.predict(mat_out_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true,
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_line_out_80_h11.csv')

# save history
joblib.dump(inc_line_out_80.history, './dat/inc_line_out_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 80% Training Data w/o Outliers (line kernel)

# In[18]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_line_noout_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

inception_line.summary()

print('Training 80% data w/o outliers (line kernel)')

# define the model
inception_line = inc_model(input_shape=input_shape,
                           model_name='inc_line',
                           learning_rate=0.001,
                           filters=[32, 64, 32],
                           kernel_size=[(12,1), (1,15)],
                           dropout=0.2,
                           l1_reg=1.0e-4,
                           l2_reg=1.0e-4
                          )
inc_line_noout_80 = inception_line.fit(x=mat_noout_train_80,
                                       y=lab_noout_train_80,
                                       batch_size=32,
                                       epochs=2000,
                                       verbose=0,
                                       callbacks=callbacks,
                                       validation_data=(mat_noout_val_80, lab_noout_val_80)
                                      )


# We the take a look at the accuracy of the output:

# In[19]:


import joblib

# predictions
h11_pred = np.rint(inception_line.predict(mat_noout_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_line_noout_80_h11.csv')

# save history
joblib.dump(inc_line_noout_80.history, './dat/inc_line_noout_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/ Outliers (line kernel)

# In[20]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_line_out_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           verbose=0
                                          )
            ]

inception_line.summary()

print('Training 30% data w/ outliers (line kernel)')

# define the model
inception_line = inc_model(input_shape=input_shape,
                           model_name='inc_line',
                           learning_rate=0.001,
                           filters=[32, 64, 32],
                           kernel_size=[(12,1), (1,15)],
                           dropout=0.2,
                           l1_reg=1.0e-4,
                           l2_reg=1.0e-4
                          )
inc_line_out_30 = inception_line.fit(x=mat_out_train_30,
                                     y=lab_out_train_30,
                                     batch_size=32,
                                     epochs=2000,
                                     verbose=0,
                                     callbacks=callbacks,
                                     validation_data=(mat_out_val_30, lab_out_val_30)
                                    )


# We the take a look at the accuracy of the output:

# In[21]:


import joblib

# predictions
h11_pred = np.rint(inception_line.predict(mat_out_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_line_out_30_h11.csv')

# save history
joblib.dump(inc_line_out_30.history, './dat/inc_line_out_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/o Outliers (line kernel)

# In[22]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_line_noout_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

inception_line.summary()

print('Training 30% data w/o outliers (line kernel)')

# define the model
inception_line = inc_model(input_shape=input_shape,
                           model_name='inc_line',
                           learning_rate=0.001,
                           filters=[32, 64, 32],
                           kernel_size=[(12,1), (1,15)],
                           dropout=0.2,
                           l1_reg=1.0e-4,
                           l2_reg=1.0e-4
                          )
inc_line_noout_30 = inception_line.fit(x=mat_noout_train_30,
                                       y=lab_noout_train_30,
                                       batch_size=32,
                                       epochs=2000,
                                       verbose=0,
                                       callbacks=callbacks,
                                       validation_data=(mat_noout_val_30, lab_noout_val_30)
                                      )


# We the take a look at the accuracy of the output:

# In[23]:


import joblib

# predictions
h11_pred = np.rint(inception_line.predict(mat_noout_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_line_noout_30_h11.csv')

# save history
joblib.dump(inc_line_noout_30.history, './dat/inc_line_noout_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 80% Training Data w/ Outliers (square kernel)

# In[16]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_square_out_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

inception_square.summary()

print('Training 80% data w/ outliers (square kernel)')

# define the model
inception_square = inc_model(input_shape=input_shape,
                             model_name='inc_square',
                             learning_rate=0.001,
                             filters=[32, 64, 32],
                             kernel_size=[(3,3), (5,5)],
                             dropout=0.2,
                             l1_reg=1.0e-4,
                             l2_reg=1.0e-4
                            )
inc_square_out_80 = inception_square.fit(x=mat_out_train_80,
                                         y=lab_out_train_80,
                                         batch_size=32,
                                         epochs=2000,
                                         verbose=0,
                                         callbacks=callbacks,
                                         validation_data=(mat_out_val_80, lab_out_val_80)
                                        )


# We the take a look at the accuracy of the output:

# In[17]:


import joblib

# predictions
h11_pred = np.rint(inception_square.predict(mat_out_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_square_out_80_h11.csv')

# save history
joblib.dump(inc_square_out_80.history, './dat/inc_square_out_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 80% Training Data w/o Outliers (square kernel)

# In[18]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_square_noout_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

inception_square.summary()

print('Training 80% data w/o outliers (square kernel)')

# define the model
inception_square = inc_model(input_shape=input_shape,
                             model_name='inc_square',
                             learning_rate=0.001,
                             filters=[32, 64, 32],
                             kernel_size=[(3,3), (5,5)],
                             dropout=0.2,
                             l1_reg=1.0e-4,
                             l2_reg=1.0e-4
                            )
inc_square_noout_80 = inception_square.fit(x=mat_noout_train_80,
                                           y=lab_noout_train_80,
                                           batch_size=32,
                                           epochs=2000,
                                           verbose=0,
                                           callbacks=callbacks,
                                           validation_data=(mat_noout_val_80, lab_noout_val_80)
                                          )


# We the take a look at the accuracy of the output:

# In[19]:


import joblib

# predictions
h11_pred = np.rint(inception_square.predict(mat_noout_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_square_noout_80_h11.csv')

# save history
joblib.dump(inc_square_noout_80.history, './dat/inc_square_noout_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/ Outliers (square kernel)

# In[23]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_square_out_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

inception_square.summary()

print('Training 30% data w/ outliers (square kernel)')

# define the model
inception_square = inc_model(input_shape=input_shape,
                             model_name='inc_square',
                             learning_rate=0.001,
                             filters=[32, 64, 32],
                             kernel_size=[(3,3), (5,5)],
                             dropout=0.2,
                             l1_reg=1.0e-4,
                             l2_reg=1.0e-4
                            )
inc_square_out_30 = inception_square.fit(x=mat_out_train_30,
                                         y=lab_out_train_30,
                                         batch_size=32,
                                         epochs=2000,
                                         verbose=0,
                                         callbacks=callbacks,
                                         validation_data=(mat_out_val_30, lab_out_val_30)
                                        )


# We the take a look at the accuracy of the output:

# In[24]:


import joblib

# predictions
h11_pred = np.rint(inception_square.predict(mat_out_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_square_out_30_h11.csv')

# save history
joblib.dump(inc_square_out_30.history, './dat/inc_square_out_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/o Outliers (square kernel)

# In[25]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/inception_square_noout_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

inception_square.summary()

print('Training 30% data w/o outliers (square kernel)')

# define the model
inception_square = inc_model(input_shape=input_shape,
                             model_name='inc_square',
                             learning_rate=0.001,
                             filters=[32, 64, 32],
                             kernel_size=[(3,3), (5,5)],
                             dropout=0.2,
                             l1_reg=1.0e-4,
                             l2_reg=1.0e-4
                            )
inc_square_noout_30 = inception_square.fit(x=mat_noout_train_30,
                                           y=lab_noout_train_30,
                                           batch_size=32,
                                           epochs=2000,
                                           verbose=0,
                                           callbacks=callbacks,
                                           validation_data=(mat_noout_val_30, lab_noout_val_30)
                                          )


# We the take a look at the accuracy of the output:

# In[26]:


import joblib

# predictions
h11_pred = np.rint(inception_square.predict(mat_noout_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/inc_square_noout_30_h11.csv')

# save history
joblib.dump(inc_square_noout_30.history, './dat/inc_square_noout_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ## Sequential CNN

# In[32]:


from tensorflow import keras

def seq_model(input_shape,
              model_name='sequential',
              learning_rate=0.001,
              filters=[32],
              padding='same',
              kernel_size=(2,2),
              dropout=0.2,
              momentum=0.99,
              l1_reg=0.0,
              l2_reg=0.0
             ):
    
    # reset session
    keras.backend.clear_session()
    
    # define the regularisation factor
    kernel_regularizer = keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    
    # input of the model
    I = keras.layers.Input(shape=input_shape, name=model_name + '_input')
    x = I
    
    # convolutional layers
    for n in range(len(filters)):
        x = keras.layers.Conv2D(filters=filters[n],
                                kernel_size=kernel_size,
                                padding=padding,
                                kernel_regularizer=kernel_regularizer,
                                activation='relu',
                                name=model_name + '_conv_' + str(n+1)
                               )(x)
        if momentum > 0.0:
            x = keras.layers.BatchNormalization(momentum=momentum,
                                                name=model_name + '_bnorm_' + str(n+1)
                                               )(x)
    
    # dropout layer
    if dropout > 0.0:
        x = keras.layers.Dropout(rate=dropout, name=model_name + '_drop')(x)
        
    # flatten
    x = keras.layers.Flatten(name=model_name + '_flat')(x)
    
    # output
    h11 = keras.layers.Dense(units=1,
                             activation='relu',
                             name='h11_output'
                            )(x)
    
    # define the model
    model = keras.models.Model(inputs=I, outputs=h11, name=model_name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mse', 'mae']
                 )
    
    return model


# We then define the model.

# In[33]:


sequential_small = seq_model(input_shape=input_shape,
                             model_name='seq_small',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(3,3),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
sequential_large = seq_model(input_shape=input_shape,
                             model_name='seq_large',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(5,5),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )


# The model can be visualised using `keras` utility functions:

# In[34]:


#from IPython.display import Image

sequential_dot = keras.utils.model_to_dot(sequential_small,
                                          show_shapes=True,
                                          dpi=150
                                         )
sequential_dot.write_pdf('./img/seq_arch_h11.pdf')
#Image(sequential_dot.create_png(), width=480)


# We then train the model.

# In[35]:


os.makedirs('./dat', exist_ok=True)


# ### 80% Training Data w/ Outliers (small kernel)

# In[56]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_small_out_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_small.summary()

print('Training 80% data w/ outliers (small kernel)')

# define the model
sequential_small = seq_model(input_shape=input_shape,
                            model_name='seq_small',
                            learning_rate=0.001,
                            filters=[180, 100, 40, 20],
                            kernel_size=(3,3),
                            dropout=0.3,
                            l1_reg=1.0e-5,
                            l2_reg=1.0e-5
                           )
seq_small_out_80 = sequential_small.fit(x=mat_out_train_80,
                                        y=lab_out_train_80,
                                        batch_size=32,
                                        epochs=2000,
                                        verbose=0,
                                        callbacks=callbacks,
                                        validation_data=(mat_out_val_80, lab_out_val_80)
                                       )


# We the take a look at the accuracy of the output:

# In[57]:


import joblib

# predictions
h11_pred = np.rint(sequential_small.predict(mat_out_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_small_out_80_h11.csv')

# save history
joblib.dump(seq_small_out_80.history, './dat/seq_small_out_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 80% Training Data w/o Outliers (small kernel)

# In[41]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_small_noout_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_small.summary()

print('Training 80% data w/o outliers (small kernel)')

# define the model
sequential_small = seq_model(input_shape=input_shape,
                             model_name='seq_small',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(3,3),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
seq_small_noout_80 = sequential_small.fit(x=mat_noout_train_80,
                                          y=lab_noout_train_80,
                                          batch_size=32,
                                          epochs=2000,
                                          verbose=0,
                                          callbacks=callbacks,
                                          validation_data=(mat_noout_val_80, lab_noout_val_80)
                                         )


# We the take a look at the accuracy of the output:

# In[42]:


import joblib

# predictions
h11_pred = np.rint(sequential_small.predict(mat_noout_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_small_noout_80_h11.csv')

# save history
joblib.dump(seq_small_noout_80.history, './dat/seq_small_noout_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/ Outliers (small kernel)

# In[43]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_small_out_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_small.summary()

print('Training 30% data w/ outliers (small kernel)')

# define the model
sequential_small = seq_model(input_shape=input_shape,
                             model_name='seq_small',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(3,3),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
seq_small_out_30 = sequential_small.fit(x=mat_out_train_30,
                                        y=lab_out_train_30,
                                        batch_size=32,
                                        epochs=2000,
                                        verbose=0,
                                        callbacks=callbacks,
                                        validation_data=(mat_out_val_30, lab_out_val_30)
                                       )


# We the take a look at the accuracy of the output:

# In[44]:


import joblib

# predictions
h11_pred = np.rint(sequential_small.predict(mat_out_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true,
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_small_out_30_h11.csv')

# save history
joblib.dump(seq_small_out_30.history, './dat/seq_small_out_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/o Outliers (small kernel)

# In[45]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_small_noout_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_small.summary()

print('Training 30% data w/o outliers (small kernel)')

# define the model
sequential_small = seq_model(input_shape=input_shape,
                             model_name='seq_small',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(3,3),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
seq_small_noout_30 = sequential_small.fit(x=mat_noout_train_30,
                                          y=lab_noout_train_30,
                                          batch_size=32,
                                          epochs=2000,
                                          verbose=0,
                                          callbacks=callbacks,
                                          validation_data=(mat_noout_val_30, lab_noout_val_30)
                                         )


# We the take a look at the accuracy of the output:

# In[46]:


import joblib

# predictions
h11_pred = np.rint(sequential_small.predict(mat_noout_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_small_noout_30_h11.csv')

# save history
joblib.dump(seq_small_noout_30.history, './dat/seq_small_noout_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 80% Training Data w/ Outliers (large kernel)

# In[47]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_large_out_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_large.summary()

print('Training 80% data w/ outliers (large kernel)')

# define the model
sequential_large = seq_model(input_shape=input_shape,
                             model_name='seq_large',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(5,5),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
seq_large_out_80 = sequential_large.fit(x=mat_out_train_80,
                                        y=lab_out_train_80,
                                        batch_size=32,
                                        epochs=2000,
                                        verbose=0,
                                        callbacks=callbacks,
                                        validation_data=(mat_out_val_80, lab_out_val_80)
                                       )


# We the take a look at the accuracy of the output:

# In[48]:


import joblib

# predictions
h11_pred = np.rint(sequential_large.predict(mat_out_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_large_out_80_h11.csv')

# save history
joblib.dump(seq_large_out_80.history, './dat/seq_large_out_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 80% Training Data w/o Outliers (large kernel)

# In[49]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_large_noout_80_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_large.summary()

print('Training 80% data w/o outliers (large kernel)')

# define the model
sequential_large = seq_model(input_shape=input_shape,
                             model_name='seq_large',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(5,5),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
seq_large_noout_80 = sequential_large.fit(x=mat_noout_train_80,
                                          y=lab_noout_train_80,
                                          batch_size=32,
                                          epochs=2000,
                                          verbose=0,
                                          callbacks=callbacks,
                                          validation_data=(mat_noout_val_80, lab_noout_val_80)
                                         )


# We the take a look at the accuracy of the output:

# In[50]:


import joblib

# predictions
h11_pred = np.rint(sequential_large.predict(mat_noout_test_80)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_80.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_large_noout_80_h11.csv')

# save history
joblib.dump(seq_large_noout_80.history, './dat/seq_large_noout_80_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/ Outliers (large kernel)

# In[51]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_large_out_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_large.summary()

print('Training 30% data w/ outliers (large kernel)')

# define the model
sequential_large = seq_model(input_shape=input_shape,
                             model_name='seq_large',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(5,5),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
seq_large_out_30 = sequential_large.fit(x=mat_out_train_30,
                                        y=lab_out_train_30,
                                        batch_size=32,
                                        epochs=2000,
                                        verbose=0,
                                        callbacks=callbacks,
                                        validation_data=(mat_out_val_30, lab_out_val_30)
                                       )


# We the take a look at the accuracy of the output:

# In[52]:


import joblib

# predictions
h11_pred = np.rint(sequential_large.predict(mat_out_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_out_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_large_out_30_h11.csv')

# save history
joblib.dump(seq_large_out_30.history, './dat/seq_large_out_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))


# ### 30% Training Data w/o Outliers (large kernel)

# In[53]:


callbacks = [keras.callbacks.ModelCheckpoint('./mod/sequential_large_noout_30_h11.h5',
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True
                                            ),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.3,
                                               patience=80,
                                               verbose=0,
                                               min_lr=1.0e-6
                                              ),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=200,
                                           restore_best_weights=True,
                                           verbose=0
                                          )
            ]

sequential_large.summary()

print('Training 30% data w/o outliers (large kernel)')

# define the model
sequential_large = seq_model(input_shape=input_shape,
                             model_name='seq_large',
                             learning_rate=0.001,
                             filters=[180, 100, 40, 20],
                             kernel_size=(5,5),
                             dropout=0.3,
                             l1_reg=1.0e-5,
                             l2_reg=1.0e-5
                            )
seq_large_noout_30 = sequential_large.fit(x=mat_noout_train_30,
                                          y=lab_noout_train_30,
                                          batch_size=32,
                                          epochs=2000,
                                          verbose=0,
                                          callbacks=callbacks,
                                          validation_data=(mat_noout_val_30, lab_noout_val_30)
                                         )


# We the take a look at the accuracy of the output:

# In[54]:


import joblib

# predictions
h11_pred = np.rint(sequential_large.predict(mat_noout_test_30)).astype(int).reshape(-1,)

# true values
h11_true = lab_noout_test_30.astype(int).reshape(-1,)

# accuracy
h11_acc = np.mean((h11_pred == h11_true).astype(int))

# save predictions
predictions = {'h11_pred': h11_pred,
               'h11_true': h11_true
              }
predictions = pd.DataFrame(predictions)
predictions.to_csv('./dat/seq_large_noout_30_h11.csv')

# save history
joblib.dump(seq_large_noout_30.history, './dat/seq_large_noout_30_h11.pkl')

# print accuracy
print('Accuracy for h_11: {:.2f}%'.format(h11_acc * 100))

