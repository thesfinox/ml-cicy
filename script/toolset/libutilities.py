# Machine Learning for Complete Intersection Calabi-Yau Manifolds
#
# LIBTRANSFORMERS:
#
#   Library of definitions and classes to transform and extract data using
#   Scikit-learn.
#
# AUTHOR: Riccardo Finotello
#

import numpy   as np
import sklearn as sk
import pandas  as pd

assert np.__version__  >  '1.16'   # to avoid issues with pytables
assert sk.__version__  >= '0.22.1' # for the recent implementation

from os           import path
from sklearn.base import BaseEstimator, TransformerMixin

# Load a Pandas dataset
def load_dataset(dataset):

    if path.isfile(dataset):
        print('Reading database...', flush=True)
        df = pd.read_hdf(dataset)
        print('Database loaded.', flush=True)
    else:
        print('Cannot read the database!', flush=True)

    return df

# Remove the outliers from a Pandas dataset
class RemoveOutliers(BaseEstimator, TransformerMixin):

    def __init__(self, filter_dict=None):

        '''
        The filter_dict must be a Python dictionary containing the limits of the
        outliers to consider. E.g.:
        
            filter_dict = {'h11': [ 1, 15 ],
                           'h21': [ 1, 84 ]
                          }
        
        will keep only 'h11' inside the interval [1,15] and 'h21' in [1,84].
        '''
        self.filter_dict = filter_dict

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        x = X.copy() # avoid overwriting

        if self.filter_dict is not None:
            for key in self.filter_dict:
                x = x.loc[x[key] >= self.filter_dict[key][0]]
                x = x.loc[x[key] <= self.filter_dict[key][1]]

        return x

# Extract the tensors from a Pandas dataset
class ExtractTensor(BaseEstimator, TransformerMixin):

    def __init__(self, flatten=False, shape=None):

        self.flatten = flatten
        self.shape   = shape

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        x = X.copy() # avoid overwriting
        if self.shape is None:
            self.shape = x.apply(np.shape).max() # get the shape of the tensor

        if len(self.shape) > 0:
            offset = lambda s : [ (0, self.shape[i] - np.shape(s)[i])
                                  for i in range(len(self.shape)) ]
            x      = x.apply(lambda s: np.pad(s, offset(s), mode='constant'))

        if self.flatten and len(self.shape) > 0:
            return list(np.stack(x.apply(np.ndarray.flatten).values))
        else:
            return list(np.stack(x.values))

    def get_shape(self):
        
        return self.shape

# Get the accuracy (possibly after rounding)
def accuracy_score(y_true, y_pred, rounding=np.rint):

    if len(y_true) == len(y_pred):
        accuracy = 0
        if rounding is not None:
            for n in range(len(y_true)):
                accuracy = accuracy + 1 \
                           if int(y_true[n]) == int(rounding(y_pred[n])) \
                           else accuracy
        else:
            for n in range(len(y_true)):
                accuracy = accuracy + 1 \
                           if y_true[n] == y_pred[n] \
                           else accuracy
        return accuracy / len(y_true)
    else:
        raise ValueError('Lists have different lengths!')

# Get the error difference (possibly after rounding)
def error_diff(y_true, y_pred, rounding=np.rint):

    if len(y_true) == len(y_pred):
        err = y_true - rounding(y_pred)
        return err.astype('int')
    else:
        raise ValueError('Lists have different lengths!')

# Print GridSearchCV and RandomizedSearchCV scores
def gridcv_score(estimator, rounding=np.rint):
    
    best_params = estimator.best_params_              # get best parameters
    df          = pd.DataFrame(estimator.cv_results_) # dataframe with CV res.
    
    cv_best_res = df.loc[df['params'] == best_params] # get best results
    accuracy    = cv_best_res.loc[:, 'mean_test_score'].values[0]
    std         = cv_best_res.loc[:, 'std_test_score'].values[0]
    
    print('    Best parameters: {}'.format(best_params), flush=True)
    print('    Accuracy ({}) of cross-validation: '.format(rounding.__name__),
          '({:.3f} ± {:.3f})%'.format(accuracy*100, std*100))
    
# Print predictions
def prediction_score(estimator, X, y, use_best_estimator=False, rounding=np.rint):
    
    if use_best_estimator:
        estimator = estimator.best_estimator_
    
    accuracy = accuracy_score(y, estimator.predict(X), rounding=rounding)
    print('    Accuracy ({}) of the predictions: '.format(rounding.__name__),
          '{:.3f}%'.format(accuracy*100))
