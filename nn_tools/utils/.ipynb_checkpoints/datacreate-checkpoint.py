import pandas as pd
import numpy as np
from tensorflow import keras

def create_features(data, rescaling=None, reshape=None):
    '''
    Create the training features (rescaled if necessary).
    
    Needed arguments:
        data: the Pandas Series with the data.
        
    Optional arguments:
        rescaling: dictionary containing the min and max rescaling parameters,
        reshape:   reshape each single feature to that shape.
    '''
    name = data.name
    
    if rescaling is not None:
        data = data.apply(lambda x: (x - rescaling['min']) / (rescaling['max'] - rescaling['min']))
    
    if reshape is not None:
        return {name: np.array([np.array(data.iloc[n]).reshape(reshape).astype(np.float32) for n in range(data.shape[0])])}
    else:
        return {name: np.array([np.array(data.iloc[n]).astype(np.float32) for n in range(data.shape[0])])}
    

def create_labels(data, one_hot=False, num_classes=None):
    '''
    Create the training features (rescaled if necessary).
    
    Needed arguments:
        data: the Pandas DataFrame with the data.
        
    Optional arguments:
        one_hot:     one hot encoding,
        num_classes: number of classes.
    '''
    
    if one_hot:
            
        if num_classes is not None:
            if not isinstance(num_classes, list):
                num_classes = [num_classes] * data.shape[1]
            
            labels = {}
            for n, name in enumerate(data.columns):
                labels[name] = keras.utils.to_categorical(data[name].values.reshape(-1,).astype(np.int), num_classes=num_classes[n])
                
            return labels
        
        else:
            return {name: keras.utils.to_categorical(data[name].values.reshape(-1,).astype(np.int)) for name in data.columns}
    else:
        return {name: data[name].values.reshape(-1,).astype(np.int) for name in data.columns}