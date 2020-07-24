import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt
import pandas            as pd
import joblib
import json
import time
import os
import sys
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.utils         import shuffle

# prepare common settings
os.makedirs('./img', exist_ok=True)
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)
sns.set()

# set up the argument parser
parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', type=str, help='name of the session')
parser.add_argument('-r', '--rand', type=int, default=42, help='random state')
parser.add_argument('-e', '--estimator', type=str, help='path to the pickle file of the estimator')
parser.add_argument('-p', '--hyperparams', type=str, help='path to the hyperparameters JSON file')
parser.add_argument('-f', '--features', type=str, help='path to the features CSV file')
parser.add_argument('-l', '--labels', type=str, help='path to the labels CSV file')
parser.add_argument('-s', '--scale', action='store_true', help='standard scaling of the input features')

args = parser.parse_args()

# import estimator, features and labels
estimator = joblib.load(args.estimator)

# set the name of the session
if args.name is not None:
    name = args.name
else:
    name = estimator.__class__.__name__

with open(args.hyperparams, 'r') as f:
    hyperparams = json.load(f)
    
features = pd.read_csv(args.features)
labels   = pd.read_csv(args.labels)

# set the hyperparameters
estimator.set_params(**hyperparams)

# reshape feature of labels if needed
if features.shape[1] == 1:
    features = features.values.reshape(-1,)
if labels.shape[1] == 1:
    labels = labels.values.reshape(-1,)
    
# scale the features if necessary
if args.scale:
    columns  = features.columns
    features = StandardScaler().fit_transform(features)
    features = pd.DataFrame(features, columns=columns)
    
# train the algorithm
features, labels = shuffle(features, labels, random_state=args.rand)

t = time.time()

estimator.fit(features, labels)

elapsed = time.time() - t
print('{} trained in {:.2f} seconds.'.format(name, elapsed))

# save the trained estimator to file
joblib.dump(estimator, args.estimator)
print('{} saved to {}'.format(name, args.estimator))