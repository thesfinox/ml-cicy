import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt
import pandas            as pd
import joblib
import os
import sys
import argparse

# prepare common settings
os.makedirs('./img', exist_ok=True)

from sklearn.decomposition import PCA

def pca_plot(df: pd.DataFrame, label: str, type: str) -> None:
    '''
    Plot the 2D PCA visualisation.
    
    Required arguments:
        df:    the dataset to use,
        label: the label to be used for visualisation,
        type:  the type of the dataset (orig. or fav.).
    '''
    
    sns.set()

    _, ax = plt.subplots(1, 1, figsize=(6,5))
    
    sns.scatterplot(x='pca_1',
                    y='pca_2',
                    hue=label,
                    data=df,
                    palette=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True),
                    ax=ax
                   )
    ax.set(xlabel='PCA #1',
           ylabel='PCA #2'
          )
    
    plt.tight_layout()
    plt.savefig('./img/pca2d_{}_{}.pdf'.format(label, type), dpi=150, format='pdf')
    plt.savefig('./img/pca2d_{}_{}.png'.format(label, type), dpi=150, format='png')

# set up the argument parser
parser = argparse.ArgumentParser()

parser.add_argument('-r', '--rand', type=int, default=42, help='random state')

args   = parser.parse_args()

# fix the random state
RAND = args.rand
    
np.random.RandomState(RAND)
np.random.seed(RAND)

# load the dataset
df_o = pd.read_csv('./data/cicy3o_tidy.csv')
df_f = pd.read_csv('./data/cicy3f_tidy.csv')

# load the labels
lab = pd.read_csv('./data/labels.csv')

# select only the matrix components
mat_o = df_o.filter(regex='^matrix_')
mat_f = df_f.filter(regex='^matrix_')

# compute the PCA for 2D visualisation
pca2_o, pca2_f = joblib.Parallel(n_jobs=-1)\
                 (joblib.delayed(PCA(n_components=2, random_state=RAND).fit_transform)(mat)
                                 for mat in [mat_o, mat_f]
                 )

# compute the PCA with 99% of variance retained
pca99_o, pca99_f = joblib.Parallel(n_jobs=-1)\
                   (joblib.delayed(PCA(n_components=0.99, random_state=RAND).fit_transform)(mat)
                                   for mat in [mat_o, mat_f]
                   )

# add the PCA to the dataframe
pca99_o = pd.DataFrame(pca99_o).rename(columns=lambda x: 'pca_' + str(x+1))

pca99_f = pd.DataFrame(pca99_f).rename(columns=lambda x: 'pca_' + str(x+1))

# save the new dataframe
pca99_o.to_csv('./data/cicy3o_pca.csv', index=False)
pca99_f.to_csv('./data/cicy3f_pca.csv', index=False)

# produce the visualisation data
pca2_o_data = {'pca_1': pca2_o[:,0],
               'pca_2': pca2_o[:,1],
               'h11':   lab['h11'].values.astype(int),
               'h21':   lab['h21'].values.astype(int)
              }
pca2_o_data = pd.DataFrame(pca2_o_data)

pca2_f_data = {'pca_1': pca2_f[:,0],
               'pca_2': pca2_f[:,1],
               'h11':   lab['h11'].values.astype(int),
               'h21':   lab['h21'].values.astype(int)
              }
pca2_f_data = pd.DataFrame(pca2_f_data)

# produce the visualisation plots
joblib.Parallel(n_jobs=-1)(joblib.delayed(pca_plot)(*args)
                           for args in [(pca2_o_data, 'h11', 'orig'),
                                        (pca2_f_data, 'h11', 'fav'),
                                        (pca2_o_data, 'h21', 'orig'),
                                        (pca2_f_data, 'h21', 'fav'),
                                       ]
                          )