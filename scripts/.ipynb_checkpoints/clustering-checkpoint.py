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

from sklearn.cluster import KMeans

# produce the plots
def kmeans_plot(df: pd.DataFrame, clustering: str, name: str) -> None:
    '''
    Scatter plot of the clustering algorithm.
    
    Required arguments:
        df:         the dataframe to use,
        clustering: the column of the dataframe
        name:       the name of the dataframe (orig. or fav.).
    '''
    
    sns.set()
    
    _, ax = plt.subplots(1, 1, figsize=(6,5))
    
    sns.scatterplot(x='h11',
                    y='h21',
                    hue=clustering,
                    data=df,
                    ax=ax
                   )
    ax.set(xlabel='$h^{1,1}$',
           ylabel='$h^{2,1}$',
           xticks=np.arange(min(df['h11']),
                            max(df['h11']) + 1,
                            int((max(df['h11'])+1 - min(df['h11'])) / 10)
                           ),
           yticks=np.arange(min(df['h21']),
                            max(df['h21']) + 1,
                            int((max(df['h21'])+1 - min(df['h21'])) / 10)
                           )
          )
    
    plt.tight_layout()
    plt.savefig('./img/{}_{}.pdf'.format(clustering, name), dpi=150, format='pdf')
    plt.savefig('./img/{}_{}.png'.format(clustering, name), dpi=150, format='png')

# set up the argument parser
parser = argparse.ArgumentParser()

parser.add_argument('-b', '--begin', type=int, default=2, help='start of the cluster range')
parser.add_argument('-e', '--end', type=int, default=15, help='end of the cluster range')
parser.add_argument('-r', '--rand', type=int, default=42, help='random state')

args = parser.parse_args()

from sklearn.decomposition import PCA

# fix the random state
RAND=args.rand
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

# define the cluster range
clusters = range(args.begin, args.end + 1)

# compute the clustering algorithms
kmeans = lambda n, x: KMeans(n_clusters=n, random_state=RAND).fit_predict(x)

labels_o = joblib.Parallel(n_jobs=-1)(joblib.delayed(kmeans)(n, mat_o) for n in clusters)
labels_f = joblib.Parallel(n_jobs=-1)(joblib.delayed(kmeans)(n, mat_f) for n in clusters)

# put everything in a dataframe and save to CSV file
columns = ['kmeans_' + str(k) for k in clusters]

labels_o = pd.DataFrame(np.array(labels_o).T, columns=columns)
labels_o.to_csv('./data/cicy3o_kmeans.csv', index=False)

labels_f = pd.DataFrame(np.array(labels_f).T, columns=columns)
labels_f.to_csv('./data/cicy3f_kmeans.csv', index=False)

# produce the plots
joblib.Parallel(n_jobs=-1)\
(joblib.delayed(lambda s: kmeans_plot(labels_o.join(lab), s, 'orig'))(name)
                for name in columns
               )

joblib.Parallel(n_jobs=-1)\
(joblib.delayed(lambda s: kmeans_plot(labels_f.join(lab), s, 'fav'))(name)
                for name in columns
               )