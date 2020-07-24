import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt
import pandas            as pd
import joblib
import os

# prepare common settings
sns.set()
os.makedirs('./img', exist_ok=True)

# load the dataset
df_o = pd.read_csv('./data/cicy3o_tidy.csv')
df_f = pd.read_csv('./data/cicy3f_tidy.csv')

# select only the matrix components
df_o = df_o.filter(regex='^matrix_')
df_f = df_f.filter(regex='^matrix_')

# center the distribution
mat_o = df_o - df_o.mean()
mat_f = df_f - df_f.mean()

# compute the singular value decomposition of the matrices
[(_, S_o, _), (_, S_f, _)] = joblib.Parallel(n_jobs=-1)\
                             (joblib.delayed(np.linalg.svd)(mat)
                                             for mat in [mat_o, mat_f]
                                            )

# compute the variance
S2_o, S2_f = joblib.Parallel(n_jobs=-1)\
             (joblib.delayed(lambda x: np.square(x) / np.sum(np.square(x)))(s)
              for s in [S_o, S_f]
             )

# compute the cumulative sum of the variance
S2cum_o, S2cum_f = joblib.Parallel(n_jobs=-1)\
                   (joblib.delayed(np.cumsum)(s)
                    for s in [S2_o, S2_f]
                   )

# plot the retained variance
ret_var       = [S2_o, S2_f]
ret_var_names = ['orig', 'fav']

for n in range(len(ret_var)):
    _, ax = plt.subplots(1, 1, figsize=(6,5))
    
    sns.lineplot(x=np.arange(len(ret_var[n])),
                 y=ret_var[n],
                 ax=ax
                )
    ax.set(title='SVD - Variance Retained per Component',
           xlabel='component',
           ylabel='fraction of variance'
          )
    
    plt.tight_layout()
    plt.savefig('./img/svd_ret_{}.pdf'.format(ret_var_names[n]), dpi=150, format='pdf')
    plt.savefig('./img/svd_ret_{}.png'.format(ret_var_names[n]), dpi=150, format='png')

# plot the cumulative variance
cum_var       = [S2cum_o, S2cum_f]
cum_var_names = ['orig', 'fav']
threshold     = 0.99

for n in range(len(cum_var)):
    _, ax = plt.subplots(1, 1, figsize=(6,5))
    
    sns.lineplot(x=np.arange(len(cum_var[n])),
                 y=cum_var[n],
                 ax=ax
                )
    ax.set(title='SVD - Cumulative Variance Retained',
           xlabel='component',
           ylabel='fraction of variance'
          )
    
    # add lines where part of the total variance is retained
    ax.axhline(threshold,
               xmin=0,
               xmax=np.argmax(cum_var[n] >= threshold) / len(cum_var[n]),
               linestyle='--',
               linewidth=0.75,
               c='black'
              )
    ax.axvline(np.argmax(cum_var[n] >= threshold),
               ymin=0,
               ymax=threshold / ax.get_ylim()[1],
               linestyle='--',
               linewidth=0.75, c='black'
              )
    ax.text(np.argmax(cum_var[n] >= threshold) + 5, threshold - 0.05,
            '{}% of variance retained'.format(threshold * 100),
            fontweight='bold'
           )
    ax.text(np.argmax(cum_var[n] >= threshold) + 5, 0.1,
            '{:d} components'.format(np.argmax(cum_var[n] >= threshold) + 1),
            fontweight='bold'
           )
    
    plt.tight_layout()
    plt.savefig('./img/svd_cum_{}.pdf'.format(cum_var_names[n]), dpi=150, format='pdf')
    plt.savefig('./img/svd_cum_{}.png'.format(cum_var_names[n]), dpi=150, format='png')