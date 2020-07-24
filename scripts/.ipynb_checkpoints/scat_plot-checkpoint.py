import numpy             as np
import seaborn           as sns
import matplotlib        as mpl
import matplotlib.pyplot as plt
import pandas            as pd
import os

# prepare common settings
sns.set()
os.makedirs('./img', exist_ok=True)

# define a function to extract the information
def get_unique(df: pd.DataFrame, feature: str, label: str) -> pd.DataFrame:
    '''
    Get the unique values and the no. of occurrencies.
    
    Required arguments:
        feature: the name of the feature,
        label:   the name of the label.
        
    Returns:
        a dataframe with the information.
    '''
    counts = df.groupby(feature)[label].value_counts()
    
    # store data in a dataframe
    data = {feature:      [c[0] for c in counts.index],
            label:        [c[1] for c in counts.index],
            'frequency':  counts.values
           }
    return pd.DataFrame(data)

# load the datasets
df_o = pd.read_csv('./data/cicy3o_tidy.csv')
df_f = pd.read_csv('./data/cicy3f_tidy.csv')

# define the features of the plot (only the relevant ones)
dat_title = [df_o, df_f]
dat_names = ['orig', 'fav']
lab_title = ['h11', 'h21']
lab_names = ['$h^{1,1}$', '$h^{2,1}$']
var_title = ['num_cp', 'rank_matrix', 'norm_matrix']
var_names = ['m', 'rank', 'norm']

# plot the distributions
for n in range(len(var_title)):
    for m in range(len(lab_title)):
        for p in range(len(dat_title)):
            _, ax = plt.subplots(1, 1, figsize=(6,5))

            sns.scatterplot(data=get_unique(dat_title[p],
                                            var_title[n],
                                            lab_title[m]
                                           ),
                            x=var_title[n],
                            y=lab_title[m],
                            size='frequency',
                            sizes=(25,200),
                            alpha=0.5,
                            ax=ax
                           )
            ax.set(xlabel=var_names[n],
                   ylabel=lab_names[m],
                   yticks=np.arange(min(dat_title[p][lab_title[m]]),
                                    max(dat_title[p][lab_title[m]])+1,
                                    int((max(dat_title[p][lab_title[m]])+1 -
                                         min(dat_title[p][lab_title[m]])
                                        )/10
                                       )
                                   )
                  )
            
            plt.tight_layout()
            plt.savefig('./img/scat_{}_{}_{}.pdf'.format(var_title[n],
                                                         lab_title[m],
                                                         dat_names[p]
                                                        ),
                        dpi=150,
                        format='pdf'
                       )
            plt.savefig('./img/scat_{}_{}_{}.png'.format(var_title[n],
                                                         lab_title[m],
                                                         dat_names[p]
                                                        ),
                        dpi=150,
                        format='png'
                       )