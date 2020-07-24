import numpy             as np
import seaborn           as sns
import matplotlib        as mpl
import matplotlib.pyplot as plt
import pandas            as pd
import os
import json

# prepare common settings
sns.set()
os.makedirs('./img', exist_ok=True)

# load the datasets
df_o = pd.read_csv('./data/cicy3o_tidy.csv')
df_f = pd.read_csv('./data/cicy3f_tidy.csv')

# load the shapes
with open('./data/cicy3o_shapes.json', 'r') as o:
    sh_o = json.load(o)
    
with open('./data/cicy3f_shapes.json', 'r') as f:
    sh_f = json.load(f)

# select the columns
col = [key for key, value in sh_f.items()
                          if np.prod(value) == 1
                          and key != 'isprod'
                          and key != 'favour'
                          and key != 'kahlerpos'
      ]

# plot the correlation matrix
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.heatmap(df_o[col].corr(),
            vmin=-1.0,
            vmax=1.0,
            cmap='RdBu_r',
            ax=ax
           )

plt.tight_layout()
plt.savefig('./img/corr_mat_orig.pdf', dpi=150, format='pdf')
plt.savefig('./img/corr_mat_orig.png', dpi=150, format='png')

_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.heatmap(df_f[col].corr(),
            vmin=-1.0,
            vmax=1.0,
            cmap='RdBu_r',
            ax=ax
           )

plt.tight_layout()
plt.savefig('./img/corr_mat_fav.pdf', dpi=150, format='pdf')
plt.savefig('./img/corr_mat_fav.png', dpi=150, format='png')