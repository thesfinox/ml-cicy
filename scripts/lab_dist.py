import numpy             as np
import seaborn           as sns
import matplotlib.pyplot as plt
import pandas            as pd
import os

# prepare common settings
sns.set()
os.makedirs('./img', exist_ok=True)

# load the dataset
lab = pd.read_csv('./data/labels.csv')

# plot the distribution
_, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.distplot(lab['h11'],
             kde=False,
             bins=np.arange(min(lab['h11']), max(lab['h11'])+1),
             ax=ax
            )
ax.set(xlabel='$h^{1,1}$',
       ylabel='count',
       xticks=np.arange(min(lab['h11']), max(lab['h11'])+1, 1),
       yscale='log'
      )

plt.tight_layout()
plt.savefig('./img/lab_h11_dist.pdf', dpi=150, format='pdf')
plt.savefig('./img/lab_h11_dist.png', dpi=150, format='png')

_, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.distplot(lab['h21'],
             kde=False,
             bins=np.arange(min(lab['h21']), max(lab['h21'])+1),
             ax=ax
            )
ax.set(xlabel='$h^{2,1}$',
       ylabel='count',
       xticks=np.arange(min(lab['h21']), max(lab['h21'])+1, 10),
       yscale='log'
      )

plt.tight_layout()
plt.savefig('./img/lab_h21_dist.pdf', dpi=150, format='pdf')
plt.savefig('./img/lab_h21_dist.png', dpi=150, format='png')