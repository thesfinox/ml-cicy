#! /usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set()

# create data
data = {'type': ['Inception, 1d, no outliers (regr.)',
                 'Inception, 1d, no outliers (regr.)',
                 'Inception, 1d, no outliers (class.)',
                 'Inception, 1d, no outliers (class.)',
                 'Inception, 1d, outliers (regr.)',
                 'Inception, 1d, outliers (regr.)',
                 'Inception, 1d, outliers (class.)',
                 'Inception, 1d, outliers (class.)',
                 'Inception, 2d, no outliers (regr.)',
                 'Inception, 2d, no outliers (regr.)',
                 'Inception, 2d, no outliers (class.)',
                 'Inception, 2d, no outliers (class.)',
                 'Inception, 2d, outliers (regr.)',
                 'Inception, 2d, outliers (regr.)',
                 'Inception, 2d, outliers (class.)',
                 'Inception, 2d, outliers (class.)',
                 'Convnet, no outliers (regr.)',
                 'Convnet, no outliers (regr.)',
                 'Convnet, no outliers (class.)',
                 'Convnet, no outliers (class.)',
                 'Convnet, outliers (regr.)',
                 'Convnet, outliers (regr.)',
                 'Convnet, outliers (class.)',
                 'Convnet, outliers (class.)'
                ],
        'accuracy': [0.97,
                     0.99,
                     0.87,
                     0.95,
                     0.89,
                     0.97,
                     0.87,
                     0.95,
                     0.69,
                     0.86,
                     0.76,
                     0.88,
                     0.62,
                     0.78,
                     0.74,
                     0.87,
                     0.79,
                     0.93,
                     0.84,
                     0.92,
                     0.68,
                     0.85,
                     0.82,
                     0.84
                    ],
        'ratio': ['30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%',
                  '30%',
                  '80%'
                 ]
       }

# import to Pandas (in case we want to exclude some values it is easier to do it in Pandas than in a dict)
df = pd.DataFrame(data)

# plot the data
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))

sns.barplot(data=df,
            x='type',
            y='accuracy',
            hue='ratio',
            ax=ax
           )
ax.set(title='',
       xlabel='',
       ylabel='accuracy'
      )

# customise xticks
_, lab = plt.xticks()
ax.set_xticklabels(lab, rotation=45, ha='right', va='top')

# set legend outside for readability
ax.legend(title='training ratio', loc='lower left', bbox_to_anchor=(1.0, 0.0))

# add text on top of the bars and a different hatch for regression and classification
for i, p in enumerate(ax.patches):
    x   = p.get_x() + p.get_width() / 2
    y   = p.get_y() + p.get_height()
    txt = f'{int(100 * p.get_height()):d}%'

    ax.text(x, y, txt, ha='center', fontsize=10)


plt.tight_layout()
plt.savefig('ablation_study_v2.pdf', dpi=144, format='pdf')

