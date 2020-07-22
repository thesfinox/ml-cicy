#!/usr/bin/env python
# coding: utf-8


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set()

data = {'he_reg': 0.37,
        'bull_reg': 0.75,
        'bull_class': 0.85,
        'inception_30': 0.97,
        'inception_80': 0.99,
       }
data = pd.DataFrame(data, index=[0])

labels = ['He (reg., 63%)',
          'Bull et al. (reg., 70%)',
          'Bull et al. (class., 70%)',
          'Inception (30%)',
          'Inception (80%)'
         ]

_, ax = plt.subplots(1, 1, figsize=(6,5))

plot = sns.barplot(data=data,
                   color=sns.color_palette('muted',1)[0],
                   orient='h',
                   ax=ax
                  )
ax.set_xlim([0.0, 1.0])
ax.set_yticklabels(labels)
for p in plot.patches:
    plot.annotate('{:d}%'.format(int(p.get_width() * 100)),
                  xy=(p.get_width() + 0.01,
                      p.get_y() + p.get_height() / 2
                     ),
                  ha='left',
                  va='center'
                 )

plt.tight_layout()
plt.savefig('summary_hor_nolrsvr.pdf', dpi=150, format='pdf')


_, ax = plt.subplots(1, 1, figsize=(6,5))

plot = sns.barplot(data=data,
                   color=sns.color_palette('muted',1)[0],
                   orient='v',
                   ax=ax
                  )
ax.set_ylim([0.0, 1.0])
ax.set_yticks(np.arange(0.1, 1.0 + 0.1, 0.1))
ax.set_xticklabels(labels,
                   rotation=30,
                   va='top',
                   ha='right'
                  )
for p in plot.patches:
    plot.annotate('{:d}%'.format(int(p.get_height() * 100)),
                  xy=(p.get_x() + p.get_width() / 2,
                      p.get_height()
                     ),
                  ha='center',
                  va='bottom'
                 )

plt.tight_layout()
plt.savefig('summary_ver_nolrsvr.pdf', dpi=150, format='pdf')
