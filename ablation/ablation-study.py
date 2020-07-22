#!/usr/bin/env python
# coding: utf-8


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set()

data = [('inception_1d_noout', '30%', 0.98),
        ('inception_1d_out', '30%', 0.94),
        ('inception_2d_noout', '30%', 0.74),
        ('inception_2d_out', '30%', 0.67),
        ('conv_noout', '30%', 0.84),
        ('conv_out', '30%', 0.77),
        ('inception_1d_noout', '80%', 1.00),
        ('inception_1d_out', '80%', 0.97),
        ('inception_2d_noout', '80%', 0.86),
        ('inception_2d_out', '80%', 0.81),
        ('conv_noout', '80%', 0.93),
        ('conv_out', '80%', 0.90)
        ]
data = pd.DataFrame(data, columns=('id', 'percentage', 'accuracy'))

labels = ['Inception, 1d, no outliers',
          'Inception, 1d, outliers',
          'Inception, 2d, no outliers',
          'Inception, 2d, outliers',
          'convnet, no outliers',
          'convnet, outliers'
         ]

_, ax = plt.subplots(1, 1, figsize=(7, 5))

plot = sns.barplot(x='id', hue='percentage', y='accuracy', data=data,
                   orient='v',
                   ax=ax
                  )
ax.set_xlabel("")
ax.set_yticks(np.arange(0.0, 1.0 + 0.1, 0.1))
ax.set_ylim([0.0, 1.0])
ax.set_xticklabels(labels,
                   rotation=45,
                   va='top',
                   ha='right'
                  )
ax.legend(bbox_to_anchor=(0.0, 0.0),
          loc="lower left",
          title="training data",
          ncol=1
         )
for p in plot.patches:
    plot.annotate('{:d}%'.format(int(p.get_height() * 100)),
                  xy=(p.get_x() + p.get_width() / 2,
                      p.get_height()
                     ),
                  ha='center',
                  va='bottom', fontsize=10,
                 )

plt.tight_layout()
plt.savefig('ablation.pdf', dpi=150, format='pdf')
