import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set()

data = [('lin_reg', '80', 0.51, 0.11),
        ('lin_reg', '30', 0.49, 0.10),
        ('lin_svr', '80', 0.61, 0.11),
        ('lin_svr', '30', 0.61, 0.10),
        ('svr_rbf', '80', 0.70, 0.22),
        ('svr_rbf', '30', 0.58, 0.16),
        ('rnd_for', '80', 0.55, 0.12),
        ('rnd_for', '30', 0.54, 0.11),
        ('grd_bst', '80', 0.50, 0.14),
        ('grd_bst', '30', 0.46, 0.10),
        ('convnet', '80', 0.95, 0.36),
        ('convnet', '30', 0.83, 0.23),
        ('inception', '80', 0.99, 0.50),
        ('inception', '30', 0.97, 0.33),
       ]
data = pd.DataFrame(data, columns=('id', 'percentage', 'h11_acc', 'h21_acc'))

labels =['lin. reg.',
         'lin. SVM',
         'SVM (Gauss.)',
         'rand. for.',
         'grad. boost.',
         'ConvNet',
         'Inception'
        ]

_, ax = plt.subplots(1, 1, figsize=(12,10))

acc_plt = sns.barplot(data=data,
                      x='id',
                      y='h11_acc',
                      hue='percentage',
                      palette=sns.color_palette('Paired', 2),
                      orient='v',
                      ax=ax
                     )
acc_plt = sns.barplot(data=data,
                      x='id',
                      y='h21_acc',
                      hue='percentage',
                      palette=sns.color_palette('Paired_r', 2),
                      orient='v',
                      ax=ax
                     )
ax.set(title='',
       xlabel='',
       ylabel='accuracy',
       ylim=[0.0, 1.0],
       yticks=np.arange(0.0, 1.0 + 0.1, 0.1)
      )
ax.set_xticklabels(labels,
                   rotation=45,
                   va='top',
                   ha='right'
                  )
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles,
          ['$h^{1,1}$ (30% ratio)',
           '$h^{1,1}$ (80% ratio)',
           '$h^{2,1}$ (30% ratio)',
           '$h^{2,1}$ (80% ratio)'
          ],
          loc='best',
         )

for p in acc_plt.patches:
    acc_plt.annotate('{:d}%'.format(int(p.get_height() * 100)),
                     xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                     ha='center',
                     va='bottom'
                    )
    
plt.tight_layout()
plt.savefig('./cicy_matrix_plots.pdf', dpi=150, format='pdf')
plt.savefig('./cicy_matrix_plots.png', dpi=150, format='png')