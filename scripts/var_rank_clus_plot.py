import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

# set argument parser
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, help='path to the CSV file of the variable ranking')
parser.add_argument('-o', '--output', type=str, help='base name of the output files')

args = parser.parse_args()

# set common options
sns.set()
os.makedirs('./img', exist_ok=True)

# load the dataset
df = pd.read_csv(args.input)

# plot the results
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(x='feature',
             y='value',
             hue='hodge',
             style='hodge',
             markers=True,
             sort=False,
             data=df,
             ax=ax
            )
ax.set(xlabel='', ylabel='importance')
ax.legend(['$h^{1,1}$', '$h^{2,1}$'])
plt.xticks(rotation=45,
           va='top',
           ha='right'
          )

plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')