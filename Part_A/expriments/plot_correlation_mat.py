import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/admission_predict.csv")

USED_data = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance of Admit']]

print(USED_data.head())
corr = USED_data.corr()
print(corr)
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()