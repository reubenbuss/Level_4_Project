#from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
#import statsmodels.api as sm

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best_outliers_removed.csv")

df_b = rp_df.query('Species == "Black"')
df_r = rp_df.query('Species == "Red"')
df_w = rp_df.query('Species == "White"')
colours = ['black','red','green']


fig = plt.figure(figsize = (8,2),dpi=300)
length_b = len(df_b.columns)


for i in list(df_b.columns)[1:]:
    f,p = stats.levene(df_b[i].tolist(), df_r[i].tolist(), df_w[i].tolist(), center='median')
    if p > 0.05:
        plt.scatter(int(i),1,c='green',marker ='s')
    else:
        plt.scatter(int(i),0,c='red',marker='s')


plt.xlabel("Wavelengths (nm)")
#plt.ylabel("Wavelengths (nm)")
#plt.legend(loc="upper left",markerscale = 1)
#plt.title("levene Test on Reduced Precision Data")
plt.ylim(-0.25,1.25)
plt.yticks([0,1],[r'$\sigma_{i} \ne \sigma_{j}$' '\n' r'$\forall i,j \in C$',r'$\sigma_{i} = \sigma_{j}$' '\n' r'$\forall i,j \in C$'])
#plt.xticks(range(350,682,50),new_cols[0::50])
#plt.xlim(350,682)
#fig.savefig('levene–Wilk on Cleaned Black Mangrove Data.png', dpi=fig.dpi)
plt.show()
