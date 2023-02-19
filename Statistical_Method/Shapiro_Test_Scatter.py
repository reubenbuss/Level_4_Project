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

fig = plt.figure(figsize = (8,2),dpi=2400)
length_b = len(df_b.columns)

for i in range(15,length_b):
    b = list(df_b.iloc[:,i])

    result_b = stats.shapiro(b)[1]
    j = i+350
    if result_b > 0.05:
        plt.scatter(j,1.05,s=1,fc = "black",edgecolors=None,marker='s')
    else:
        plt.scatter(j,0.05,s=1,fc = "black",edgecolors=None,marker='s')

length_r = len(df_r.columns)
for i in range(15,length_r):
    r = list(df_r.iloc[:,i])

    result_r = stats.shapiro(r)[1]
    j = i+350
    if result_r > 0.05:
        plt.scatter(j,1,s=1,fc = "red",edgecolors=None,marker='s')
    else:
        plt.scatter(j,0,s=1,fc = "red",edgecolors=None,marker='s')

length_w = len(df_w.columns)
for i in range(15,length_w):
    w = list(df_w.iloc[:,i])

    result_w = stats.shapiro(w)[1]
    j = i+350
    if result_w > 0.05:
        plt.scatter(j,0.95,s=1,fc = "green",edgecolors=None,marker='s')
    else:
        plt.scatter(j,-0.05,s=1,fc = "green",edgecolors=None,marker='s')

new_cols = list(range(350, 702, 3)) + list(range(707, 1398, 6)) + \
    list(range(1407, 2098, 10)) + list(range(2112, 2488, 15)) + [2501]

plt.plot([3,4],[3,4],color = "black",label = "Black Mangrove")
plt.plot([3,4],[3,4],color = "red",label = "Red Mangrove")
plt.plot([3,4],[3,4],color = "green",label = "White Mangrove")
plt.xlabel("Wavelengths (nm)")
#plt.ylabel("Wavelengths (nm)")
#plt.legend(loc="upper left",markerscale = 1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=False, ncol=3)
plt.title("Shapiro–Wilk Test on Reduced Precision Data")
plt.ylim(-0.25,1.25)
plt.yticks([0,1],["Not Normally Distributed", "Normally Distributed"])
plt.xticks(range(350,682,50),new_cols[0::50])
plt.xlim(350,682)
#fig.savefig('Shapiro–Wilk on Cleaned Black Mangrove Data.png', dpi=fig.dpi)
plt.show()
