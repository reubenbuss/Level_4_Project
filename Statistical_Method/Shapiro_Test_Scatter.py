#from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
#import statsmodels.api as sm

file_b = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black Data Cleaned.csv"
df_b = pd.read_csv(file_b)
file_r = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Red Data Cleaned.csv"
df_r = pd.read_csv(file_r)
file_w = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\White Data Cleaned.csv"
df_w = pd.read_csv(file_w)

fig = plt.figure(figsize = (8,2),dpi=2400)
length_b = len(df_b.columns)

for i in range(15,length_b):
    b = list(df_b.iloc[:,i])

    result_b = stats.shapiro(b)[1]
    j = i+350
    if result_b > 0.05:
        plt.scatter(j,1.05,s=0.01,c = "black",marker = "s")
    else:
        plt.scatter(j,0.05,s=0.01,c = "black",marker = "s")

length_r = len(df_r.columns)
for i in range(15,length_r):
    r = list(df_r.iloc[:,i])

    result_r = stats.shapiro(r)[1]
    j = i+350
    if result_r > 0.05:
        plt.scatter(j,1,s=0.01,c = "red",marker = "s")
    else:
        plt.scatter(j,0,s=0.01,c = "red",marker = "s")

length_w = len(df_w.columns)
for i in range(15,length_w):
    w = list(df_w.iloc[:,i])

    result_w = stats.shapiro(w)[1]
    j = i+350
    if result_w > 0.05:
        plt.scatter(j,0.95,s=0.01,c = "green",marker = "s")
    else:
        plt.scatter(j,-0.05,s=0.01,c = "green",marker = "s")


plt.plot([3,4],[3,4],color = "black",label = "Black Mangrove")
plt.plot([3,4],[3,4],color = "red",label = "Red Mangrove")
plt.plot([3,4],[3,4],color = "green",label = "White Mangrove")
plt.xlabel("Wavelengths (nm)")
#plt.ylabel("Wavelengths (nm)")
#plt.legend(loc="upper left",markerscale = 1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=False, ncol=3)
plt.title("Shapiro–Wilk Test")
plt.ylim(-0.25,1.25)
plt.yticks([0,1],["Not Normally Distributed", "Normally Distributed"])
plt.xlim(300,2500)
fig.savefig('Shapiro–Wilk on Cleaned Black Mangrove Data.png', dpi=fig.dpi)
plt.show()

