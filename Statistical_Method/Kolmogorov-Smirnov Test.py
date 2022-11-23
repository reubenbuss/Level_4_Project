#from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
#import statsmodels.api as sm

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black.csv"
df = pd.read_csv(file)

fig = plt.figure(dpi=2400)
df = df.to_numpy()
for i in range(15,2165):
    x = list(df[:,i])
    result = stats.kstest(x,"norm")[1]
    j = i+350
    if result < 0.05:
        plt.scatter(j,j,s=0.01,c = "green")
    else:
        plt.scatter(j,j,s=0.01,c = "red")

plt.scatter(1,1,c = "green",label = "Normally Distributed")
plt.scatter(1,1,c = "red", label = "Not Normally Distributed")
plt.xlabel("Wavelengths (nm)")
plt.ylabel("Wavelengths (nm)")
plt.legend(loc="upper left",markerscale = 1)        
plt.title("Black Mangrove Normally Distributed")
plt.ylim(350,2500)
plt.xlim(350,2500)
fig.savefig('Black Mangrove Normally Distributed.png', dpi=fig.dpi)
plt.show()
    