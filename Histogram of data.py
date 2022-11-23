from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Red.csv"
df = pd.read_csv(file)

length = 250        #number of rows in dataset

Species = df.iloc[1,4]

df = df.to_numpy()
for i in range(1700,1701):
    x = df[:,i]
    #plt.xlim([-0.2,0.2])
    m = np.mean(x)
    std = np.std(x)
    y = np.arange(-1,1,0.001)
    sm.qqplot(x,line="s")
    #plt.plot(y,stats.norm.pdf(y,0,std),label = "Normal",c = "crimson")
    #plt.hist(x-m,bins = 20, density = True, label = "Histogram",color = "teal")
        

plt.xlabel("Wavelengths (nm)")
plt.ylabel("Reflectance")
#plt.legend(loc="upper right",markerscale = 1)
#fig.savefig('White.png', dpi=fig.dpi)
plt.show()
