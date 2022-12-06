import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Red.csv"
df = pd.read_csv(file)
df = df.iloc[:,15:]

df = df[(df < 1).all(axis=1)] #remove any rows which have a value > 1
df = df[(df > 0).all(axis=1)] #remove any rows which have a value < 0

m=df.mean()
std = df.std()

df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] # remove any rows which have a value greater then 3 standard deviations from the mean
length = df.shape[0]

fig = plt.figure(dpi=2400)
x = list(range(350,2501))
for i in range(0,length):
    y=df.values[i]
    plt.ylim(0,1)
    plt.xlim(300,2550)
    plt.scatter(x,y,s=0.01,c="Green")


y=m+3*std
plt.scatter(x,y,s=0.01,c="Red",label = "Three Standard Deviations")
y=m-3*std
plt.scatter(x,y,s=0.01,c="Red")
plt.scatter(2,2,s=0.01,c="Green",label = "Red Mangrove Reflectance")
plt.xlabel("Wavelengths (nm)")
plt.ylabel("Reflectance")
plt.legend(loc="upper right",markerscale = 100)
plt.title("Red Mangrove Cleaned Data")
fig.savefig('Red Cleaned.png', dpi=fig.dpi)
plt.show()

df.to_csv("Red Data Cleaned.csv")
print("Finished")
