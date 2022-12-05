import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack.csv"
df = pd.read_csv(file)

length = 250        #number of rows in dataset

std = list(df.std())
std = std[2:]
m = list(df.mean())
m = m[2:]
m = np.array(m)
std = np.array(std)

fig = plt.figure(dpi=2400)
fgh = df.iloc[0:length,15:2166]
x = list(range(350,2501))
for i in range(0,length):
    species = df.iloc[i,4]
    if species == "White":
        colour = "green"
    elif species == "Red":
        colour = "red"
    elif species == "Black":
        colour = "black"
    elif species == "Mud":
        colour = "brown"
    else:
        colour = "blue"
    y=fgh.values[i]
    plt.ylim(0,1)
    plt.scatter(x,y,s=0.01,c=colour)


y = m+3*std
print(len(x),len(y))
Species = df.iloc[1,4]  
plt.scatter(2,2,s=0.01,c=colour,label = Species + " Mangrove")
plt.scatter(x,y,s=0.01,c = "yellow",label = "$\sigma$")
y = m-3*std
plt.scatter(x,y,s=0.01,c="yellow")
plt.xlabel("Wavelengths (nm)")
plt.ylabel("Reflectance")
plt.legend(loc="upper right",markerscale = 100)
fig.savefig('White.png', dpi=fig.dpi)
plt.show()
