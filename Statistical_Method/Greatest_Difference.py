import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

Black_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black Data Cleaned.csv"
White_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\White Data Cleaned.csv"
Red_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Red Data Cleaned.csv"

df_Black = pd.read_csv(Black_file)
df_White = pd.read_csv(White_file)
df_Red = pd.read_csv(Red_file)

df_Black = df_Black.iloc[:,1:]
df_White = df_White.iloc[:,1:]
df_Red =   df_Red.iloc[:,1:]

#Black_std = Black_df.std()
Black_m = df_Black.mean()
#White_std = White_df.std()
White_m = df_White.mean()
#Red_std = Red_df.std()
Red_m = df_Red.mean()

fig = plt.figure(dpi=2400)
#fig.set_facecolor("paleturquoise")
#ax = plt.axes()
#ax.set_facecolor("paleturquoise")

x = list(range(350,2501))

for i in range(0,len(df_Black.index)):
    plt.ylim(0,1)
    y_B = df_Black.values[i]
    plt.scatter(x,y_B,s=0.001,c="Black",alpha=0.1)
    B_W = len(df_Black.index) - len(df_White.index)
    if i > B_W:
        y_W = df_White.values[i-B_W]
        plt.scatter(x,y_W,s=0.001,c="Green",alpha=0.1)
    B_R = len(df_Black.index) - len(df_Red.index)
    if i > B_R:
        y_R = df_Red.values[i-B_R]
        plt.scatter(x,y_R,s=0.001,c="Red",alpha=0.1) 

plt.scatter(x,Black_m,s=0.01,c="Black",label = "Black")
plt.scatter(x,White_m,s=0.01,c="green",label = "White")
plt.scatter(x,Red_m,s=0.01,c="Red",label = "Red")
plt.legend(loc="upper right",markerscale = 100)
plt.title("Mean with Standard Deviation of Black, Red, White Mangrove Tree Reflectance Data",wrap=True)
#fig.savefig('Mean with Standard Deviation of Black, Red, White Mangrove Tree Reflectance Data.png', dpi=fig.dpi)
plt.show()