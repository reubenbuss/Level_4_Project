import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

Black_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black Data Cleaned.csv"
White_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\White Data Cleaned.csv"
Red_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Red Data Cleaned.csv"

Black_df = pd.read_csv(Black_file)
White_df = pd.read_csv(White_file)
Red_df = pd.read_csv(Red_file)

Black_df = Black_df.iloc[:,1:]
White_df = White_df.iloc[:,1:]
Red_df =   Red_df.iloc[:,1:]

Black_std = Black_df.std()
Black_m = Black_df.mean()
White_std = White_df.std()
White_m = White_df.mean()
Red_std = Red_df.std()
Red_m = Red_df.mean()

fig = plt.figure(dpi=2400)
fig.set_facecolor("paleturquoise")
ax = plt.axes()
ax.set_facecolor("paleturquoise")

x = list(range(350,2501))

plt.scatter(x,Black_m,s=0.01,c="Black",label = "Black Mean")
plt.scatter(x,White_m,s=0.01,c="white",label = "White Mean")
plt.scatter(x,Red_m,s=0.01,c="Red",label = "Red Mean")
plt.legend(loc="upper right",markerscale = 100,facecolor = "turquoise")
plt.title("Mean of Black, Red, White Mangrove Tree Reflectance Data")
fig.savefig('Mean of Black, Red, White Mangrove Tree Reflectance Data.png', dpi=fig.dpi)
plt.show()