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
#fig.set_facecolor("paleturquoise")
#ax = plt.axes()
#ax.set_facecolor("paleturquoise")

x = list(range(350,2501))

plt.scatter(x,Black_m,s=0.01,c="Black",label = "Black")
plt.errorbar(x, Black_m, yerr=Black_std,elinewidth=0.01,c="Black")
plt.scatter(x,White_m,s=0.01,c="green",label = "White")
plt.errorbar(x, White_m, yerr=White_std,elinewidth=0.01,c="green")
plt.scatter(x,Red_m,s=0.01,c="Red",label = "Red")
plt.errorbar(x, Red_m, yerr=Red_std,elinewidth=0.01,c="Red")
plt.legend(loc="upper right",markerscale = 100)
plt.title("Mean with Standard Deviation of Black, Red, White Mangrove Tree Reflectance Data",wrap=True)
fig.savefig('Mean with Standard Deviation of Black, Red, White Mangrove Tree Reflectance Data.png', dpi=fig.dpi)
plt.show()