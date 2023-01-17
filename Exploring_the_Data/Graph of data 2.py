import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file1 = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData.csv"
df = pd.read_csv(file1)

file2 = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black Data Cleaned.csv"
df_Black = pd.read_csv(file2)
df_Black = df_Black.iloc[:,1:]

file3 = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\White Data Cleaned.csv"
df_White = pd.read_csv(file3)
df_White = df_White.iloc[:,1:]

file4 = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Red Data Cleaned.csv"
df_Red = pd.read_csv(file4)
df_Red = df_Red.iloc[:,1:]

df_new = df.iloc[:,15:]

# std = list(df_new.std())
# m = list(df_new.mean())
# m = np.array(m)
# std = np.array(std)

fig, (ax1, ax2) = plt.subplots(2,sharex=True,sharey=True,figsize=(10,6))
fig.suptitle('Reflectance Data Before and After Removing Erroneous Data',y=0.93)

x = list(range(350,2501))
#print(len(x))
for i in range(0,len(df.index)):
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
    y=df_new.values[i]
    ax1.set_ylim(0,1)
    ax1.set_xlim(300,2600)
    ax1.scatter(x,y,s=0.01,c=colour)

# for i in range(0,len(df_Black.index)-1):
#     y=df_Black.values[i]
#     ax2.set_ylim(0,1)
#     ax2.scatter(x,y,s=0.01,c="Black")

# for i in range(0,len(df_Red.index)-1):
#     y=df_Red.values[i]
#     ax2.set_ylim(0,1)
#     ax2.scatter(x,y,s=0.01,c="Red")

# for i in range(0,len(df_White.index)-1):
#     y=df_White.values[i]
#     ax2.set_ylim(0,1)
#     ax2.scatter(x,y,s=0.01,c="Green")

for i in range(0,len(df_Black.index)):
    ax2.set_ylim(0,1)
    y_B = df_Black.values[i]
    ax2.scatter(x,y_B,s=0.01,c="Black")
    B_W = len(df_Black.index) - len(df_White.index)
    if i > B_W:
        y_W = df_White.values[i-B_W]
        ax2.scatter(x,y_W,s=0.01,c="Green")
    B_R = len(df_Black.index) - len(df_Red.index)
    if i > B_R:
        y_R = df_Red.values[i-B_R]
        ax2.scatter(x,y_R,s=0.01,c="Red")    

# y = m+3*std
# print(len(x),len(y))
# plt.scatter(2,2,c=colour,label = Species + "Mangrove")
# plt.scatter(x,y,s=0.01,c = "yellow",label = "$\sigma$")
# y = m-3*std
# plt.scatter(x,y,s=0.01,c="yellow")
ax2.scatter(2,2,c="Green", label = "White Mangrove")
ax2.scatter(2,2,c="Red",label = "Red Mangrove")
ax2.scatter(2,2,c="Black",label="Black Mangrove")
ax2.scatter(2,2,c="blue", label = "Unknown")
ax2.scatter(2,2,c="brown" ,label = "Mud")

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Wavelengths (nm)")
plt.ylabel("Reflectance")
ax2.legend(loc="upper right")
fig.savefig('Reflectance Data Before and After Removing Erroneous Data.png', dpi=2400)
plt.show()
