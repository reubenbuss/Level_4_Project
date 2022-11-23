import pandas as pd
from matplotlib import pyplot as plt

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData_Edited.csv"
df = pd.read_csv(file)
df.head()

'''
for j in range(1,50):
    species = df.iloc[j,4]
    if species == "White":
        colour = "white"
    elif species == "Red":
        colour = "red"
    elif species == "Black":
        colour = "black"
    else:
        colour = "blue"
    y=[]
    x=[]
    for i in range(350,2166):
        y.append(df.loc[j,str(i)])
        x.append(i+335)
        plt.scatter(x,y,color=colour,s=1)
'''



"""
y=[]
x=[]
for i in range(350,2166):
    y.append(df.loc[1,str(i)])
    x.append(i+335)
    plt.scatter(x,y,color="red")
"""

print("hello")


plt.show()