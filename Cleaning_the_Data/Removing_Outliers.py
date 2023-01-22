import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack.csv"
df = pd.read_csv(file)
df_new = df.iloc[:,15:]
df_new.insert(loc=0,column="Species",value=df["Species"])
for i in range(0,df_new.shape[0]):
    if df_new.iat[i,0] == "Black":
        df_new.iat[i,0] = 0.1
    if df_new.iat[i,0] == "Red":
        df_new.iat[i,0] = 0.5
    if df_new.iat[i,0] == "White":
        df_new.iat[i,0] = 0.9

print(list(df_new.iloc[:,0]))
df_new = df_new[(df_new < 1).all(axis=1)] #remove any rows which have a value > 1
df_new = df_new[(df_new > 0).all(axis=1)] #remove any rows which have a value < 0
print(list(df_new.iloc[:,0]))
for i in range(0,df_new.shape[0]):
    if df_new.iat[i,0] == 0.1:
        df_new.iat[i,0] = "Black"
    if df_new.iat[i,0] == 0.5:
        df_new.iat[i,0] = "Red"
    if df_new.iat[i,0] == 0.9:
        df_new.iat[i,0] = "White"
print(list(df_new.iloc[:,0]))


df_new.to_csv("RedWhiteBlack Non Erroneous Data.csv",index=False)
print("Finished")
