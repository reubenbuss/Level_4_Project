import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat\Wheat_data_all_headers.csv"
df = pd.read_csv(file)
days = {"2 dpi":0.05,"3 dpi":0.1,"4 dpi":0.15,"5 dpi":0.2,"6 dpi":0.25,"7 dpi":0.3,
        "8 dpi":0.35,"9 dpi":0.4,"10 dpi":0.45,"11 dpi":0.5,"12 dpi":0.55,"13 dpi":0.6}
days_unpacked = dict(zip(days.values(), days.keys()))
condition = {"healthy":0.2,"inf":0.4,"infected":0.4}
condition_unpacked = dict(zip(condition.values(), condition.keys()))
print(df.iat[1,0])
print(df.iat[1,1])
df.iat[1,0] = days[df.iat[1,0]]
df.iat[1,1] = condition[df.iat[1,1]]
print(df.iat[1,0])
print(df.iat[1,1])
df.iat[1,0] = days_unpacked[df.iat[1,0]]
df.iat[1,1] = condition_unpacked[df.iat[1,1]]
print(df.iat[1,0])
print(df.iat[1,1])

for i in range(0,df.shape[0]):
    df.iat[i,0] = days[df.iat[i,0]]
    df.iat[i,1] = condition[df.iat[i,1]]
print(df.head())

df = df[(df < 1).all(axis=1)] #remove any rows which have a value > 1
df = df[(df > 0).all(axis=1)] #remove any rows which have a value < 0

#all data removed no samples valid

for i in range(0,df.shape[0]):
    df.iat[i,0] = days_unpacked[df.iat[i,0]]
    df.iat[i,1] = condition_unpacked[df.iat[i,1]]

#df.to_csv("RedWhiteBlack Non Erroneous Data.csv",index=False)
print(df.head())
print("Finished")
