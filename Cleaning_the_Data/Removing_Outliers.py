import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Calibrated_data.csv"

df = pd.read_csv(file)
df_new = df.copy()
#convert labels to floats
for i in range(0,df_new.shape[0]):
    if df_new.iat[i,0] == "Black":
        df_new.iat[i,0] = 0.1
    if df_new.iat[i,0] == "Red":
        df_new.iat[i,0] = 0.2
    if df_new.iat[i,0] == "White":
        df_new.iat[i,0] = 0.3

#remove outliers
df_new = df_new.astype(float)
df_new = df_new[(df_new < 1).all(axis=1)] #remove any rows which have a value > 1
df_new = df_new[(df_new > 0).all(axis=1)] #remove any rows which have a value < 0
#df_new = df_new[(np.abs(stats.zscore(df_new)) < 3).all(axis=1)]

#return floats to labels

for i in range(0,df_new.shape[0]):
    if df_new.iat[i,0] == 0.1:
        df_new.iat[i,0] = "Black"
    if df_new.iat[i,0] == 0.2:
        df_new.iat[i,0] = "Red"
    if df_new.iat[i,0] == 0.3:
        df_new.iat[i,0] = "White"

df_new.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Calibrated_data_outliers_removed.csv",index=False)
print("Finished")
