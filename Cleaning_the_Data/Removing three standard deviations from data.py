#from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import random
#import statsmodels.api as sm



file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black.csv"
df = pd.read_csv(file)
df = df.iloc[:,15:]

df =df[(np.abs(df)<1).all(axis=1)]
df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]

# for i in range(15,2165):
#     df_filtered = df[(df.iloc[:,i] < 1) & (df.iloc[:,i] > 0)]
#     m = df_filtered.iloc[:,i].mean()
#     std  = df_filtered.iloc[:,i].std()
#     q_hi = m+3*std
#     q_low = m-3*std
#     df_final = df_filtered[(df_filtered.iloc[:,i] < q_hi) & (df_filtered.iloc[:,i] > q_low)]


df.to_csv('Black Filtered.csv', index=False)
print("finished")