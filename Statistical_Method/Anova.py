#from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
#import statsmodels.api as sm

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best_outliers_removed.csv")

df_b = rp_df.query('Species == "Black"').iloc[:,1:]
df_r = rp_df.query('Species == "Red"').iloc[:,1:]
df_w = rp_df.query('Species == "White"').iloc[:,1:]
f_b,p_b = stats.f_oneway(*df_b.to_numpy())
f_r,p_r = stats.f_oneway(*df_r.to_numpy())
f_w,p_w = stats.f_oneway(*df_w.to_numpy())
f_all,p_all = stats.f_oneway(df_b.to_numpy(),df_r.to_numpy(),df_w.to_numpy())
cols = list(map(int,rp_df.columns.tolist()[1:]))
f_all = [x/(max(list(f_all))) for x in list(f_all)]
plt.plot(cols,f_all,c='g')
plt.plot(cols,p_all,c='b')
plt.show()
