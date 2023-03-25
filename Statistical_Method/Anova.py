#from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
#import statsmodels.api as sm

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best_outliers_removed.csv")
rp_df = pd.read_csv(r'C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv')
df_b = rp_df.query('Species == "Black"').iloc[:,1:]
df_r = rp_df.query('Species == "Red"').iloc[:,1:]
df_w = rp_df.query('Species == "White"').iloc[:,1:]
df_br = rp_df.query('Species == "Black" | Species == "Red"').iloc[:,1:]
df_bw = rp_df.query('Species == "Black" | Species == "White"').iloc[:,1:]
df_rw = rp_df.query('Species == "Red" | Species ==  "White"').iloc[:,1:]

f_all,_ = stats.f_oneway(df_b.to_numpy(),df_r.to_numpy(),df_w.to_numpy())
f_b_r,_ = stats.f_oneway(df_b.to_numpy(),df_r.to_numpy())
f_b_w,_ = stats.f_oneway(df_b.to_numpy(),df_w.to_numpy())
f_r_w,_ = stats.f_oneway(df_r.to_numpy(),df_w.to_numpy())
f_br_w,_ = stats.f_oneway(df_br.to_numpy(),df_w.to_numpy())
f_bw_r,_ = stats.f_oneway(df_bw.to_numpy(),df_r.to_numpy())
f_rw_b,_ = stats.f_oneway(df_rw.to_numpy(),df_b.to_numpy())
cols = list(map(int,rp_df.columns.tolist()[1:]))
f_all = [x/(max(list(f_all))) for x in list(f_all)]
f_b_r = [x/(max(list(f_b_r))) for x in list(f_b_r)]
f_b_w = [x/(max(list(f_b_w))) for x in list(f_b_w)]
f_r_w = [x/(max(list(f_r_w))) for x in list(f_r_w)]
f_br_w = [x/(max(list(f_br_w))) for x in list(f_br_w)]
f_bw_r = [x/(max(list(f_bw_r))) for x in list(f_bw_r)]
f_rw_b = [x/(max(list(f_rw_b))) for x in list(f_rw_b)]

plt.scatter(cols,f_all,c='blue',marker=".")
#plt.scatter(cols,f_b_r,c='black',s=0.8,marker="|")
#plt.scatter(cols,f_b_w,c='red',s=0.8,marker="|")
#plt.scatter(cols,f_r_w,c='green',s=0.8,marker="|")
plt.scatter(cols,f_br_w,c='green',marker=".")
plt.scatter(cols,f_bw_r,c='red',marker=".")
plt.scatter(cols,f_rw_b,c='black',marker=".")

plt.show()

print('Finished')


