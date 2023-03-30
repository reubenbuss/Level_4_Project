#from pstats import Stats
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
#import statsmodels.api as sm

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best_outliers_removed.csv")
rp_df = pd.read_csv(r'C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv')
rp_df = rp_df[['Species', '350', '353', '356', '359', '362', '365', '368', '371', '374', '377', '380', '383', '386', '389', '392', '395', '398', '401', '404', '407', '410', '413', '416', '419', '422', '425', '428', '431', '434', '437', '440', '443', '446', '449', '452', '455', '458', '461', '464', '467', '470', '473', '476', '479', '482', '485', '488', '491', '494', '497', '500', '503', '506', '509', '512', '515', '518', '521', '524', '527', '530', '533', '536', '539', '542', '545', '548', '551', '554', '557', '560', '563', '566', '569', '572', '575', '578', '581', '584', '587', '590', '593', '596', '599', '602', '605', '608', '611', '614', '617', '620', '623', '626', '629', '632', '635', '638', '641', '644', '647', '650', '653', '656', '659', '662', '665', '668', '671', '674', '677', '680', '683', '686', '689', '692', '695', '698', '701', '707', '713', '719', '725', '731', '737', '743', '749', '755', '761', '767', '773', '779', '785', '791', '797', '803', '809', '815', '821', '827', '833', '839', '845', '851', '857', '863', '869', '875', '881', '887', '893', '899', '905', '911', '917', '923', '929', '935', '941', '947', '953', '959', '965', '971', '977', '983', '989', '995']]
coffee_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision.csv")
mangrove_and_coffee_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_and_Coffee_data.csv")

#print(mangrove_and_coffee_df.Species.unique())

df_b = rp_df.query('Species == "Black"').iloc[:,1:]
df_r = rp_df.query('Species == "Red"').iloc[:,1:]
df_w = rp_df.query('Species == "White"').iloc[:,1:]
df_br = rp_df.query('Species == "Black" | Species == "Red"').iloc[:,1:]
df_bw = rp_df.query('Species == "Black" | Species == "White"').iloc[:,1:]
df_rw = rp_df.query('Species == "Red" | Species ==  "White"').iloc[:,1:]
f_all,_ = stats.f_oneway(df_b.to_numpy(),df_r.to_numpy(),df_w.to_numpy())
f_br_w,_ = stats.f_oneway(df_br.to_numpy(),df_w.to_numpy())
f_bw_r,_ = stats.f_oneway(df_bw.to_numpy(),df_r.to_numpy())
f_rw_b,_ = stats.f_oneway(df_rw.to_numpy(),df_b.to_numpy())



df_gc = coffee_df.query('Species == "green_control"').iloc[:,1:]
df_rust = coffee_df.query('Species == "Rust"').iloc[:,1:]
df_rc = coffee_df.query('Species == "Rust_Canopy"').iloc[:,1:]
df_ab = coffee_df.query('Species == "AribicavarGeisha"').iloc[:,1:]
df_gc_rust_rc = coffee_df.query('Species == "green_control" | Species == "Rust" | Species == "Rust_Canopy"').iloc[:,1:]
df_gc_rust_ab = coffee_df.query('Species == "green_control" | Species == "Rust"  | Species == "AribicavarGeisha"').iloc[:,1:]
df_gc_rc_ab = coffee_df.query('Species == "green_control" | Species ==  "Rust_Canopy"  | Species == "AribicavarGeisha"').iloc[:,1:]
df_rust_rc_ab = coffee_df.query('Species == "Rust" | Species ==  "Rust_Canopy"  | Species == "AribicavarGeisha"').iloc[:,1:]
f_cof_all,_ = stats.f_oneway(df_gc.to_numpy(),df_rust.to_numpy(),df_rc.to_numpy(),df_ab.to_numpy())
f_cof_ab,_ = stats.f_oneway(df_gc_rust_rc.to_numpy(),df_ab.to_numpy())
f_cof_rc,_ = stats.f_oneway(df_gc_rust_ab.to_numpy(),df_rc.to_numpy())
f_cof_rust,_ = stats.f_oneway(df_gc_rc_ab.to_numpy(),df_rust.to_numpy())
f_cof_gc,_ = stats.f_oneway(df_rust_rc_ab.to_numpy(),df_gc.to_numpy())



df_mc_b = mangrove_and_coffee_df.query('Species == "Black"').iloc[:,1:]
df_mc_r = mangrove_and_coffee_df.query('Species == "Red"').iloc[:,1:]
df_mc_w = mangrove_and_coffee_df.query('Species == "White"').iloc[:,1:]
df_mc_gc = mangrove_and_coffee_df.query('Species == "green_control"').iloc[:,1:]
df_mc_rust = mangrove_and_coffee_df.query('Species == "Rust"').iloc[:,1:]
df_mc_rc = mangrove_and_coffee_df.query('Species == "Rust_Canopy"').iloc[:,1:]
df_mc_ab = mangrove_and_coffee_df.query('Species == "AribicavarGeisha"').iloc[:,1:]
df_mc__b = mangrove_and_coffee_df.query('Species != "Black"').iloc[:,1:]
df_mc__r = mangrove_and_coffee_df.query('Species != "Red"').iloc[:,1:]
df_mc__w = mangrove_and_coffee_df.query('Species != "White"').iloc[:,1:]
df_mc__gc = mangrove_and_coffee_df.query('Species != "green_control"').iloc[:,1:]
df_mc__rust = mangrove_and_coffee_df.query('Species != "Rust"').iloc[:,1:]
df_mc__rc = mangrove_and_coffee_df.query('Species != "Rust_Canopy"').iloc[:,1:]
df_mc__ab = mangrove_and_coffee_df.query('Species != "AribicavarGeisha"').iloc[:,1:]
f_mc_all,_ = stats.f_oneway(df_mc_b.to_numpy(),df_mc_r.to_numpy(),df_mc_w.to_numpy(),df_mc_gc.to_numpy(),df_mc_rust.to_numpy(),df_mc_rc.to_numpy(),df_mc_ab.to_numpy())
f_mc_b,_ = stats.f_oneway(df_mc_b.to_numpy(),df_mc__b.to_numpy())
f_mc_r,_ = stats.f_oneway(df_mc_r.to_numpy(),df_mc__r.to_numpy())
f_mc_w,_ = stats.f_oneway(df_mc_w.to_numpy(),df_mc__w.to_numpy())
f_mc_gc,_ = stats.f_oneway(df_mc_gc.to_numpy(),df_mc__gc.to_numpy())
f_mc_rust,_ = stats.f_oneway(df_mc_rust.to_numpy(),df_mc__rust.to_numpy())
f_mc_rc,p_rc = stats.f_oneway(df_mc_rc.to_numpy(),df_mc__rc.to_numpy())
f_mc_ab,_ = stats.f_oneway(df_mc_ab.to_numpy(),df_mc__ab.to_numpy())


cols = list(map(int,rp_df.columns.tolist()[1:]))
cols_coffee = list(map(int,coffee_df.columns.tolist()[1:]))
cols_mc = list(map(int,mangrove_and_coffee_df.columns.tolist()[1:]))


plt.figure(dpi=300,figsize=(8,4))

# plt.plot(cols_mc,f_mc_all,c='grey',label='ANOVA')
# plt.plot(cols_mc,f_mc_gc,c='blue',label='Green Control')
# plt.plot(cols_mc,f_mc_b,c='black',label='Black Mangrove')
# plt.plot(cols_mc,f_mc_rc,c='lawngreen',label='Rust Canopy')
# plt.plot(cols_mc,f_mc_r,c='red',label='Red Mangrove')
# plt.plot(cols_mc,f_mc_ab,c='cyan',label='Geisha')
# plt.plot(cols_mc,f_mc_w,c='green',label='White Mangrove')
# plt.plot(cols_mc,f_mc_rust,c='magenta',label='Rust')

# plt.plot(cols,f_all,c='grey',label='ANOVA')
# plt.plot(cols,f_br_w,c='green',label='White Mangrove')
# plt.plot(cols,f_bw_r,c='red',label='Red Mangrove')
# plt.plot(cols,f_rw_b,c='black',label='Black Mangrove')

plt.plot(cols_coffee,f_cof_all,c='grey',label='ANOVA')
plt.plot(cols_coffee,f_cof_ab,c='cyan',label='Geisha')
plt.plot(cols_coffee,f_cof_rc,c='lawngreen',label='Rust Canopy')
plt.plot(cols_coffee,f_cof_rust,c='magenta',label='Rust')
plt.plot(cols_coffee,f_cof_gc,c='blue',label='Green Control')


plt.xlabel('Wavelength (nm)')
plt.ylabel('F Statistic')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
    fancybox=True, shadow=True, ncol=4)
plt.show()

print('Finished')


