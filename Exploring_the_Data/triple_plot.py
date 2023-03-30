import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData.csv")
rwb_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")
clean_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best.csv")
coffee_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision.csv")
coffee_df = coffee_df.sample(frac=1)
rp_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv")
rp_df = rp_df.sample(frac=1)
rp_df = rp_df[['Species', '350', '353', '356', '359', '362', '365', '368', '371', '374', '377', '380', '383', '386', '389', '392', '395', '398', '401', '404', '407', '410', '413', '416', '419', '422', '425', '428', '431', '434', '437', '440', '443', '446', '449', '452', '455', '458', '461', '464', '467', '470', '473', '476', '479', '482', '485', '488', '491', '494', '497', '500', '503', '506', '509', '512', '515', '518', '521', '524', '527', '530', '533', '536', '539', '542', '545', '548', '551', '554', '557', '560', '563', '566', '569', '572', '575', '578', '581', '584', '587', '590', '593', '596', '599', '602', '605', '608', '611', '614', '617', '620', '623', '626', '629', '632', '635', '638', '641', '644', '647', '650', '653', '656', '659', '662', '665', '668', '671', '674', '677', '680', '683', '686', '689', '692', '695', '698', '701', '707', '713', '719', '725', '731', '737', '743', '749', '755', '761', '767', '773', '779', '785', '791', '797', '803', '809', '815', '821', '827', '833', '839', '845', '851', '857', '863', '869', '875', '881']]
labels = {"Black": "black", "White": "green","Red": "red", "na": 'blue', 'Mud': 'brown','green_control':'blue','Rust':'magenta','Rust_Canopy':'lawngreen','AribicavarGeisha':'cyan'}
all_df = all_df.drop(["SpectraID", 'WhiteReference', 'ContactProbe', 'FibreOptic', 'SPAD_1', 'SPAD_2', 'SPAD_3',
                     'SPAD_Ave', 'Location', 'Lat', 'Long', 'StandAge', 'StandHealth', 'SurfaceDescription'], axis=1, inplace=False)
all_df = all_df.sample(frac=1)
rwb_df = rwb_df.sample(frac=1)
clean_df = clean_df.sample(frac=1)


#new_df = clean_df[['Species','395','749','1427','2232']]


def list_maker(df):
    '''
    Returns lists of wavelengths, reflectance value, and species colours
    '''
    value_list = []
    columns_list = []
    colour_list = []
    for i in range(0, df.shape[0]):
        value_list += (df.iloc[i, :].values.tolist()[1:])
        colour_list += ([labels[df.iloc[i, :].values.tolist()[0]]]
                        * (df.shape[1]-1))
        columns_list += (list(map(int, df.columns.tolist()[1:])))
    return columns_list, value_list, colour_list


#c_cols, c_vals, c_colours = list_maker(new_df)
#rwb_cols, rwb_vals, rwb_colours = list_maker(rwb_df)
coffee_cols, coffee_vals, coffee_colours = list_maker(coffee_df)
rp_cols, rp_vals, rp_colours = list_maker(rp_df)
#all_cols, all_vals, all_colours = list_maker(all_df)

fig = plt.figure(figsize=(12, 6), dpi=300)
#ax1 = fig.add_axes([0.1, 0.7, 0.8, 0.3], ylim=(0, 1),xlim=(250,2600))
#ax2 = fig.add_axes([0.1, 0.35, 0.8, 0.3], ylim=(0, 1),xlim=(250,2600))
#ax3 = fig.add_axes([0.1, 0, 0.8, 0.3], ylim=(0, 1),xlim=(250,2600))
ax3 = fig.add_axes([0.1,0.1,0.8,0.4],ylim=(0, 1),xlim=(340,900))
ax2 = fig.add_axes([0.1,0.5,0.8,0.4],ylim=(0, 1),xlim=(340,900),xticks=([]),yticks=([0.25,0.5,0.75,1]))

ax3.scatter(0, 2, c="blue", label="Green Coffee")
ax3.scatter(0.2, 2, c="cyan", label="Geisha")
ax3.scatter(0, 2, c="magenta", label="Coffee Rust")
ax3.scatter(0, 2, c="lawngreen", label="Rust Canopy")
ax3.scatter(0, 2, c="red", label="Red Mangroves")
ax3.scatter(0.2, 2, c="green", label="White Mangroves")
ax3.scatter(0, 2, c="black", label="Black Mangroves")
# ax3.scatter(0, 2, c="brown", label="Mud")
# ax3.scatter(0, 2, c="blue", label='Calibration')

#ax3.scatter(c_cols, c_vals, s=1, fc=c_colours,edgecolors="none")
ax3.scatter(rp_cols, rp_vals, s=1, fc=rp_colours,edgecolors="none") # bottom
ax2.scatter(coffee_cols, coffee_vals, s=1, fc=coffee_colours,edgecolors="none") #top
#ax1.scatter(all_cols, all_vals, s=1, fc=all_colours,edgecolors="none")


# ax1.text(1,1, 'All Data', size=20, transform=ax1.transAxes, ha="right", va="top",
#          bbox=dict(boxstyle="square",
#                    ec='black',
#                    fc='white',
#                    ))
# ax2.text(1, 1, 'Non Erroneous Data', size=20, transform=ax2.transAxes, ha="right", va="top",
#          bbox=dict(boxstyle="square",
#                    ec='black',
#                    fc='white',
#                    ))
# ax3.text(1,1, 'Precision Matched Data', size=20, transform=ax3.transAxes, ha="right", va="top",
#          bbox=dict(boxstyle="square",
#                    ec='black',
#                    fc='white',
#                    ))

fig.text(0.5, 0.04, 'Wavelength (nm)', ha='center', va='center', size=15)
fig.text(0.05, 0.5, 'Reflectance', ha='center',
         va='center', rotation='vertical', size=15)
handles, labels = ax3.get_legend_handles_labels()
order = [0,1,2,3,4,5,6]
order = [0,4,1,5,2,6,3]
ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(0.5, -0.32),loc='center', ncol=4, fontsize=12)
plt.show()
#plt.savefig(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Graphs\triple plot v5 new swag.svg")
print("Finished")
