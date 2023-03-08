import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData.csv")
rwb_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")
clean_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best.csv")
labels = {"Black": "black", "White": "green",
          "Red": "red", "na": 'blue', 'Mud': 'brown'}
all_df = all_df.drop(["SpectraID", 'WhiteReference', 'ContactProbe', 'FibreOptic', 'SPAD_1', 'SPAD_2', 'SPAD_3',
                     'SPAD_Ave', 'Location', 'Lat', 'Long', 'StandAge', 'StandHealth', 'SurfaceDescription'], axis=1, inplace=False)
all_df = all_df.sample(frac=1)
rwb_df = rwb_df.sample(frac=1)
clean_df = clean_df.sample(frac=1)

new_df = clean_df[['Species','395','749','1427','2232']]


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


c_cols, c_vals, c_colours = list_maker(new_df)
rwb_cols, rwb_vals, rwb_colours = list_maker(rwb_df)
all_cols, all_vals, all_colours = list_maker(all_df)

fig = plt.figure(figsize=(12, 9), dpi=480)
ax1 = fig.add_axes([0.1, 0.7, 0.8, 0.3], ylim=(0, 1),xlim=(250,2600))
ax2 = fig.add_axes([0.1, 0.35, 0.8, 0.3], ylim=(0, 1),xlim=(250,2600))
ax3 = fig.add_axes([0.1, 0, 0.8, 0.3], ylim=(0, 1),xlim=(250,2600))

ax3.scatter(0, 2, c="red", label="Red Mangroves")
ax3.scatter(0.2, 2, c="green", label="White Mangroves")
ax3.scatter(0, 2, c="black", label="Black Mangroves")
ax3.scatter(0, 2, c="brown", label="Mud")
ax3.scatter(0, 2, c="blue", label='Calibration')

ax3.scatter(c_cols, c_vals, s=1, fc=c_colours,edgecolors="none")
ax2.scatter(rwb_cols, rwb_vals, s=1, fc=rwb_colours,edgecolors="none")
ax1.scatter(all_cols, all_vals, s=1, fc=all_colours,edgecolors="none")


ax1.text(1,1, 'All Data', size=20, transform=ax1.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax2.text(1, 1, 'Non Erroneous Data', size=20, transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax3.text(1,1, 'Precision Matched Data', size=20, transform=ax3.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))

fig.text(0.5, -0.05, 'Wavelength (nm)', ha='center', va='center', size=15)
fig.text(0.06, 0.5, 'Reflectance', ha='center',
         va='center', rotation='vertical', size=15)
ax3.legend(bbox_to_anchor=(0.98, -0.2), ncol=5, fontsize=12)
#plt.show()
plt.savefig(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Graphs\triple plot v5 new swag.svg")
print("Finished")
