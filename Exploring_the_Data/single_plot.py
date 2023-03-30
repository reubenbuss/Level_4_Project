import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Calibrated_data_outliers_removed.csv")

labels = {"Black": "black", "White": "green",
          "Red": "red", "na": 'blue', 'Mud': 'brown'}

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


c_cols, c_vals, c_colours = list_maker(df)

plt.scatter(c_cols, c_vals, s=1, fc=c_colours,edgecolors="none")
plt.show()
