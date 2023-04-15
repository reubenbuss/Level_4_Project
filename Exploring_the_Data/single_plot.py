import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv(r"Your file directory goes here")
df = df.sample(frac=1) # random shuffle

print(df.head())

# This is the colours you want to correspond with each species. Think this is the only thing you should need to change
colour_labels = {"Black": "black", "White": "green","Red": "red", "na": 'blue', 'Mud': 'brown','green_control':'blue','Rust':'magenta','Rust_Canopy':'lawngreen','AribicavarGeisha':'cyan'}

#this is just to add legend as plotting all points as one scatter to be more efficent. Not that plotting loads of points is effcient...
handles = [(Line2D([], [], marker='.', markersize=10, color=colour_labels[i], linestyle='None')) for i in df.Species.unique()]

def list_maker(df):
    '''
    Returns lists of wavelengths, reflectance value, and species colours
    '''
    value_list = []
    columns_list = []
    colour_list = []
    for i in range(0, df.shape[0]):
        value_list += (df.iloc[i, :].values.tolist()[1:])
        colour_list += ([colour_labels[df.iloc[i, :].values.tolist()[0]]]
                        * (df.shape[1]-1))
        columns_list += (list(map(int, df.columns.tolist()[1:])))   #if your reflectance wavelengths are not interger then change the int to float
    return columns_list, value_list, colour_list


c_cols, c_vals, c_colours = list_maker(df)

plt.figure(figsize=(12,3),dpi=300) # can change the scale with figsize=(x-axis,y-axis)
plt.scatter(c_cols, c_vals, s=1, fc=c_colours,edgecolors="none")

plt.ylabel('Reflectance Values')
plt.xlabel('Wavelengths (nm)')
plt.legend(handles=handles, labels = list(df.Species.unique()+' Mangrove'),bbox_to_anchor=(0.5, -0.32),loc='center', ncol=3, fontsize=10) #If its not mangroves can change or get rid of that +' Mangrove'
plt.show()
