import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision.csv")
rp_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_and_Coffee_data.csv")

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv")


fig = plt.figure(figsize=(12, 3),dpi=240)

#rp_df = rp_df.query('Species != "Rust" & Species != "Rust_Canopy"')
#print(rp_df.Species.unique())
#print(list(rp_df.columns))
ax1 = fig.add_axes([0, 0.1, 0.5, 0.9],yticks=[])
ax2 = fig.add_axes([0.5, 0.1, 0.5, 0.9],yticks=[])
ax3 = fig.add_axes([1,0.1,0.5,0.9],yticks=[])
ax4 = fig.add_axes([1.5,0.1,0.5,0.9],yticks=[])

#print(list(rp_df.columns))
#[485,1667,2067,2442]
#1647,2142,2232,2007
a='1647'
b='2007'
c='2142'
d='2232'

# first = rp_df[['Species',a]].query(f'@rp_df["{a}"]<0.125')
# second = rp_df[['Species',b]].query(f'@rp_df["{b}"]<0.2')
# third = rp_df[['Species',c]].query(f'@rp_df["{c}"]<0.1')
# forth = rp_df[['Species',d]].query(f'@rp_df["{d}"]<0.1')

first = rp_df[['Species',a]]
second = rp_df[['Species',b]]
third = rp_df[['Species',c]]
forth = rp_df[['Species',d]]


first_points = [first.query('Species == "Black"')[a].tolist(), first.query('Species == "Red"')[a].tolist(), first.query('Species == "White"')[a].tolist(),first.query('Species == "green_control"')[a].tolist(), first.query('Species == "Rust"')[a].tolist(), first.query('Species == "Rust_Canopy"')[a].tolist(),first.query('Species == "AribicavarGeisha"')[a].tolist()]
second_points = [second.query('Species == "Black"')[b].tolist(), second.query('Species == "Red"')[b].tolist(), second.query('Species == "White"')[b].tolist(),second.query('Species == "green_control"')[b].tolist(), second.query('Species == "Rust"')[b].tolist(), second.query('Species == "Rust_Canopy"')[b].tolist(),second.query('Species == "AribicavarGeisha"')[b].tolist()]
third_points = [third.query('Species == "Black"')[c].tolist(), third.query('Species == "Red"')[c].tolist(), third.query('Species == "White"')[c].tolist(),third.query('Species == "green_control"')[c].tolist(), third.query('Species == "Rust"')[c].tolist(), third.query('Species == "Rust_Canopy"')[c].tolist(),third.query('Species == "AribicavarGeisha"')[c].tolist()]
forth_points = [forth.query('Species == "Black"')[d].tolist(), forth.query('Species == "Red"')[d].tolist(), forth.query('Species == "White"')[d].tolist(),forth.query('Species == "green_control"')[d].tolist(), forth.query('Species == "Rust"')[d].tolist(), forth.query('Species == "Rust_Canopy"')[d].tolist(),forth.query('Species == "AribicavarGeisha"')[d].tolist()]

colours = ['black','red','green','blue','magenta','lawngreen','cyan']
labels = ['Black Mangrove','Red Mangrove','White Mangrove','Green Coffee','Rust','Rust Canopy','Geisha']
labels = ['Black Mangrove','Red Mangrove','White Mangrove']

ax1.hist(first_points,bins=50,stacked=True,color=colours,label=labels)
ax2.hist(second_points,bins=50,stacked=True,color=colours)
ax3.hist(third_points,bins=50,stacked=True,color=colours)
ax4.hist(forth_points,bins=50,stacked=True,color=colours)
ax1.text(0.98,0.96, f'{a}nm', size=20, transform=ax1.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax2.text(0.98, 0.96, f'{b}nm', size=20, transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax3.text(0.98, 0.96, f'{c}nm', size=20, transform=ax3.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax4.text(0.98, 0.96, f'{d}nm', size=20, transform=ax4.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))


fig.legend(loc='upper center', bbox_to_anchor=(1, 0.02),
    fancybox=True, shadow=True, ncol=3)
fig.text(-0.012,0.5,'Frequency',ha='center', va='center',rotation=90,size=15)
fig.text(0.15,-0.02,'Reflectance Values',ha='center', va='top',size=15)


plt.show()
