import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision.csv")
rp_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_and_Coffee_data.csv")
fig = plt.figure(figsize=(12, 3),dpi=240)

rp_df = rp_df.query('Species != "Rust" & Species != "Rust_Canopy"')
#print(rp_df.Species.unique())
#print(list(rp_df.columns))
ax1 = fig.add_axes([0.5, 0.1, 0.5, 0.9],yticks=[])
ax2 = fig.add_axes([0, 0.1, 0.5, 0.9],yticks=[])
ax3 = fig.add_axes([1,0.1,0.5,0.9],yticks=[])

print(list(rp_df.columns))
a='479'
b='602'
c='680'

first = rp_df[['Species',a]].query(f'@rp_df["{a}"]<0.125')
second = rp_df[['Species',b]].query(f'@rp_df["{b}"]<0.2')
third = rp_df[['Species',c]].query(f'@rp_df["{c}"]<0.1')


first_points = [first.query('Species == "Black"')[a].tolist(), first.query('Species == "Red"')[a].tolist(), first.query('Species == "White"')[a].tolist(),first.query('Species == "green_control"')[a].tolist(), first.query('Species == "Rust"')[a].tolist(), first.query('Species == "Rust_Canopy"')[a].tolist(),first.query('Species == "AribicavarGeisha"')[a].tolist()]
second_points = [second.query('Species == "Black"')[b].tolist(), second.query('Species == "Red"')[b].tolist(), second.query('Species == "White"')[b].tolist(),second.query('Species == "green_control"')[b].tolist(), second.query('Species == "Rust"')[b].tolist(), second.query('Species == "Rust_Canopy"')[b].tolist(),second.query('Species == "AribicavarGeisha"')[b].tolist()]
third_points = [third.query('Species == "Black"')[c].tolist(), third.query('Species == "Red"')[c].tolist(), third.query('Species == "White"')[c].tolist(),third.query('Species == "green_control"')[c].tolist(), third.query('Species == "Rust"')[c].tolist(), third.query('Species == "Rust_Canopy"')[c].tolist(),third.query('Species == "AribicavarGeisha"')[c].tolist()]
colours = ['black','red','green','blue','magenta','lawngreen','cyan']
labels = ['Black Mangrove','Red Mangrove','White Mangrove','Green Coffee','Rust','Rust Canopy','Geisha']
ax2.hist(first_points,bins=50,stacked=True,color=colours,label=labels)
ax1.hist(second_points,bins=50,stacked=True,color=colours)
ax3.hist(third_points,bins=50,stacked=True,color=colours)
ax1.text(0.98,0.96, f'{b}nm', size=20, transform=ax1.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax2.text(0.98, 0.96, f'{a}nm', size=20, transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax3.text(2.98, 0.96, f'{c}nm', size=20, transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))

fig.legend(loc='upper center', bbox_to_anchor=(0.75, 0.02),
    fancybox=True, shadow=True, ncol=7)
fig.text(-0.012,0.5,'Frequency',ha='center', va='center',rotation=90,size=15)
fig.text(0.15,-0.02,'Reflectance Values',ha='center', va='top',size=15)
ax1.set_xlim(0,0.2)
ax2.set_xlim(0,0.125)
ax3.set_xlim(0,0.1)

plt.show()
