import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best_outliers_removed.csv")
fig = plt.figure(figsize=(12, 3))
ax1 = fig.add_axes([0.5, 0.1, 0.5, 0.9],yticks=[])
ax2 = fig.add_axes([0, 0.1, 0.5, 0.9],yticks=[])

first = rp_df[['Species','803']]
second = rp_df[['Species','1997']]

first_points = [first.query('Species == "Black"')['803'].tolist(), first.query('Species == "Red"')['803'].tolist(), first.query('Species == "White"')['803'].tolist()]
second_points = [second.query('Species == "Black"')['1997'].tolist(), second.query('Species == "Red"')['1997'].tolist(), second.query('Species == "White"')['1997'].tolist()]
colours = ['black','red','green']
labels = ['Black Mangrove','Red Mangrove','White Mangrove']
ax2.hist(first_points,bins=50,stacked=True,color=colours,label=labels)
ax1.hist(second_points,bins=50,stacked=True,color=colours)
ax1.text(0.98,0.96, '1997nm', size=20, transform=ax1.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax2.text(0.18, 0.96, '803nm', size=20, transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.02),
    fancybox=True, shadow=True, ncol=3)
fig.text(-0.012,0.5,'Frequency',ha='center', va='center',rotation=90,size=15)
fig.text(0.15,-0.02,'Reflactance Values',ha='center', va='top',size=15)
fig.text(0.5,1.01,'Histrogram of reflectance values at 803nm and 1997nm',ha='center', va='bottom',size=20)
plt.show()
