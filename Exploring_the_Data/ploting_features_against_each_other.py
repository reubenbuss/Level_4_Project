import pandas as pd
import matplotlib.pyplot as plt

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best_outliers_removed.csv")
fig = plt.figure(figsize=(9, 3))
ax1 = fig.add_axes([0.5, 0.1, 0.5, 0.9])
ax2 = fig.add_axes([0, 0.1, 0.5, 0.9])
ax1.yaxis.tick_right()

#print(rp_df.columns.tolist())

#first plot
a='701'
b='539'
#second plot
c='452'
d='599'

labels = {"Black": "black", "White": "green",
          "Red": "red", "na": 'blue', 'Mud': 'brown'}

new_points = rp_df[['Species',a,b,c,d]]

new_points['Species'] = new_points['Species'].map(labels)

new_points = new_points.sample(frac=1)
ax2.scatter(new_points[a].tolist(),new_points[b].tolist(),fc=new_points['Species'],edgecolors=None)
ax1.scatter(new_points[c].tolist(),new_points[d].tolist(),fc=new_points['Species'],edgecolors=None)

ax2.scatter(0, 2, c="red", label="Red Mangroves")
ax2.scatter(0.2, 2, c="green", label="White Mangroves")
ax2.scatter(0, 2, c="black", label="Black Mangroves")

small = min(new_points[b].tolist())
big = max(new_points[b].tolist())

ax2.set_ylim(small-0.01,big+0.01)
ax1.text(0.98,0.97, f'{c}nm vs {d}nm', size=15, transform=ax1.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax2.text(0.02, 0.97, f'{a}nm vs {b}nm', size=15, transform=ax2.transAxes, ha="left", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))

fig.legend(loc='upper center', bbox_to_anchor=(0.65, 0.02),fancybox=True, shadow=True, ncol=3)
fig.text(-0.06,0.5,'Reflectance Values',ha='center', va='center',rotation=90,size=15)
fig.text(0.15,-0.02,'Reflactance Values',ha='center', va='top',size=15)
fig.text(0.5,1.01,'Plot of highly correlated vs  uncorrelated wavelengths',ha='center', va='bottom',size=20)
plt.show()
