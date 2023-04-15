import pandas as pd
import matplotlib.pyplot as plt

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best_outliers_removed.csv")
species_dictionary = {"Black": 0, "Red": 1, "White": 2}

fig = plt.figure(figsize=(9, 3),dpi=300)
ax1 = fig.add_axes([0, 0.1, 0.5, 0.9])
ax2 = fig.add_axes([0.6, 0.1, 0.5, 0.9])
ax3 = fig.add_axes([1.2, 0.1, 0.5, 0.9])
#ax1.yaxis.tick_right()

#print(rp_df.columns.tolist())

#first plot
a='476'
b='677'
#second plot
c='602'
d='1867'

#third
e='1647'
f='2442'

labels = {"Black": "black", "White": "green",
          "Red": "red", "na": 'blue', 'Mud': 'brown'}

def label_to_float(labels_df):
    '''
    Tranforms the str labels to flaot labels
    '''
    new_labels=[]
    for i in range(0, len(labels_df)):
        new_labels.append(species_dictionary[labels_df.iat[i]])
    return new_labels

new_points = rp_df[['Species',a,b,c,d,e,f]]
new_points = new_points.sample(frac=1)
colour_labels = new_points['Species'].map(labels)

corr_matrix = new_points.corr('pearson',numeric_only=True)
print(corr_matrix)
ab = round(corr_matrix.iloc[0,1],2)
cd = round(corr_matrix.iloc[2,3],2)
ef = round(corr_matrix.iloc[4,5],2)

ax1.scatter(new_points[a].tolist(),new_points[b].tolist(),fc=colour_labels,edgecolors='black',linewidths=0.5,s=20)
ax2.scatter(new_points[c].tolist(),new_points[d].tolist(),fc=colour_labels,edgecolors='black',linewidths=0.5,s=20)
ax3.scatter(new_points[e].tolist(),new_points[f].tolist(),fc=colour_labels,edgecolors='black',linewidths=0.5,s=20)

# ax2.scatter(new_points[a].tolist(),label_to_float(new_points.Species),fc=colour_labels,edgecolors=None)
# ax1.scatter(new_points[c].tolist(),label_to_float(new_points.Species),fc=colour_labels,edgecolors=None)

ax1.scatter(0, -2, fc="red",edgecolors='black',linewidths=0.5, label="Red Mangroves")
ax1.scatter(0, -2, fc="green",edgecolors='black',linewidths=0.5, label="White Mangroves")
ax1.scatter(0, -2, fc="black",edgecolors='black',linewidths=0.5, label="Black Mangroves")

small = min(new_points[a].tolist())
big = max(new_points[a].tolist())

# small = -0.1
# big = 2.1

ax1.set_ylim(small-0.01,big+0.01)

ax1.text(0.02,0.97, f'{a}nm vs {b}nm', size=15, transform=ax1.transAxes, ha="left", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax2.text(0.02, 0.97, f'{c}nm vs {d}nm', size=15, transform=ax2.transAxes, ha="left", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax3.text(0.02, 0.97, f'{e}nm vs {f}nm', size=15, transform=ax3.transAxes, ha="left", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))

ax1.text(0.98,0.97, f'r={ab}', size=15, transform=ax1.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax2.text(0.98, 0.97, f'r={cd}', size=15, transform=ax2.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))
ax3.text(0.98, 0.97, f'r={ef}', size=15, transform=ax3.transAxes, ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec='black',
                   fc='white',
                   ))

fig.legend(loc='upper center', bbox_to_anchor=(0.85, 0.02),fancybox=True, shadow=True, ncol=3)
fig.text(-0.06,0.5,'Reflectance Values',ha='center', va='center',rotation=90,size=15)
fig.text(0.2,-0.02,'Reflectance Values',ha='center', va='top',size=15)
#fig.text(0.5,1.01,'Plot of highly correlated vs  uncorrelated wavelengths',ha='center', va='bottom',size=20)
plt.show()


