import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision.csv")
df = df.sample(frac=1)
fig = plt.figure(dpi=300,figsize=(9,3))
for i in range(0,df.shape[0]):
    if df.iloc[i,0] == "green_control":
        COLOUR = "Green"
    if df.iloc[i,0] == "Rust":
        COLOUR = "brown"
    if df.iloc[i,0] == "Rust_Canopy":
        COLOUR = "red"
    if df.iloc[i,0] == "AribicavarGeisha":
        COLOUR = "blue"
    y=df.values[i,1:]
    plt.scatter(list(map(int,df.columns[1:])),y,s=0.01,c=COLOUR,marker="o")
plt.scatter(0,-50,color = "Green",label = "Green Control")
plt.scatter(1655,-50,color = "brown",label = "Rust")
plt.scatter(0,-20,color = "red",label = "Rust Canopy")
plt.scatter(0,-20,color = "blue",label = "Geisha")
#plt.vlines(x=[280,1655], ymin=0, ymax=95, colors='black', ls=':', lw=2, label='Acceptable Range')
#plt.xticks(list(range(0,2001,400)),[187,370,544,710,865,1010])
plt.ylim(0,1)
plt.xlim(340,900)
plt.ylabel('Reflectance')
plt.xlabel('Wavelength (nm)')
#plt.title("Coffee Data")
plt.legend(bbox_to_anchor=(0.5, -0.25),loc='center', ncol=4, fontsize=8)
plt.show()

print("Finished")
