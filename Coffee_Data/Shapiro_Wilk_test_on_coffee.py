import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
import numpy as np

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data.csv")
green_control = df[df.Species == "green_control"]
Rust = df[df.Species == "Rust"]
Rust_Canopy = df[df.Species == "Rust_Canopy"]
AribicavarGeisha = df[df.Species == "AribicavarGeisha"]

fig = plt.figure(figsize = (8,2),dpi=2400)

for i in range(1,green_control.shape[1]):
    g = list(green_control.iloc[:,i])

    result_g = stats.shapiro(g)[1]
    if result_g > 0.05:
        plt.scatter(df.columns[i],1.06,s=0.01,c = "green",marker = "s")
    else:
        plt.scatter(df.columns[i],0.06,s=0.01,c = "green",marker = "s")

    
for i in range(1,Rust.shape[1]):
    r = list(Rust.iloc[:,i])

    result_r = stats.shapiro(r)[1]
    if result_r > 0.05:
        plt.scatter(df.columns[i],1.02,s=0.01,c = "brown",marker = "s")
    else:
        plt.scatter(df.columns[i],0.02,s=0.01,c = "brown",marker = "s")

for i in range(1,Rust_Canopy.shape[1]):
    rc = list(Rust_Canopy.iloc[:,i])

    result_rc = stats.shapiro(rc)[1]
    if result_rc > 0.05:
        plt.scatter(df.columns[i],0.98,s=0.01,c = "red",marker = "s")
    else:
        plt.scatter(df.columns[i],-0.02,s=0.01,c = "red",marker = "s")


for i in range(1,AribicavarGeisha.shape[1]):
    a = list(AribicavarGeisha.iloc[:,i])

    result_a = stats.shapiro(a)[1]
    if result_a > 0.05:
        plt.scatter(df.columns[i],0.94,s=0.01,c = "blue",marker = "s")
    else:
        plt.scatter(df.columns[i],-0.06,s=0.01,c = "blue",marker = "s")

plt.plot([3,4],[3,4],color = "green",label = "green_control")
plt.plot([3,4],[3,4],color = "brown",label = "Rust")
plt.plot([3,4],[3,4],color = "red",label = "Rust_Canopy")
plt.plot([3,4],[3,4],color = "blue",label = "AribicavarGeisha")
plt.xlabel("Wavelengths (nm)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=False, ncol=4)
plt.title("Shapiro–Wilk Test")
plt.ylim(-0.25,1.25)
plt.yticks([0,1],["Not Normally Distributed", "Normally Distributed"])
plt.xticks(df.columns[1::200],labels=range(180,1021,84))
fig.savefig('Shapiro–Wilk on Coffee Data.png', dpi=fig.dpi)
plt.show()
