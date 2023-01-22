import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data.csv")
df = df.sample(frac=1)
fig = plt.figure(dpi=2400)

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
    plt.scatter(list(df)[1:],y,s=0.01,c=COLOUR)
plt.scatter(0,-20,color = "Green",label = "Green_Control")
plt.scatter(0,-20,color = "brown",label = "Rust")
plt.scatter(0,-20,color = "red",label = "Rust_Canopy")
plt.scatter(0,-20,color = "blue",label = "AribicavarGeisha")
plt.xticks(df.columns[1::200],labels=range(180,1021,84))
plt.ylim(-10,100)
plt.legend(loc="upper left")
plt.title("Coffee Data")
plt.show()

print("Finished")
