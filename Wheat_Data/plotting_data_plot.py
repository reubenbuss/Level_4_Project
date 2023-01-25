import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_data_all_headers.csv")
df1 = df[df["Day"]=="2 dpi"]
df1_i = df1[df1["Condition"] == "infected"]
if df1_i.empty:
    df1_i = df1[df1["Condition"] == "inf"]
y = list(df.columns)
y = y[2:]

fig, ax = plt.subplots(2, 12,figsize = (6,24))
for i in range(0,12):
    df1 = df[df["Day"] == f"{i} dpi"]
    for j in range(0,2):
        df1_i = df1[df1["Condition"] == "infected"]
        if df1_i.empty:
            df1_i = df1[df1["Condition"] == "inf"]
        df1_h = df1[df1["Condition"] == "healthy"]
        if j == 0:
            data = df1_h.drop(["Day","Condition"],axis = 1,inplace = False)
            for k in range(0,data.shape[0]):
                ax[j, i//2].scatter(data.iloc[k,:],y,s=0.01,c="green")
        else:
            data = df1_i.drop(["Day","Condition"],axis=1,inplace=False)
            for k in range(0,data.shape[0]):
                ax[j,i//2].scatter(data.iloc[k,:],y,s=0.01,c="red")
plt.show()
