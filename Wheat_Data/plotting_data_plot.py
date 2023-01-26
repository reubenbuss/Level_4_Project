import pandas as pd
from matplotlib import pyplot as plt
#all data
#df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_data_all_headers.csv")
#reduced precision data
df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat\Wheat_data_reduced_precision.csv")
x = list(df.columns)
x = x[2:]

fig, ax = plt.subplots(4, 6,figsize = (24,12),sharey=True,dpi=600)
plt.setp(ax, ylim=(0,1),xticks=[])
plt.suptitle("Infected vs Healthy Wheat Reflectance Data From 2 To 13 Days Post Infection",fontsize=24)
for i in range(0,12):
    if i+2 != 10:
        df1 = df[df["Day"] == f"{i+2} dpi"]
        df1_i = df1.query('Condition == "infected" | Condition == "inf"')
        df1_h = df1.query('Condition == "healthy"')
        data = df1_h.drop(["Day","Condition"],axis = 1,inplace = False)
        data = df1_i.drop(["Day","Condition"],axis=1,inplace=False)
        if i<=5:
            for j in range(0,2):
                if j == 0:
                    for k in range(0,data.shape[0]):
                        ax[j,i].set_title(f"Day {i+2}",fontsize=16)
                        ax[j, i].scatter(x,data.iloc[k,:],s=0.01,c="green")
                else:
                    for k in range(0,data.shape[0]):
                        ax[j,i].scatter(x,data.iloc[k,:],s=0.01,c="red")
        if i>5:
            for j in range(0,2):
                if j == 0:
                    for k in range(0,data.shape[0]):
                        ax[j+2,i-6].set_title(f"Day {i+2}",fontsize=16)
                        ax[j+2, i-6].scatter(x,data.iloc[k,:],s=0.01,c="green")
                else:
                    for k in range(0,data.shape[0]):
                        ax[j+2,i-6].scatter(x,data.iloc[k,:],s=0.01,c="red")
                        ax[j+2,i-6].set_xticks([0,104,204,304,404,484],[396,500,600,700,800,880])
ax[0,0].set_ylabel("Healthy",fontsize=16)
ax[1,0].set_ylabel("Infected",fontsize=16)
ax[2,0].set_ylabel("Healthy",fontsize=16)
ax[3,0].set_ylabel("Infected",fontsize=16)
plt.show()
print("Finished")