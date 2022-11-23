
import pandas as pd
import numpy as np
import tabulate as tb
from matplotlib import pyplot as plt

Black_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black.csv"
White_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\White.csv"
Red_file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Red.csv"

Black_df = pd.read_csv(Black_file)
White_df = pd.read_csv(White_file)
Red_df = pd.read_csv(Red_file)

Black_std = list(Black_df.std())
Black_std = Black_std[6:]
Black_m = list(Black_df.mean())
Black_m = Black_m[6:]
Black_m = np.array(Black_m)
Black_std = np.array(Black_std)

White_std = list(White_df.std())
White_std = White_std[6:]
White_m = list(White_df.mean())
White_m = White_m[6:]
White_m = np.array(White_m)
White_std = np.array(White_std)

Red_std = list(Red_df.std())
Red_std = Red_std[2:]
Red_m = list(Red_df.mean())
Red_m = Red_m[2:]
Red_m = np.array(Red_m)
Red_std = np.array(Red_std)

B_W_m = np.absolute(Black_m - White_m)
B_W_std = np.absolute(Black_std + White_std)
B_R_m = np.absolute(Black_m - Red_m)
B_R_std = np.absolute(Black_std + Red_std)
W_R_m = np.absolute(White_m - Red_m)
W_R_std = np.absolute(White_std + Red_std)

B_W_2 = B_W_m - 0.5*B_W_std
B_R_2 = B_R_m - 0.5*B_R_std
W_R_2 = W_R_m - 0.5*W_R_std
B_W = B_W_m - B_W_std
B_R = B_R_m - B_R_std
W_R = W_R_m - W_R_std
B_W_1 = B_W_m - 0.1*B_W_std
B_R_1 = B_R_m - 0.1*B_R_std
W_R_1 = W_R_m - 0.1*W_R_std

n = 50 #number of wavelengths displayed 

B_W_2_sorted = np.argsort(B_W_2)[::-1][:n]
B_R_2_sorted = np.argsort(B_W_2)[::-1][:n]
W_R_2_sorted = np.argsort(W_R_2)[::-1][:n]

B_W_sorted = np.argsort(B_W)[::-1][:n]
B_R_sorted = np.argsort(B_W)[::-1][:n]
W_R_sorted = np.argsort(W_R)[::-1][:n]

B_W_1_sorted = np.argsort(B_W_1)[::-1][:n]
B_R_1_sorted = np.argsort(B_W_1)[::-1][:n]
W_R_1_sorted = np.argsort(W_R_1)[::-1][:n]

colours = ["blue","red","green","skyblue","lightcoral","palegreen","blueviolet","firebrick","darkseagreen"]
fig = plt.figure(dpi=2400)
plt.hist([B_W_sorted+350,B_R_sorted+350,W_R_sorted+350,B_W_1_sorted+350,B_R_1_sorted+350,W_R_1_sorted+350,B_W_2_sorted+350,B_R_2_sorted+350,W_R_2_sorted+350],color = colours,bins = 50,label=["B_W","B_R","W_R","B_W_1","B_R_1","W_R_1","B_W_2","B_R_2","W_R_1"],stacked = True)
plt.legend(loc="upper center",ncol=3,bbox_to_anchor=(0.5, -0.05))
plt.show()
print("finished")


#data = {"B_W":B_W_sorted+350,"B_R":B_R_sorted+350,"W_R":W_R_sorted+350,
#    "B_W_0.5":B_W_2_sorted+350,"B_R_0.5":B_R_2_sorted+350,"W_R_0.5":W_R_2_sorted+350,
#    "B_W_0.1":B_W_1_sorted+350,"B_R_0.1":B_R_1_sorted,"W_R_0.1":W_R_1_sorted}
#df = pd.DataFrame(data)
#print(df)
#print(df.to_latex())

#table = zip(B_W_sorted+350,B_R_sorted+350,W_R_sorted+350)
#headers = ["B_W","B_R","W_R"]
#tables = tb.tabulate(table,headers=headers, tablefmt="plain")

#open("Greatest Difference table.txt","w",encoding="utf-8").write(tables)






