import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData.csv")
rwb_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")
clean_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best.csv")
labels = {"Black":"black","White":"green","Red":"red","na":'blue','Mud':'brown'}

def list_maker(df,species,colour):
    selection = df.query("Species == "f"'{species}'")
    value_list = []
    columns_list = []
    for i in range(0,selection.shape[0]):
        value_list += selection.iloc[i,:].values.tolist()[1:]
        columns_list += clean_df.columns.tolist()[1:]
    columns_list = list(map(int,columns_list))
    ax1.scatter(columns_list,value_list,s=0.1,c=f'{colour}')

#print(labels[clean_df.iloc[0,:].values.tolist()[0]])

def plotter(df):
    value_list = []
    columns_list = []
    colour_list = []
    for i in range(0,df.shape[0]):
        value_list += (df.iloc[i,:].values.tolist()[1:])
        colour_list += ([labels[df.iloc[i,:].values.tolist()[0]]]*df.shape[1])
        columns_list += (list(map(int,clean_df.columns.tolist()[1:])))
    print(len(columns_list),len(value_list),len(colour_list))
    print(colour_list[0])
    #ax1.scatter(columns_list,value_list,s=0.1,c=colour_list)


fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.6, 0.8, 0.3],
                   xticklabels=[], ylim=(0,1))
ax2 = fig.add_axes([0.1, 0.3, 0.8, 0.3],
                   ylim=(0,1))
ax3 = fig.add_axes([0.1, 0, 0.8, 0.3],ylim=(0,1))

x = np.linspace(0, 10)
#list_maker(clean_df,"Black","black")
#list_maker(clean_df,"Red","red")
#list_maker(clean_df,"White","green")
plotter(clean_df)
ax2.plot(np.cos(x))
ax3.plot(np.sin(x-(np.pi)/2))

