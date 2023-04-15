import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_selection import mutual_info_classif
from skrebate import ReliefF

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv")
coffee_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision.csv")
mangrove_and_coffee_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_and_Coffee_data.csv")

colour_labels = {"Black": "black", "White": "green","Red": "red", "na": 'blue', 'Mud': 'brown','green_control':'blue','Rust':'magenta','Rust_Canopy':'lawngreen','AribicavarGeisha':'cyan'}
species_dictionary = {"Black": 0, "Red": 1, "White": 2,'green_control':3,'Rust':4,'Rust_Canopy':5,'AribicavarGeisha':6}


def make_groups(df,coef=0.9):
    '''
    makes highly correlated groups
    '''
    corr_matrix = df.corr(method = 'pearson',numeric_only=True)
    corr_matrix_columns = list(map(int,corr_matrix.columns))

    points = []
    heights = []
    for i in corr_matrix_columns:
        correlated = [x for x in corr_matrix_columns if corr_matrix.loc[f'{i}',f'{x}'] > coef]
        points.append(correlated)
        heights.append([i]*(len(correlated)))
    groups = []
    for i in points:
        if len(groups) == 0:
            groups.append(i)
        else:
            add = 0
            swap = 0
            for j in groups:
                if len(set(j).intersection(set(i)))/len(set(j).union(set(i))) < 0.1:
                    add += 1
                if set(j).intersection(set(i)) == set(j) and len(i) > len(j):
                    swap += 1
            if add == len(groups):
                groups.append(i)
            if swap == len(groups):
                groups.remove(j)
                groups.append(i)
    return groups

def label_to_float(labels_df):
    '''
    Tranforms the str labels to flaot labels
    '''
    new_labels=[]
    for i in range(0, len(labels_df)):
        new_labels.append(species_dictionary[labels_df.iat[i]])
    return np.array(new_labels)

def bounds_of_lists(groups):
    return [[x[0],x[-1]] for x in groups]

def relief_fs(df):
    '''
    Returns 
    '''
    labels_list=list(df.Species.unique())
    x=df.drop(['Species'],axis=1,inplace=False).to_numpy()
    handles = [(Line2D([], [], marker='.', markersize=10, color=colour_labels[i], linestyle='None')) for i in labels_list]
    y = label_to_float(df.Species)
    r = ReliefF(n_neighbors=10)
    r.fit(x,y)
    plt.figure(figsize=(9,3),dpi=300)
    plt.plot(list(map(int,list(df.columns)[1:])),r.feature_importances_,c='black')
    plt.ylabel('Feature Importance')
    plt.xlabel('Wavelengths (nm)')
    plt.legend(handles=handles, labels = labels_list,bbox_to_anchor=(0.5, -0.32),loc='center', ncol=3, fontsize=10)
    plt.show()
    #return r.feature_importances_

def relief_one_vs_rest(df):
    '''
    graph of one vs rest for releifF
    '''
    labels_list=list(df.Species.unique())
    x=df.drop(['Species'],axis=1,inplace=False).to_numpy()
    handles = [(Line2D([], [], marker='.', markersize=10, color=colour_labels[i], linestyle='None')) for i in labels_list]
    importances = []
    r = ReliefF(n_neighbors=10)
    plt.figure(figsize=(9,3),dpi=300)
    for i in labels_list:
        y = [0 if x!=i else 1 for x in list(df.Species)]
        r.fit(x,y)
        importances.append(r.feature_importances_)
        plt.plot(list(map(int,list(df.columns)[1:])),r.feature_importances_,c=colour_labels[i])
    plt.ylabel('Feature Importance')
    plt.xlabel('Wavelengths (nm)')
    plt.legend(handles=handles, labels = labels_list,bbox_to_anchor=(0.5, -0.32),loc='center', ncol=3, fontsize=10)
    plt.show()
    #return importances

def top_from_importances_in_each_group(df,groups, importances):
    importances.insert(0,list(df.columns)[1:])
    new_importances = np.reshape(importances,(len(importances),-1)).T
    for i in groups:
        for j in i:
            break

def mutual_info_one_vs_rest(df):
    '''
    graph of one vs rest for releifF
    '''
    labels_list=list(df.Species.unique())
    x=df.drop(['Species'],axis=1,inplace=False).to_numpy()
    handles = [(Line2D([], [], marker='.', markersize=10, color=colour_labels[i], linestyle='None')) for i in labels_list]
    importances = []
    plt.figure(figsize=(9,3),dpi=300)
    for i in labels_list:
        y = [0 if x!=i else 1 for x in list(df.Species)]
        importance =  mutual_info_classif(x,y)
        importances.append(importance)
        plt.plot(list(map(int,list(df.columns)[1:])),importance,c=colour_labels[i])
    plt.ylabel('Feature Importance')
    plt.xlabel('Wavelengths (nm)')
    plt.legend(handles=handles, labels = labels_list,bbox_to_anchor=(0.5, -0.32),loc='center', ncol=3, fontsize=10)
    plt.show()




# importances = relief_one_vs_rest(rp_df)
# groups = make_groups(rp_df)
# top_from_importances_in_each_group(rp_df,groups,importances)

mutual_info_one_vs_rest(mangrove_and_coffee_df)

relief_one_vs_rest(mangrove_and_coffee_df.query('Species != "Rust" & Species != "Rust_Canopy"'))
#relief_fs(mangrove_and_coffee_df.query('Species != "Rust"'))
print('Finished')
