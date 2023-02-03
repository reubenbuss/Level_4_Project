import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv") 
x_df = df.iloc[:,1:652]
y_ser = df.Species
species_dictionary = {"Black": 0.2, "Red": 0.4, "White": 0.6}

def label_to_float(labels_df):
    '''
    Tranforms the str labels to flaot labels
    '''
    for i in range(0, len(labels_df)):
        labels_df.iat[i] = species_dictionary[labels_df.iat[i]]
    return labels_df


def float_to_label(labels_df):
    '''
    Transforms the float labels to str labels
    '''
    dictionary_unpacked = dict(
        zip(species_dictionary.values(), species_dictionary.keys()))
    for i in range(0, len(labels_df)):
        labels_df.iat[i] = dictionary_unpacked[labels_df.iat[i]]
    return labels_df


def corr_matrix(method="pearson"):
    '''
    Produce heatmap of correlation between each feature
    '''
    corr_df = x_df.corr(method)
    f = plt.figure(dpi=2400)
    plt.matshow(corr_df, fignum=f.number)
    plt.xticks(range(0,len(x_df.columns),50), range(350,len(x_df.columns)+350,50), fontsize=14, rotation=45)
    plt.yticks(range(0,len(x_df.columns),50), range(350,len(x_df.columns)+350,50), fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.suptitle('Correlation Matrix 350nm to 1000nm', y=1.05, size=16)
    plt.title(f'{method} Method', y=1.12, fontsize=10)
    plt.show()


def corr_with_label(method='pearson'):
    '''
    Produce a heatmap of the correlation between the labels and the featues
    '''
    labels = label_to_float(y_ser).astype("float")
    new_df = pd.concat([labels,x_df],axis=1)
    corr_ser = new_df.corr(method=method,numeric_only=True).abs()
    ser = corr_ser["Species"]
    fig = plt.figure(dpi=2400)
    plt.scatter(list(map(int,ser.index[1:])),ser.values[1:])
    plt.title("Correlaton between Species and Wavelengths 350-1000nm")
    plt.xticks(range(350,1001,50), rotation=45)
    plt.fill_betweenx(x1=350,x2=500,y=[0,0.3],alpha=0.2)
    plt.show()


def corr_groups(method="pearson"):
    ''''
    Identify the groups of highest correlation
    '''
    corr_matrix = x_df.corr(method=method,numeric_only=True)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    all_features = range(350,1001)
    groups = []
    while len(all_features)>0:
        to_drop = []
        for j in range(all_features[0]-350,651):
            if corr_matrix.iloc[all_features[0]-350,j] > 0.9:
                to_drop.append(j+350)
        groups.append(to_drop)
        all_features = [x for x in all_features if x not in to_drop]
    final_groups = []
    for i in range(0,len(groups)):
        if corr_matrix.iloc[groups[i][-1]-350,groups[i+1][0]-350] > 0.9:
            final_groups += groups[i] + groups[i+1]
    print(final_groups)

corr_groups()


