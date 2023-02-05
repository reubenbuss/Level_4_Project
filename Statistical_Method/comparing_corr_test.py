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
    f = plt.figure(dpi=240)
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
    plt.plot(list(map(int,ser.index[1:])),ser.values[1:],label = "Correlation with Species")
    plt.title("Correlaton between Species and Wavelengths 350-1000nm")
    plt.xticks(range(350,1001,50), rotation=45)
    a = [754, 680, 396, 512]
    for i in a:
        plt.vlines(x=i,ymin=0,ymax=0.3,colors="green",label=f'{i}nm')
    plt.fill_betweenx(x1=350,x2=508,y=[0,0.3],alpha=0.2,fc = plt.cm.Pastel2(0),lw=0)
    plt.fill_betweenx(x1=512,x2=706,y=[0,0.3],alpha=0.1,fc = plt.cm.Pastel2(1),lw=0)
    plt.fill_betweenx(x1=706,x2=732,y=[0,0.3],alpha=0.3,fc = plt.cm.Pastel2(1),lw=0)
    plt.fill_betweenx(x1=741,x2=1000,y=[0,0.3],alpha=0.2,fc = plt.cm.Pastel2(2),lw=0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5)
    plt.show()

def correlated_features(all_features_df, target_feature, labels_df, THRESHOLD):
    '''
    returns a list of all the features correlated above the THRESHOLD with the target feature
    '''
    cor_matrix = all_features_df.corr().abs()
    correlated = (
        cor_matrix.index[cor_matrix[target_feature] > THRESHOLD]).tolist()
    highly_correlated_with_label = pd.concat(
        [label_to_float(labels_df), all_features_df[correlated]], axis=1)
    label_corr_matrix = (
        highly_correlated_with_label.astype(float)).corr().abs()
    corr_to_labels = label_corr_matrix[labels_df.name].copy()
    corr_to_labels.sort_values(ascending=False, inplace=True)
    float_to_label(labels_df)
    return list(corr_to_labels.index)[1:]

def corr_groups():
    ''''
    Identify the groups of highest correlation
    '''
    list_of_features = list(map(str,range(350,1000)))
    divisions = []
    while len(list_of_features)>0:
        group = correlated_features(x_df,list_of_features[0],y_ser,0.9)
        group.sort()
        group_2 = correlated_features(x_df,str(group[len(group)//2]),y_ser,0.9)
        group_2.sort()
        divisions.append([group_2[0],group_2[-1]])
        list_of_features = [x for x in list_of_features if x not in group_2]
    print(divisions)

#corr_groups()
corr_with_label("kendall")
