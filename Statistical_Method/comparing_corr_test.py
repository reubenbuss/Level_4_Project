import pandas as pd
import matplotlib.pyplot as plt

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
    Return 
    '''
    corr_df = x_df.corr(method)
    f = plt.figure(dpi=2400)
    plt.matshow(corr_df, fignum=f.number)
    plt.xticks(range(0,len(x_df.columns),50), range(350,len(x_df.columns)+350,50), fontsize=14, rotation=45)
    plt.yticks(range(0,len(x_df.columns),50), range(350,len(x_df.columns)+350,50), fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.suptitle('Correlation Matrix 350nm to 1000nm', y=1.05, size=16)
    plt.title('Pearson Method', y=1.03, fontsize=10)
    plt.setp(fig.xticks.)
    plt.show()

corr_matrix("pearson")
