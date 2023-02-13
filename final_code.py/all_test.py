import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold


all_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData.csv")
rwb_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")
rp_df= pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best.csv")
labels = {"Black": "black", "White": "green",
          "Red": "red", "na": 'blue', 'Mud': 'brown'}
all_df = all_df.drop(["SpectraID", 'WhiteReference', 'ContactProbe', 'FibreOptic', 'SPAD_1', 'SPAD_2', 'SPAD_3',
                     'SPAD_Ave', 'Location', 'Lat', 'Long', 'StandAge', 'StandHealth', 'SurfaceDescription'], axis=1, inplace=False)
species_dictionary = {"Black": 0, "Red": 1, "White": 2}
model_dict = {'lightgbm':lgb.LGBMClassifier,'random forest':RandomForestClassifier,'xgboost':XGBClassifier,'gradient boosted':GradientBoostingClassifier}


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

def fisher_test(df):
    '''
    Returns Fisher score values in a list and wavelengths in a list
    '''
    x = df.drop(["Species"],axis=1,inplace=False)
    importance = fisher_score.fisher_score(x.to_numpy(),df.Species.to_numpy())
    cols = list(map(int,x.columns.tolist()))
    normed_importance = [x/max(importance) for x in importance]
    return cols,normed_importance

def information_gain_test(df):
    '''
    Returns Information Gain values in a list and wavelengths in a list
    '''
    x = df.drop(["Species"],axis=1,inplace=False)
    importance = mutual_info_classif(x,df.Species)
    cols = list(map(int,x.columns.tolist()))
    normed_importance = [x/max(importance) for x in importance]
    return cols,normed_importance

def variance_transformation(df):
    x = df.drop(["Species"],axis=1,inplace=False)
    selector = VarianceThreshold(0.5)
    return selector.fit_transform(x)

print(variance_transformation(rp_df).head())

def ANOVA():
    '''
    Should work this out at somepoint
    '''


def classifier_test(df,model_name):
    '''
    Returns model prediction score
    '''
    x = df.drop(["Species"],axis=1,inplace=False)
    if model_name == 'xgboost' and df.Species.iat[0] == str:
        y=label_to_float(df.Species)
    else:
        y=df.Species
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)
    model = model_dict[model_name](n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    if model_name =='xgboost':
        return metrics.r2_score(y_test,y_pred)
    else:
        return metrics.accuracy_score(y_test, y_pred)

def classifier_importance(df,model_name):
    '''
    Returns model feature importance list and wavelengths list
    '''
    x = df.drop(["Species"],axis=1,inplace=False)
    if model_name == 'xgboost' and df.Species.iat[0] == str:
        y=label_to_float(df.Species)
    else:
        y=df.Species
    model = model_dict[model_name](n_estimators=100,random_state=42)
    model.fit(x,y)
    importance = model.feature_importances_
    cols = list(map(int,x.columns.tolist()))
    return cols,importance

#print(classifier_test(rp_df,'xgboost'))
#print(classifier_importance(rp_df,'xgboost'))

def dendrogram_plot(df):
    '''
    plots dendrogram 
    '''
    X = df.drop(["Species"],axis=1,inplace=False)
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.xticks([])
    plt.show()

# x,y = fisher_test(clean_df)
# plt.scatter(x,y)
# x1,y1 = fisher_test(rwb_df)
# plt.scatter(x1,y1)
# x2,y2 = fisher_test(all_df)
# plt.scatter(x2,y2)
# x3,y3 = information_gain_test(clean_df)
# plt.scatter(x3,y3)
# x4,y4 = random_forest_fs(clean_df)
# plt.scatter(x4,y4)
# x5,y5 = boosted_decision_tree_fs(rp_df)
# plt.scatter(x5,y5)
# x6,y6 = xgboost_fs(rp_df)
# plt.scatter(x6,y6)
# x7,y7 = lightgbm_fs(rp_df)
# plt.scatter(x7,y7)
# plt.show()


print("Finished")
