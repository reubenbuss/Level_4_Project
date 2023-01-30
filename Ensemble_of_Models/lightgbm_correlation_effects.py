import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm 
import shap
import pandas as pd

import lightgbm as lgb

reu = 10

df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")
species_labels = df.Species
reflectance_data = df.drop(["Species"], axis=1)
species_dictionary = {"Black": 0.2, "Red": 0.4, "White": 0.6}


def classifier(features_df, labels_df, state=42):
    '''
    function that performs a lighgbm classification on selected dataset
    returning the accuracy on test data
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        features_df, labels_df, test_size=0.2, random_state=state)
    model = lgb.LGBMClassifier(
        learning_rate=0.4, max_depth=5, num_leaves=10, random_state=state)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
              eval_metric='logloss', feature_name=list(features_df.columns),
              callbacks=[lgb.log_evaluation(period=-1), lgb.early_stopping(stopping_rounds=10)])
    print(f'Training accuracy {model.score(x_train,y_train)}')
    print(f'Testing accuracy {model.score(x_test,y_test)}')
    return model.score(x_test, y_test)


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


def test_each_feature(all_features_df, selected_features, target_feature, labels_series, THRESHOLD):
    '''
    For a selected feature it finds all correlated features and then calls the lightgbm algorithm to get a score of it performance.
    Repeating for all correlated features produces a series with the scores and feature names as index
    '''
    corr_features = correlated_features(
        all_features_df, target_feature, labels_series, THRESHOLD)
    score = []
    test_selected_features = selected_features.copy()
    test_selected_features.remove(target_feature)
    for i in corr_features:
        test_selected_features.append(i)
        score.append(classifier(
            all_features_df[test_selected_features], labels_series))
        test_selected_features.remove(i)
    scores_ser = pd.Series(dict(zip(corr_features, score)))
    return scores_ser

def find_best(reflectance_data, selected_features, species_labels, THRESHOLD):
    features = []
    scores = []
    best_features = selected_features.copy()
    for i in range(0,len(selected_features)):
        scores_series = test_each_feature(
            reflectance_data, best_features, best_features[i], species_labels, THRESHOLD)
        scores_series.sort_values(ascending=False,inplace=True)
        best_features[i] = scores_series.index[0]
        features += list(scores_series.index)
        scores += list(scores_series.values)
    return best_features, features, scores

def SHAP_values(features_df, labels_df, state=42):
    '''
    function that performs a lighgbm classification on selected dataset
    returning the accuracy on test data
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        features_df, labels_df, test_size=0.2, random_state=state)
    model = lgb.LGBMClassifier(learning_rate=0.4, max_depth=5, num_leaves=10, random_state=state)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)],
              eval_metric='logloss', feature_name=list(features_df.columns),
              callbacks=[lgb.log_evaluation(period=-1), lgb.early_stopping(stopping_rounds=10)])
    print(f'Training accuracy {model.score(x_train,y_train)}')
    print(f'Testing accuracy {model.score(x_test,y_test)}')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    sv = explainer(x_test)
    #shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test.iloc[0,:],matplotlib=True)
    cmap = ListedColormap(["red","green","black"])


    shap.summary_plot(shap_values, x_test, class_names=model.classes_, color=cmap, show = False)               
    plt.title("SHAP Feature Importance Top 4 from 350nm to 750nm",fontsize = 20)
    plt.ylabel("Fetaures", fontsize = 16)
    plt.xlabel("mean(|SHAP value|)",fontsize = 16)
    plt.show()
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(features_df.columns, sum(vals))), columns=['feature','feature_importance'])
    feature_importance.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    feature_importance.head()
    plt.scatter(list(map(int,feature_importance["feature"])),list(map(float,feature_importance["feature_importance"])))
    plt.show()
    return feature_importance

def filter_feature_importance(all_features_df,feature_importance,labels_df):
    correlated = []
    to_save = []
    features = list(feature_importance["feature"])
    while len(features) > 0:
        correlated += correlated_features(all_features_df, features[0], labels_df, 0.95)
        to_save.append(features[0])
        features = [x for x in features if x not in correlated]
    return to_save

def plot_for_best(all_features,selected_features,labels,THRESHOLD,individual_features):
    a, wavelengths, accuracy = find_best(all_features, selected_features, labels, THRESHOLD)
    individual_features = list(map(int,individual_features))
    a = list(map(int,a))
    a.sort()
    cmap = cm.get_cmap("viridis")
    # individual_feature = [individual_features[0],individual_features[-1]]
    # for i in individual_features[1:-1]:
    #     individual_feature.extend([i,i])
    # individual_feature.sort()
    wavelengths = list(map(int,wavelengths))
    plt.scatter(wavelengths,accuracy)
    for i in a:
        plt.vlines(x=i,ymin=0,ymax=1,colors="green",label=f'{i}nm')
    color_individual_features = [(x-min(individual_features))/max(individual_features) for x in individual_features]
    print(color_individual_features)
    for i in range(0,len(individual_features)-1):
        plt.fill_betweenx(x1=individual_features[i],x2=individual_features[i+1],y=[0,1],color=cmap(color_individual_features[i]),alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
    plt.show()

x = map(str,list(range(350,1000)))
feature_importance_list = SHAP_values(reflectance_data[x],species_labels)
individual_features = filter_feature_importance(reflectance_data,feature_importance_list,species_labels)
individual_features = list(map(int,individual_features))
individual_features.sort()
#small_individual_features = [x for x in individual_features if x<1000]
plot_for_best(reflectance_data,['754', '680', '396', '512'],species_labels,0.95,individual_features)

print("Finshed")

#['754', '680', '396', '512'] 0.8375
