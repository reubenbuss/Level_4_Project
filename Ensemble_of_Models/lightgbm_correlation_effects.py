import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
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
    best_features = selected_features.copy()
    for i in range(0,len(selected_features)):
        scores_series = test_each_feature(
            reflectance_data, best_features, best_features[i], species_labels, THRESHOLD)
        scores_series.sort_values(ascending=False,inplace=True)
        best_features[i] = scores_series.index[0]
    return best_features


a = find_best(reflectance_data, ["750", "680", "351", "511"], species_labels, 0.95)
print(classifier(reflectance_data[["750", "680", "351", "511"]],species_labels))
print(classifier(reflectance_data[a],species_labels))
print(a)
print("Finshed")

#['754', '680', '396', '512'] 0.8375
