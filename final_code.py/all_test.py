import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import median_abs_deviation
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram

from skfeature.function.similarity_based import fisher_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.decomposition import PCA, FactorAnalysis

import mpl_scatter_density 

from xgboost import XGBClassifier
from skrebate import ReliefF
import lightgbm as lgb


all_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\SurinameMangrove_SpectralData.csv")
rwb_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")
rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv")
df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision5.csv")

labels = {"Black": "black", "White": "green",
          "Red": "red", "na": 'blue', 'Mud': 'brown'}
all_df = all_df.drop(["SpectraID", 'WhiteReference', 'ContactProbe', 'FibreOptic', 'SPAD_1', 'SPAD_2', 'SPAD_3',
                     'SPAD_Ave', 'Location', 'Lat', 'Long', 'StandAge', 'StandHealth', 'SurfaceDescription'], axis=1, inplace=False)
species_dictionary = {"Black": 0, "Red": 1, "White": 2}
model_dict = {'lightgbm': lgb.LGBMClassifier, 'random forest': RandomForestClassifier,
              'xgboost': XGBClassifier, 'gradient boosted': GradientBoostingClassifier, 'adaboost': AdaBoostClassifier, "svm": SVC}
params = {'lightgbm': {'learning_rate': 0.4, 'max_depth': 5, 'num_leaves': 10, 'random_state': 42}, 'random forest': {'n_estimators': 100, 'random_state': 42},
          'xgboost': {'n_estimators': 100, 'random_state': 42}, 'gradient boosted': {'n_estimators': 100, 'random_state': 42}, 'adaboost': {'n_estimators': 100, 'random_state': 42}, "svm": {'kernel': 'linear'}}
dim_dict = {'PCA':PCA,'FactorAnalysis':FactorAnalysis}


def average(a):
    '''
    return average of list a
    '''
    return sum(a)/len(a)

def label_to_float(labels_df):
    '''
    Tranforms the str labels to flaot labels
    '''
    new_labels=[]
    for i in range(0, len(labels_df)):
        new_labels.append(species_dictionary[labels_df.iat[i]])
    return np.array(new_labels)

def float_to_label(labels_df):
    '''
    Transforms the float labels to str labels
    '''
    dictionary_unpacked = dict(
        zip(species_dictionary.values(), species_dictionary.keys()))
    new_labels=[]
    for i in range(0, len(labels_df)):
        new_labels.append(dictionary_unpacked[labels_df.iat[i]])
    return np.array(new_labels)

def duplicate_df(df):
    '''
    duplicates all columns of the dataframe
    '''
    x = df.drop(['Species'],axis=1,inplace=False)
    y = df.Species
    x_new = x.copy(deep=True)
    columns = list(x.columns)
    new_columns = list(map(str,[x+2150 for x in list(map(int,columns))]))
    x_new = x_new.rename(columns = dict(zip(columns,new_columns)))
    x_final = pd.concat((y,x,x_new), axis=1)
    return x_final

def fisher_test(df):
    '''
    Returns Fisher score values in a list and wavelengths in a list
    '''
    x = df.drop(["Species"], axis=1, inplace=False)
    importance = fisher_score.fisher_score(x.to_numpy(), df.Species.to_numpy())
    normed_importance = [x/max(importance) for x in importance]
    return normed_importance

def information_gain_test(df):
    '''
    Returns Information Gain values in a list and wavelengths in a list
    '''
    x = df.drop(["Species"], axis=1, inplace=False)
    importance = mutual_info_classif(x, df.Species)
    normed_importance = [x/max(importance) for x in importance]
    return normed_importance

def variance_transformation(df):
    '''
    returns columns of dataframe with higher variance then the argument 
    '''
    x = df.drop(["Species"], axis=1, inplace=False)
    selector = VarianceThreshold(0.001)
    selector.fit(x)
    return df[df.columns[selector.get_support(indices=True)]]

def mean_absolute_difference(df):
    '''
    Returns mean absolute difference of dataframe 
    '''
    x = df.drop(['Species'],axis=1,inplace=False)
    return (x-x.mean(axis=0)).abs().mean(axis=0).to_numpy()

def median_absolute_difference(df):
    '''
    Returns meadian absolute difference of dataframe
    '''
    x=df.drop(['Species'],axis=1,inplace=False).to_numpy()
    return median_abs_deviation(x,axis=0)

def relief_fs(df):
    '''
    Returns 
    '''
    x=df.drop(['Species'],axis=1,inplace=False).to_numpy()
    y = label_to_float(df.Species)
    r = ReliefF(n_neighbors=10)
    r.fit(x,y)
    return r.feature_importances_

def anova_test(df):
    '''
    Finds the variance and mean difference between species at each wavelength
    '''
    df_b = df.query('Species == "Black"').iloc[:,1:]
    df_r = df.query('Species == "Red"').iloc[:,1:]
    df_w = df.query('Species == "White"').iloc[:,1:]
    f_all,p_all = stats.f_oneway(df_b.to_numpy(),df_r.to_numpy(),df_w.to_numpy())
    print(list(df.columns)[list(p_all).index(max(list(p_all)))])
    f_all = [x/(max(list(f_all))) for x in list(f_all)]
    p_all = [x/(max(list(p_all))) for x in list(p_all)]
    return f_all

def dimensionality_reduction(df,model):
    '''
    return df with 4 features using dimensionality reduction techniques
    '''
    X = df.drop(['Species'],axis=1,inplace=False)
    y = df.Species
    model = dim_dict[model](n_components=4)
    X = pd.DataFrame(model.fit_transform(X,y))
    return pd.concat((y,X),axis=1)

#print(dimensionality_reduction(rp_df,'PCA').head())

def stats_plot(df):
    '''
    Produces a plot for the statistical methods
    '''
    cmap = plt.cm.get_cmap('tab10')
    wavelengths = list(map(int,df.columns.to_numpy()[1:]))
    anova = anova_test(df)
    info = information_gain_test(df)
    releif = relief_fs(df)
    all_methods = [anova,info,releif]
    normalised_all_methods = [((x-min(x))/(max(x)-min(x))) for x in all_methods]
    count = 0
    names = ['ANOVA','Information Gain','ReliefF']
    colours = ['black','red','green']
    fig = plt.figure(dpi=240)
    for i in normalised_all_methods:
        plt.plot(wavelengths,i,alpha=0.8,label=names[count],c=colours[count])
        #plt.scatter(wavelengths,i,alpha=0.8,label=names[count],fc=colours[count],edgecolors=None,s=0.5)
        count += 1
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
        fancybox=True, shadow=True, ncol=3)
    plt.xlabel('Wavelength(nm)')
    plt.ylabel('Normalised F-Statistic')
    #plt.savefig(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Graphs\Filter Method\Filter methods combined.svg")
    plt.show()
    
#stats_plot(rp_df)
#information_gain_test(rp_df)

def sequential_feature_selection(df, model, final_number_of_features):
    '''
    Returns list of forwards feature selection, selected features
    '''
    x = df.drop(["Species"], axis=1, inplace=False)
    y = label_to_float(df.Species)
    sfs = SequentialFeatureSelector(
        model, n_features_to_select=final_number_of_features)
    sfs.fit(x, y)
    return (sfs)

def recursive_feature_selection(df, model, final_number_of_features, steps):
    '''
    Returns lost of RFE features 
    '''
    X = df.drop(["Species"], axis=1, inplace=False)
    y = label_to_float(df.Species)
    sfs = RFE(model, n_features_to_select=final_number_of_features,
              step=steps, verbose=1)
    return sfs.fit_transform(X,y),sfs.get_support(indices=True)

def rfe_with_crossvaliation_plot(df):
    '''
    boxplot of each 
    '''
    results = []
    names = []
    features = []
    for i,j in model_dict.items():
        if i != 'xgboost':
            names.append(i)
            new_X,feature = recursive_feature_selection(df,j(**params[i]),4,10)
            print(feature)
            features.append(feature)
            results.append(evaluate_model(j(**params[i]),new_X,df.Species).tolist())
    plt.figure(figsize=(16,8),dpi=240)
    plt.boxplot(results, labels=names, showmeans=True)
    means = list(map(average,results))
    new_features = []
    for i,val in enumerate(features):
        s = list(map(int,val))
        a = [list(df.columns)[j+1] for j in s]
        plt.text(x=1.3+i,y=means[i]-0.03,s='\n'.join(a))
        plt.text(x=0.6+i,y=means[i]-0.005,s=str(round(means[i],2)))
        new_features.append(a)

    plt.title('rfe top 4 cross validated')
    plt.xticks(rotation=45, ha='center')
    plt.ylabel('Accuracy (10 splits, 3 repeats)')
    plt.show()

def classifier_test(df, model_name):
    '''
    Returns model prediction score
    '''
    x = df.drop(["Species"], axis=1, inplace=False)
    if model_name == 'xgboost' and type(df.Species.iat[0]) == str:
        y = label_to_float(df.Species)
    else:
        y = df.Species
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    model = model_dict[model_name](**params[model_name])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if model_name == 'xgboost':
        return metrics.r2_score(y_test, y_pred)
    else:
        return metrics.accuracy_score(y_test, y_pred)

def classifier_importance(df, model_name):
    '''
    Returns model feature importance list and wavelengths list
    '''
    x = df.drop(["Species"], axis=1, inplace=False)
    if model_name == 'xgboost' and df.Species.iat[0] == str:
        y = label_to_float(df.Species)
    else:
        y = df.Species
    model = model_dict[model_name](n_estimators=100, random_state=42)
    model.fit(x, y)
    importance = model.feature_importances_
    cols = list(map(int, x.columns.tolist()))
    return cols, importance

# cols, importance = classifier_importance(duplicate_df(rp_df),'random forest')
# cols1, importance1 = classifier_importance(rp_df,'random forest')
# plt.plot(cols,importance)
# plt.plot(cols1,importance1)
# plt.vlines(x=2500,ymin=0,ymax=0.03,colors='red')
# plt.show()

def dendrogram_plot(df):
    '''
    plots dendrogram 
    '''
    X = df.drop(["Species"], axis=1, inplace=False)
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

def evaluate_model(model,X, y):
 '''
 Returns list of scores from 10 fold cross valiation with 3 repeats
 '''
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
 scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
 return scores

def boxplot_from_crossvalidation(df):
    '''
    Produces a boxplot of all models accuracy using cross validation on df
    '''
    results = []
    names = []
    for i,j in model_dict.items():
        if i != 'xgboost':
            names.append(i)
            results.append(evaluate_model(j(**params[i]),df[list(rp_df.columns)[1:]],df.Species))
    plt.figure(dpi=240)
    plt.boxplot(results, labels=names, showmeans=True)
    plt.xticks(rotation=45, ha='center')
    plt.ylabel('Accuracy (10 splits, 3 repeats)')
    plt.show()

def boxplot_from_crossvalidation_of_dim_reduct(df):
    '''
    Produces a boxplot of all reduction techniques accuracy using cross validation on lightgbm
    '''
    df_columns = list(df.columns)
    df_columns_int_species_dropped = list(map(int,df_columns[1:]))
    to_keep = [x for x in df_columns_int_species_dropped if x < 1000]
    df = df[['Species']+list(map(str,to_keep))]
    results = []
    names = []
    for i,j in dim_dict.items():
        df_old = df.copy()
        names.append(i)
        df_new = dimensionality_reduction(df_old,i)
        X = df_new[list(df_new.columns)[1:]]
        y = df_new.Species
        results.append(evaluate_model(model_dict['lightgbm'](**params['lightgbm']),X,y))
    plt.figure(dpi=240,figsize=(3,6))
    plt.boxplot(results, labels=names, showmeans=True)
    plt.xticks(rotation=45, ha='center')
    plt.ylabel('Accuracy (10 splits, 3 repeats)')
    plt.show()

def crossvalidation_boxplot_of_different_top_4(df,list_of_top,names):
    '''
    Produces a boxplot of different sets of top 4 wavelengths using cross validation on lightgbm
    '''
    results = []
    plt.figure(dpi=300,figsize=(6.5,6))
    for i,val in enumerate(list_of_top):
        X = df[val]
        y = df.Species
        res = evaluate_model(model_dict['lightgbm'](**params['lightgbm']),X,y)
        results.append(res)
        plt.text(1.25+i,average(res),'\n'.join(val),ha='left', va='center',size=12)
    plt.boxplot(results, labels=names, showmeans=True)
    plt.xticks(rotation=45, ha='center')
    plt.ylabel('Accuracy (10 splits, 3 repeats)')
    plt.show()

def make_groups(df):
    '''
    makes highly correlated groups
    '''
    corr_matrix = df.corr(method = 'pearson',numeric_only=True)
    corr_matrix_columns = list(map(int,corr_matrix.columns))

    points = []
    heights = []
    for i in corr_matrix_columns:
        correlated = [x for x in corr_matrix_columns if corr_matrix.loc[f'{i}',f'{x}'] > 0.9]
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

def groups_plot_using_density_plot(df):
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [(0, '#ffffff'),(1e-20, '#440053'),(0.2, '#404388'),(0.4, '#2a788e'),(0.6, '#21a784'),(0.8, '#78d151'),(1, '#fde624'),], N=256)
    df_columns = list(df.columns)
    df_columns_int_species_dropped = list(map(int,df_columns[1:]))
    to_keep = [x for x in df_columns_int_species_dropped if x < 1000]
    df = df[['Species']+list(map(str,to_keep))]
    groups = make_groups(df)
    selected_wavelengths = ['389', '512', '719', '767']
    selected_wavelengths = [389, 512, 719, 767]
    selected_groups = [groups[0],groups[1],groups[4],groups[6]]
    #selected_groups = [[389,392,395],[512,515,518]]
    results = []
    positions = []
    for i,val in enumerate(selected_groups):
        states = [0,1,2,3]
        states.remove(i)
        res_per_group = []
        position_per_group = []
        for j in val:
            chosen_wavelengths = [j]
            for k in states:
                chosen_wavelengths.append(int(selected_wavelengths[k]))
            chosen_wavelengths = list(map(str,list(set(chosen_wavelengths))))
            X = df[chosen_wavelengths]
            y = df.Species
            res = (evaluate_model(model_dict['lightgbm'](**params['lightgbm']),X,y).tolist())
            res_per_group += res
            position_per_group += ([j]*30)
        results += (res_per_group)
        positions += (position_per_group)
        print('here')
    fig = plt.figure(figsize=(12,6),dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    # for i,val in enumerate(positions):
    #     ax.scatter_density(val, results[i], cmap=white_viridis)
    ax.scatter_density(positions, results, cmap=white_viridis,dpi = 5)
    ax.set_xlim(300, 1050)
    ax.set_ylim(0, 1)
    # for i,val in enumerate(groups):
    #     ax.scatter(val,[i]*(len(val)))
    plt.show()

def groups_plot_using_boxplot(df):
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [(0, '#ffffff'),(1e-20, '#440053'),(0.2, '#404388'),(0.4, '#2a788e'),(0.6, '#21a784'),(0.8, '#78d151'),(1, '#fde624'),], N=256)
    df_columns = list(df.columns)
    df_columns_int_species_dropped = list(map(int,df_columns[1:]))
    to_keep = [x for x in df_columns_int_species_dropped if x < 1000]
    df = df[['Species']+list(map(str,to_keep))]
    groups = make_groups(df)
    selected_wavelengths = ['389', '512', '719', '767']
    selected_wavelengths = [389, 512, 719, 767]
    #selected_wavelengths = [389, 512]
    selected_groups = [groups[0],groups[1],groups[4],groups[6]]
    #selected_groups = [[389,392,395],[512,515,518]]
    results = []
    positions = []
    for i,val in enumerate(selected_groups):
        states = [0,1,2,3]
        states.remove(i)
        res_per_group = []
        position_per_group = []
        for j in val:
            chosen_wavelengths = [j]
            for k in states:
                chosen_wavelengths.append(int(selected_wavelengths[k]))
            chosen_wavelengths = list(map(str,list(set(chosen_wavelengths))))
            X = df[chosen_wavelengths]
            y = df.Species
            res = (evaluate_model(model_dict['lightgbm'](**params['lightgbm']),X,y).tolist())
            res_per_group.append(res)
            position_per_group.append(j)
        results.append(res_per_group)
        positions.append(position_per_group)
        print('here')
    plt.figure(figsize=(12,6),dpi=300)
    first = plt.boxplot(results[0],positions=positions[0],patch_artist=True)
    second = plt.boxplot(results[1],positions=positions[1],patch_artist=True)
    third = plt.boxplot(results[2],positions=positions[2],patch_artist=True)
    forth = plt.boxplot(results[3],positions=positions[3],patch_artist=True)
    for box in first['boxes']:
        box.set(facecolor = 'darkgreen' )
        box.set(color='darkgreen', linewidth=2)
    for whisker in first['whiskers']:
        whisker.set(color ='forestgreen',linewidth = 1.5,linestyle ="-")
    for flier in first['fliers']:
        flier.set(markerfacecolor ='red',markeredgecolor='red',markersize=3)
    for cap in first['caps']:
        cap.set(color ='red',linewidth = 0)
    for median in first['medians']:
        median.set(color ='black',linewidth = 3)

    for box in second['boxes']:
        box.set(facecolor = 'royalblue' )
        box.set(color='royalblue', linewidth=2)
    for whisker in second['whiskers']:
        whisker.set(color ='cornflowerblue',linewidth = 1.5,linestyle ="-")
    for flier in second['fliers']:
        flier.set(markerfacecolor ='red',markeredgecolor='red',markersize=3)
    for cap in second['caps']:
        cap.set(color ='red',linewidth = 0)
    for median in second['medians']:
        median.set(color ='black',linewidth = 3)

    for box in third['boxes']:
        box.set(facecolor = 'darkorange' )
        box.set(color='darkorange', linewidth=2)
    for whisker in third['whiskers']:
        whisker.set(color ='orange',linewidth = 1.5,linestyle ="-")
    for flier in third['fliers']:
        flier.set(markerfacecolor ='red',markeredgecolor='red',markersize=3)
    for cap in third['caps']:
        cap.set(color ='red',linewidth = 0)
    for median in third['medians']:
        median.set(color ='black',linewidth = 3)

    for box in forth['boxes']:
        box.set(facecolor = 'darkturquoise' )
        box.set(color='darkturquoise', linewidth=2)
    for whisker in forth['whiskers']:
        whisker.set(color ='turquoise',linewidth = 1.5,linestyle ="-")
    for flier in forth['fliers']:
        flier.set(markerfacecolor ='red',markeredgecolor='red',markersize=3)
    for cap in forth['caps']:
        cap.set(color ='red',linewidth = 0)
    for median in forth['medians']:
        median.set(color ='black',linewidth = 3)

    plt.ylabel('Accuracy (10 splits, 3 repeats)')
    plt.xlabel('Wavelengths (nm)')
    plt.xlim(340,1010)
    plt.xticks([350,400,450,500,550,600,650,700,750,800,850,900,950,1000],labels=[350,400,450,500,550,600,650,700,750,800,850,900,950,1000])
    plt.axvspan(350,506 , ymin=0, ymax=0.12, color='forestgreen',alpha=0.5) # 1 ##
    plt.axvspan(497,524 , ymin=0, ymax=0.12, color='cornflowerblue',alpha=0.5) # 2 ##
    plt.axvspan(515,605 , ymin=0, ymax=0.1, color='plum',alpha=0.5) #3 
    plt.axvspan(692,707 , ymin=0, ymax=0.1, color='plum',alpha=0.5) #3 
    plt.axvspan(590,692 , ymin=0, ymax=0.1, color='yellow',alpha=0.5) #4 
    plt.axvspan(701,725 , ymin=0, ymax=0.12, color='orange',alpha=0.5) #5 ##
    plt.axvspan(725,749 , ymin=0, ymax=0.1, color='lime',alpha=0.5) #6 
    plt.axvspan(737,947 , ymin=0, ymax=0.12, color='turquoise',alpha=0.5) #7 ##
    plt.vlines(x=[389, 512, 719, 767],ymin=0.5,ymax=0.6,colors = ['darkgreen','royalblue','darkorange','darkturquoise'])
    plt.ylim(0.5,0.95)
    plt.show()

#boxplot_from_crossvalidation(rp_df)
#rfe_with_crossvaliation(rp_df)
#boxplot_from_crossvalidation_of_dim_reduct(rp_df)
list_of_top_4s = [['389', '512', '719', '767'],['368', '518', '680', '755'],['518','671','755','953'],['512','674','773','989']]
names = ['Correlation\n+Feature Importance','SVM+Correlation','Correlation\n+ANOVA','LightGBM \n RFE']
#crossvalidation_boxplot_of_different_top_4(rp_df,list_of_top_4s,names)

#groups_plot_using_boxplot(rp_df)
#boxplot_from_crossvalidation_of_dim_reduct(rp_df)

print("Finished")
