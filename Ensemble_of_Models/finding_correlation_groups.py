import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.svm import SVC

import lightgbm as lgb

mangrove_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv")

#coffee_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision.csv")

#df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_and_Coffee_data.csv")

#select max wavelength
df_columns = list(mangrove_df.columns)
df_columns_int_species_dropped = list(map(int,df_columns[1:]))
to_keep = [x for x in df_columns_int_species_dropped if x < 1000]
df = mangrove_df[['Species']+list(map(str,to_keep))]
#df = pd.concat([mangrove_df,coffee_df],axis=0)

corr_matrix = df.corr(method = 'pearson', numeric_only=True)
corr_matrix_columns = list(map(int,corr_matrix.columns))

points = []
heights = []
for i in corr_matrix_columns:
    correlated = [x for x in corr_matrix_columns if corr_matrix.loc[f'{i}',f'{x}'] > 0.9]
    points.append(correlated)
    heights.append([i]*(len(correlated)))

def plot_of_highly_correlated_groups():
    '''
    plot of highly correlated wavelengths
    '''
    fig = plt.figure(dpi=300)

    for i,val in enumerate(points):
        plt.scatter(x=val,y=heights[i])


    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Wavelengths (nm)')
    plt.show()

def anova_test(df):
    '''
    Finds the variance and mean difference between species at each wavelength
    '''
    df_b = df.query('Species == "Black"').iloc[:,1:]
    df_r = df.query('Species == "Red"').iloc[:,1:]
    df_w = df.query('Species == "White"').iloc[:,1:]
    f_all,p_all = stats.f_oneway(df_b.to_numpy(),df_r.to_numpy(),df_w.to_numpy())
    f_all = [x/(max(list(f_all))) for x in list(f_all)]
    p_all = [x/(max(list(p_all))) for x in list(p_all)]
    return f_all

def plot_of_highly_correlated_groups_2():
    '''
    plot of highly correlated wavelengths with colourmap to anova score
    '''
    scores = []
    for i in points:
        scores.append(anova_test(df[['Species']+list(map(str,i))]))
    fig = plt.figure(dpi=300)

    for i,val in enumerate(points):
        plt.scatter(x=val,y=heights[i],c=scores[i],cmap='viridis',alpha=scores[i],s=scores[i])

    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Wavelengths (nm)')
    plt.show()

def plot_of_highly_correlated_groups_3():
    '''
    plot of highly correlated wavelengths with colourmap to anova score different
    '''
    scores_each = anova_test(df)
    scores_dict = dict(zip(corr_matrix_columns,scores_each))
    fig = plt.figure(dpi=300)
    scores = []
    for i in points:
        a = []
        for j in i: 
            score = scores_dict[j]
            a.append(score)
        scores.append(a)
    for i,val in enumerate(points):
        plt.scatter(x=val,y=heights[i],c=scores[i],cmap='viridis',s=scores[i],vmin=0,vmax=1,alpha=scores[i])
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    cb.set_label('ANOVA Score')
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Wavelengths (nm)')
    plt.show()

def plot_of_highly_correlated_groups_4():
    '''
    plot of highly correlated wavelengths as cluster lines
    '''
    groups = []
    for i in points:
        points_sets = [set(x) for x in points]
        groups_set = [set(x) for x in groups]
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

    for i,vals in enumerate(groups):
        plt.scatter(vals,[i]*len(vals))
    plt.show()

def plot_of_highly_correlated_groups_5():
    '''
    plot of highly correlated wavelengths as cluster lines with anova colour map
    '''
    scores_each = anova_test(df)
    scores_dict = dict(zip(corr_matrix_columns,scores_each))
    scores=[]
    groups = []
    for i in points:
        points_sets = [set(x) for x in points]
        groups_set = [set(x) for x in groups]
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
    for i in groups:
        a = []
        for j in i: 
            score = scores_dict[j]
            a.append(score)
        scores.append(a)
    fig = plt.figure(dpi=300)
    for i,vals in enumerate(groups):
        plt.scatter(vals,[i]*len(vals),c=scores[i],cmap='viridis',s=np.array(scores[i])*50,vmin=0,vmax=1)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    cb.set_label('ANOVA Score')
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Highly Correlated Groups')
    plt.show()

def plot_of_highly_correlated_groups_6():
    '''
    plot of highly correlated wavelengths as cluster lines with anova colour map
    '''
    scores_each = anova_test(df)
    df_b = df.query('Species == "Black"').iloc[:,1:]
    df_r = df.query('Species == "Red"').iloc[:,1:]
    df_w = df.query('Species == "White"').iloc[:,1:]
    df_br = df.query('Species == "Black" | Species == "Red"').iloc[:,1:]
    df_bw = df.query('Species == "Black" | Species == "White"').iloc[:,1:]
    df_rw = df.query('Species == "Red" | Species ==  "White"').iloc[:,1:]
    f_br_w,_ = stats.f_oneway(df_br.to_numpy(),df_w.to_numpy())
    f_bw_r,_ = stats.f_oneway(df_bw.to_numpy(),df_r.to_numpy())
    f_rw_b,_ = stats.f_oneway(df_rw.to_numpy(),df_b.to_numpy())
    f_br_w = [x/(max(list(f_br_w))) for x in list(f_br_w)]
    f_bw_r = [x/(max(list(f_bw_r))) for x in list(f_bw_r)]
    f_rw_b = [x/(max(list(f_rw_b))) for x in list(f_rw_b)]
    w_scores_dict = dict(zip(corr_matrix_columns,f_br_w))
    r_scores_dict = dict(zip(corr_matrix_columns,f_bw_r))
    b_scores_dict = dict(zip(corr_matrix_columns,f_rw_b))
    scores_dict = dict(zip(corr_matrix_columns,scores_each))
    scores=[]
    groups = []
    colours = []
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
    for i in groups:
        a = []
        b = []
        for j in i: 
            if w_scores_dict[j] > r_scores_dict[j] and w_scores_dict[j] > b_scores_dict[j]:
                b.append('green')
                a.append(w_scores_dict[j])
            elif r_scores_dict[j] > w_scores_dict[j] and r_scores_dict[j] > b_scores_dict[j]:
                b.append('red')
                a.append(r_scores_dict[j])
            else:
                b.append('black')
                a.append(b_scores_dict[j])
        scores.append(a)
        colours.append(b)
    fig = plt.figure(dpi=300,figsize=(12,4))
    # plt.scatter(corr_matrix_columns,np.array(f_rw_b)*(len(groups)-1),c='black',label='Black Mangrove',s=0.1)
    # plt.scatter(corr_matrix_columns,np.array(f_bw_r)*(len(groups)-1),c='red',label='Red Mangrove',s=0.1)
    # plt.scatter(corr_matrix_columns,np.array(f_br_w)*(len(groups)-1),c='green',label='White Mangrove',s=0.1)
    plt.scatter(350,-1,c='black',label='Black Mangrove',s=0.1)
    plt.scatter(350,-1,c='red',label='Red Mangrove',s=0.1)
    plt.scatter(350,-1,c='green',label='White Mangrove',s=0.1)
    plt.ylim(-0.5,len(groups)-0.5)
    for i,vals in enumerate(groups):
        plt.scatter(vals,[i]*len(vals),c=colours[i],s=np.array(scores[i])*50)
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Highly Correlated Groups')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),fancybox=True, shadow=True, ncol=3,markerscale=10)
    plt.show()

#plot_of_highly_correlated_groups_6()

def make_groups():
    '''
    makes highly correlated groups
    '''
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

def lightgbm_feature_selection_on_groups():
    '''
    returns list of most important features in each group using feature selection in lightgbm
    '''
    groups = make_groups()
    x = df.drop(["Species"], axis=1, inplace=False)
    x_columns = list(map(int,x.columns))
    y = df.Species
    model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10,random_state=42)
    model.fit(x, y)
    importance = model.feature_importances_
    importance_dict = dict(zip(x_columns,importance))
    importance_dict_reverse = dict(zip(importance,x_columns))

    most_important_in_each_group = []
    for i in groups:
        a=[]
        for j in i:
            a.append(importance_dict[j])
        a_max = max(a)
        print(a_max)
        most_important_in_each_group.append(importance_dict_reverse[a_max])
    most_important_in_each_group_str = list(map(str,most_important_in_each_group))
    return most_important_in_each_group_str

def svm_feature_selection_on_groups():
    '''
    returns list of most important features in each group using feature selection in svm
    '''
    groups = make_groups()
    x = df.drop(["Species"], axis=1, inplace=False)
    x_columns = list(map(int,x.columns))
    y = df.Species
    model = SVC(kernel='linear',C=10000)
    model.fit(x, y)
    importance = model.coef_.tolist()[0]
    importance_dict = dict(zip(x_columns,importance))
    importance_dict_reverse = dict(zip(importance,x_columns))

    most_important_in_each_group = []
    for i in groups:
        a=[]
        for j in i:
            a.append(importance_dict[j])
        a_max = max(a)
        print(a_max)
        most_important_in_each_group.append(importance_dict_reverse[a_max])
    most_important_in_each_group_str = list(map(str,most_important_in_each_group))
    return most_important_in_each_group_str

def ANOVA_feature_selection_on_groups():
    '''
    returns list of most important features in each group using ANOVA feature selection
    '''
    groups = make_groups()
    f_score = anova_test(df)
    f_score_dict = dict(zip(list(map(int,list(df.columns)[1:])),f_score))
    f_score_dict_reverse = dict(zip(f_score,list(map(int,list(df.columns)[1:]))[1:]))
    most_important_in_each_group = []
    for i in groups:
        a=[]
        for j in i:
            a.append(f_score_dict[j])
        a_max = max(a)
        most_important_in_each_group.append(f_score_dict_reverse[a_max])
    most_important_in_each_group_str = list(map(str,most_important_in_each_group))
    df_b = df.query('Species == "Black"').iloc[:,1:]
    df_r = df.query('Species == "Red"').iloc[:,1:]
    df_w = df.query('Species == "White"').iloc[:,1:]
    df_br = df.query('Species == "Black" | Species == "Red"').iloc[:,1:]
    df_bw = df.query('Species == "Black" | Species == "White"').iloc[:,1:]
    df_rw = df.query('Species == "Red" | Species ==  "White"').iloc[:,1:]
    f_br_w,_ = stats.f_oneway(df_br.to_numpy(),df_w.to_numpy())
    f_bw_r,_ = stats.f_oneway(df_bw.to_numpy(),df_r.to_numpy())
    f_rw_b,_ = stats.f_oneway(df_rw.to_numpy(),df_b.to_numpy())
    w_scores_dict = dict(zip(corr_matrix_columns,f_br_w))
    r_scores_dict = dict(zip(corr_matrix_columns,f_bw_r))
    b_scores_dict = dict(zip(corr_matrix_columns,f_rw_b))
    table = [['Wavelength','Black','Red','White']]
    for i in most_important_in_each_group:
        table.append([i,b_scores_dict[i],r_scores_dict[i],w_scores_dict[i]])
    return table

def ROC_Curve_for_selected_wavelengths(df):
    '''
    Produces a roc curve for selected wavelengths
    '''
    selected_ones = ['389', '512', '719', '767']
    x=df.drop(['Species'],axis=1,inplace=False)
    y=df.Species
    (X_train,X_test,y_train,y_test,) = train_test_split(x[selected_ones], y, test_size=0.3, stratify=y, random_state=42)
    model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10,random_state=42)
    y_score = model.fit(X_train, y_train).predict_proba(X_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    class_of_interest = "Black"
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="black",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)",c='blue')
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.title("One-vs-Rest ROC curves:\nBlack vs (Red & White)")
    plt.legend()
    plt.show()

print(ANOVA_feature_selection_on_groups())

#print(make_groups())
#print(make_groups())
#print(lightgbm_feature_selection_on_groups())
print('Finished')

