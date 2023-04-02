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
to_keep = [x for x in df_columns_int_species_dropped if x > 1500]
df = mangrove_df[['Species']+list(map(str,to_keep))]
#df = pd.concat([mangrove_df,coffee_df],axis=0)


corr_matrix = df.corr(method = 'pearson', numeric_only=True)
corr_matrix_columns = list(map(int,corr_matrix.columns))

points = []
heights = []
for i in corr_matrix_columns:
    correlated = [x for x in corr_matrix_columns if corr_matrix.loc[f'{i}',f'{x}'] > 0.95]
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
    groups = make_groups()
    f_score = anova_test(df)
    f_score_dict = dict(zip(list(map(int,list(df.columns)[1:])),f_score))
    f_score_dict_reverse = dict(zip(f_score,list(map(int,list(df.columns)[1:]))[1:]))
    most_important_in_each_group = []
    average = []
    for i in groups:
        a=[]
        b=[]
        r=[]
        w=[]
        for j in i:
            a.append(f_score_dict[j])
            b.append(b_scores_dict[j])
            r.append(r_scores_dict[j])
            w.append(w_scores_dict[j])
        a_max = max(a)
        print([f_score_dict_reverse[a_max],sum(a)/len(a),sum(b)/len(b),sum(r)/len(r),sum(w)/len(w)])
        most_important_in_each_group.append(f_score_dict_reverse[a_max])
    most_important_in_each_group_str = list(map(str,most_important_in_each_group))

    table = [['Wavelength','Black','Red','White']]
    for i in most_important_in_each_group:
        table.append([i,b_scores_dict[i],r_scores_dict[i],w_scores_dict[i]])
    return table, average

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

#print(ANOVA_feature_selection_on_groups())
plot_of_highly_correlated_groups_4()
#print(make_groups())
#print(make_groups())
#print(lightgbm_feature_selection_on_groups())
print('Finished')

# [['Wavelength', 'Black', 'Red', 'White'], 
# [485, 169.21688063719975, 190.3480303864217, 0.01930284923310581], 
# [500, 117.03714338529595, 171.1885197233533, 1.4523448589309218], 
# [518, 19.550624222376104, 58.69641432912155, 8.343200770866861], 
# [671, 0.06053421224307657, 3.3230979107891434, 4.714777090731942],
# [725, 16.208113486882905, 1.2453800208267543, 10.627020828128744], 
# [755, 20.984981016820438, 1.0219258382057967, 15.82631200356645], 
# [953, 56.00288170308484, 8.140929016643678, 26.11801403214893], 
# [1337, 106.26924359465887, 4.070837831882048, 81.36662309967802], 
# [1337, 106.26924359465887, 4.070837831882048, 81.36662309967802], 
# [1447, 4.728412658481182, 254.23314618292946, 177.71747959914666]]

# [[485, 0.5529502428951213], 
# [500, 0.32085541263612993], 
# [518, 0.03421264813733395], 
# [671, 0.007102833015883052], 
# [725, 0.04021160446598583], 
# [755, 0.01952207759875648], 
# [953, 0.06465845700390596], 
# [1337, 0.19219747684832206], 
# [1337, 0.29812467427024675], 
# [1447, 0.6645764407554791]])

# [[350, 353, 356, 359, 362, 365, 368, 371, 374, 377, 380, 383, 386, 389, 392, 395, 398, 401, 404, 407, 410, 413, 416, 419, 422, 425, 428, 431, 434, 437, 440, 443, 446, 449, 452, 455, 458, 461, 464, 467, 470, 473, 476, 479, 482, 485, 488, 491, 494, 497, 500, 503, 506], 0.5529502428951213, 150.84698766445445, 143.7136806186224, 2.024870351136941]
# [[497, 500, 503, 506, 509, 512, 515, 518, 521, 524], 0.32085541263612993, 59.11099706512571, 106.25856370405086, 5.323532527318344]
# [[515, 518, 521, 524, 527, 530, 533, 536, 539, 542, 545, 548, 551, 554, 557, 560, 563, 566, 569, 572, 575, 578, 581, 584, 587, 590, 593, 596, 599, 602, 605, 695, 698, 701, 707], 0.03421264813733395, 3.5917961749319534, 11.741551093808024, 2.8101579679560866]
# [[590, 593, 596, 599, 602, 605, 608, 611, 614, 617, 620, 623, 626, 629, 632, 635, 638, 641, 644, 647, 650, 653, 656, 659, 662, 665, 668, 671, 674, 677, 680, 683, 686, 689, 692], 0.007102833015883052, 0.062087311065148275, 1.680836574651059, 1.8836822069910357]
# [[701, 707, 713, 719, 725], 0.04021160446598583, 13.160786245181365, 1.2811020361340324, 7.979986198505212]
# [[731, 737, 743, 749], 0.01952207759875648, 5.9928364770822045, 0.31928435061786026, 4.501337859621449]
# [[737, 743, 749, 755, 761, 767, 773, 779, 785, 791, 797, 803, 809, 815, 821, 827, 833, 839, 845, 851, 857, 863, 869, 875, 881, 887, 893, 899, 905, 911, 917, 923, 929, 935, 941, 947], 0.06465845700390596, 22.085124479999287, 5.216442908824948, 8.998755473077436]
# [[905, 911, 917, 923, 929, 935, 941, 947, 953, 959, 965, 971, 977, 983, 989, 995, 1001, 1007, 1013, 1019, 1025, 1031, 1037, 1043, 1049, 1055, 1061, 1067, 1073, 1079, 1085, 1091, 1097, 1103, 1109, 1115, 1121, 1127, 1133, 1139, 1145, 1151, 1157, 1163, 1169, 1175, 1181, 1187, 1193, 1199, 1205, 1211, 1217, 1223, 1229, 1235, 1241, 1247, 1253, 1259, 1265, 1271, 1277, 1283, 1289, 1295, 1301, 1307, 1313, 1319, 1325, 1331, 1337, 1343, 1349, 1355], 0.19219747684832206, 64.0991253768679, 10.309840285043732, 31.01003838495568]
# [[1313, 1319, 1325, 1331, 1337, 1343, 1349, 1355, 1361, 1367, 1373, 1379, 1385, 1391, 1397], 0.29812467427024675, 88.64288513662225, 10.430663647518474, 60.93165196398044]
# [[1397, 1407, 1417, 1427, 1437, 1447, 1457, 1467, 1477, 1487, 1497], 0.6645764407554791, 13.066881320267264, 191.4595393156765, 109.11286510880262]

# [0.5529502428951213, 150.84698766445445, 143.7136806186224, 2.024870351136941]  black and red
# [0.32085541263612993, 59.11099706512571, 106.25856370405086, 5.323532527318344]  black and red but worse
# [0.03421264813733395, 3.5917961749319534, 11.741551093808024, 2.8101579679560866] bad
# [0.007102833015883052, 0.062087311065148275, 1.680836574651059, 1.8836822069910357] bad
# [0.04021160446598583, 13.160786245181365, 1.2811020361340324, 7.979986198505212]  bad
# [0.01952207759875648, 5.9928364770822045, 0.31928435061786026, 4.501337859621449]  bad 
# [0.06465845700390596, 22.085124479999287, 5.216442908824948, 8.998755473077436]  bad 
# [0.19219747684832206, 64.0991253768679, 10.309840285043732, 31.01003838495568]  
# [0.29812467427024675, 88.64288513662225, 10.430663647518474, 60.93165196398044]
# [0.6645764407554791, 13.066881320267264, 191.4595393156765, 109.11286510880262] best for red and white 

# [485, 0.20568795433752415, 150.84698766445445, 143.7136806186224, 2.024870351136941] medium black and red
# [500, 0.11935268012128417, 59.11099706512571, 106.25856370405086, 5.323532527318344] low red
# [518, 0.012726515085684606, 3.5917961749319534, 11.741551093808024, 2.8101579679560866]
# [671, 0.0026421313885110556, 0.062087311065148275, 1.680836574651059, 1.8836822069910357]
# [725, 0.01495802338368274, 13.160786245181365, 1.2811020361340324, 7.979986198505212]
# [755, 0.007261876194651104, 5.9928364770822045, 0.31928435061786026, 4.501337859621449]
# [953, 0.02405183092446296, 22.085124479999287, 5.216442908824948, 8.998755473077436]
# [1337, 0.07149414680565257, 64.0991253768679, 10.309840285043732, 31.01003838495568]
# [1667, 0.1681344115278582, 152.10213852752344, 31.925371041516023, 57.96264937853665] medium black
# [2067, 0.2753471633612133, 33.50792133849316, 225.978683435067, 92.56699894695336] medium red low white
# [2442, 0.8611625740237027, 12.20429566211621, 551.7475833671941, 357.21014763884585] very high red 

# [[350, 353, 356, 359, 362, 365, 368, 371, 374, 377, 380, 383, 386, 389, 392, 395, 398, 401, 404, 407, 410, 413, 416, 419, 422, 425, 428, 431, 434, 437, 440, 443, 446, 449, 452, 455, 458, 461, 464, 467, 470, 473, 476, 479, 482, 485, 488, 491, 494, 497, 500, 503, 506], 
# [497, 500, 503, 506, 509, 512, 515, 518, 521, 524], 
# [515, 518, 521, 524, 527, 530, 533, 536, 539, 542, 545, 548, 551, 554, 557, 560, 563, 566, 569, 572, 575, 578, 581, 584, 587, 590, 593, 596, 599, 602, 605, 695, 698, 701, 707], 
# [590, 593, 596, 599, 602, 605, 608, 611, 614, 617, 620, 623, 626, 629, 632, 635, 638, 641, 644, 647, 650, 653, 656, 659, 662, 665, 668, 671, 674, 677, 680, 683, 686, 689, 692], 
# [701, 707, 713, 719, 725], 
# [731, 737, 743, 749], 
# [737, 743, 749, 755, 761, 767, 773, 779, 785, 791, 797, 803, 809, 815, 821, 827, 833, 839, 845, 851, 857, 863, 869, 875, 881, 887, 893, 899, 905, 911, 917, 923, 929, 935, 941, 947], 
# [905, 911, 917, 923, 929, 935, 941, 947, 953, 959, 965, 971, 977, 983, 989, 995, 1001, 1007, 1013, 1019, 1025, 1031, 1037, 1043, 1049, 1055, 1061, 1067, 1073, 1079, 1085, 1091, 1097, 1103, 1109, 1115, 1121, 1127, 1133, 1139, 1145, 1151, 1157, 1163, 1169, 1175, 1181, 1187, 1193, 1199, 1205, 1211, 1217, 1223, 1229, 1235, 1241, 1247, 1253, 1259, 1265, 1271, 1277, 1283, 1289, 1295, 1301, 1307, 1313, 1319, 1325, 1331, 1337, 1343, 1349, 1355], 
# [1313, 1319, 1325, 1331, 1337, 1343, 1349, 1355, 1361, 1367, 1373, 1379, 1385, 1391, 1397, 1507, 1517, 1527, 1537, 1547, 1557, 1567, 1577, 1587, 1597, 1607, 1617, 1627, 1637, 1647, 1657, 1667, 1677, 1687, 1697, 1707, 1717, 1727, 1737, 1747, 1757, 1767, 1777, 1787, 1797, 1807, 1817, 1827, 1837, 1847, 1857], 
# [1397, 1407, 1417, 1427, 1437, 1447, 1457, 1467, 1477, 1487, 1497, 1507, 1867, 1877, 2057, 2067, 2077, 2087, 2097, 2112, 2127, 2142, 2187, 2202, 2217, 2232, 2247, 2262, 2277, 2292, 2307, 2322, 2337], 
# [1887, 1897, 1907, 1917, 1927, 1937, 1947, 1957, 1967, 1977, 1987, 1997, 2007, 2017, 2027, 2037, 2382, 2397, 2412, 2427, 2442, 2457, 2472, 2487]]
