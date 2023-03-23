import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats

df = pd.read_csv(r'C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv')

# select max wavelength
# df_columns = list(df.columns)
# df_columns_int_species_dropped = list(map(int,df_columns[1:]))
# to_keep = [x for x in df_columns_int_species_dropped if x < 1000]
# df = df[['Species']+list(map(str,to_keep))]

corr_matrix = df.corr('pearson')
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

def plot_of_highly_correlated_groups_4():
    '''
    plot of highly correlated wavelengths with final groups lines
    '''
    fig = plt.figure(dpi=300)

    for i,val in enumerate(points):
        plt.scatter(x=val,y=heights[i])
        if val[0] in final_groups:
            plt.axhline(y=val[0], color='r', linestyle='-',linewidth=0.1)

    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Wavelengths (nm)')
    plt.show()

# plot_of_highly_correlated_groups_4()

# print('Finished')

# a = [1,2,3,4]
# b = [5,6,7]
# c = [2,3,4,5]

# print(len(set(a).intersection(set(b)))/len(set(a).union(set(b))))
