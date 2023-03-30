import matplotlib.pyplot as plt
import pandas as pd

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
#df = pd.read_csv(file)
df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision_to_1.csv")


mangrove_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv")

coffee_df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision_to_1.csv")

#select max wavelength
df_columns = list(mangrove_df.columns)
df_columns_int_species_dropped = list(map(int,df_columns[1:]))
to_keep = [x for x in df_columns_int_species_dropped if x < 887]
mangrove_df = mangrove_df[['Species']+list(map(str,to_keep))]

df = pd.concat([mangrove_df,coffee_df],axis=0)


y=df.Species

x=df.drop(["Species"],axis=1,inplace=False)
#x.columns = x.columns.astype(int)


#x=df[["1892","671","1652","720"]]
#x=df[["1441","680","725","366"]]
#x=df[["750","680","351","390"]]
#x=df[list(map(str,list(range(350,883))))]

def create_heatmap(df):
    f = plt.figure(dpi=300)
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(0,df.shape[1],100),list(map(int,list(df.columns)[::100])), fontsize=14, rotation=45)
    plt.yticks(range(0,df.shape[1],100),list(map(int,list(df.columns)[::100])), fontsize=14)
    # plt.xticks([0,150,400,650,900,1150,1400,1650,1900,2150],[350,500,750,1000,1250,1500,1750,2000,2250,2500], fontsize=14, rotation=45)
    # plt.yticks([0,150,400,650,900,1150,1400,1650,1900,2150],[350,500,750,1000,1250,1500,1750,2000,2250,2500], fontsize=14)    
    # plt.xticks([0,33,67,100,125,142],[350,450,550,650,750,850], fontsize=14, rotation=45)
    # plt.yticks([0,33,67,100,125,142],[350,450,550,650,750,850], fontsize=14)    
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    #plt.text(x=0.8, y=1.3, s='Reduced Precison')
    plt.ylabel('Wavelength (nm)')
    plt.xlabel('Wavelength (nm)')
    plt.show()

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

#print("Top Absolute Correlations")
#print(get_top_abs_correlations(x, 10))
create_heatmap(x)
