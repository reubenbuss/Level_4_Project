import matplotlib.pyplot as plt
import pandas as pd

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
df = pd.read_csv(file)

y=df.Species

#x=df.drop(["Species"],axis=1,inplace=False)
#x=df[["1892","671","1652","720"]]
#x=df[["1441","680","725","366"]]
#x=df[["750","680","351","390"]]
x=df[list(map(str,list(range(350,401))))]

def create_heatmap(df):
    f = plt.figure(dpi=2400)
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(0,len(df.columns),430), range(350,len(df.columns)+350,430), fontsize=14, rotation=45)
    plt.yticks(range(0,len(df.columns),430), range(350,len(df.columns)+350,430), fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix 350nm to 2500nm', fontsize=16);
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