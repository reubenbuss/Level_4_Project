import pandas as pd

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Wheat_data_all_headers.csv")
cols = list(df.columns)
new_cols = list(range(396,881))
cols = list(map(float, cols[2:]))
for i in new_cols:
    combine = []
    for j in cols:
        if i<j and j<i+1:
            combine.append(j)
    combine = list(map(str,combine))
    df[str(i)] = df[combine].mean(axis=1)
    df = df.drop(combine,axis=1,inplace=False)
#df = df.reindex(sorted(df.columns), axis=1)
new_cols = list(map(str, new_cols))
new_cols.insert(0,"Condition")
new_cols.insert(0,"Day")
df=df[new_cols]
df.to_csv("Wheat_data_reduced_precision.csv",index = False)
