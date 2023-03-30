import pandas as pd


df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data.csv")

new_cols = list(range(350, 888))
cols = list(map(float, list(df.columns)[1:]))
combos = []
for i, val in enumerate(new_cols):
    if val != 887:
        combos.append(list(map(str,[x for x in cols if x>new_cols[i] and x<new_cols[i+1]])))
df1 = df.assign(**{
    str(maincol): df.loc[:, combo].mean(axis=1)
    for maincol, combo in zip(new_cols, combos)
}).loc[:, map(str, new_cols[:-1])]
df1 = df1.div(100)
df1 = pd.concat([df.Species,df1],axis=1)    


df1.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data_reduced_precision_to_1.csv", index=False)

print("Finished")


# a = [1,2,3,4,5,6,7,8,9]
# b=[3,7]
# c = [x for x in a if x>min(b) and x<max(b)]
# print(c)
