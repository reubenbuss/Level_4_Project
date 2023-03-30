import pandas as pd

df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Calibrated_data_outliers_removed.csv")

# cols = list(df.columns)
# new_cols = list(range(350, 702, 3)) + list(range(707, 1398, 6)) + \
#     list(range(1407, 2098, 10)) + list(range(2110, 2501, 13))
# cols = list(map(int, cols[1:]))
# for i, val in enumerate(new_cols):
#     combine = []
#     if i == 0:
#         combine.append("350")
#     for j in cols:
#         if new_cols[i] < j <= new_cols[i+1]:
#             combine.append(j)
#     combine = list(map(str, combine))
#     print(combine)
#     df[str(val)] = df[combine].mean(axis=1)
#     df = df.drop(combine, axis=1, inplace=False)
# #df = df.reindex(sorted(df.columns), axis=1)
# df.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision.csv", index=False)

# print("Finished")

# new_cols = list(range(350, 702, 3)) + list(range(707, 1398, 6)) + \
#     list(range(1407, 2098, 10)) + list(range(2112, 2488, 15)) + [2501]
# cols = list(map(int, list(df.columns)[1:]))
# df1 = df.copy()
# for i, val in enumerate(new_cols):
#     if val == 2501:
#         break
#     combine = list(map(str, range(new_cols[i], new_cols[i+1])))
#     print(combine)
#     df1 = df1.drop(combine, axis=1, inplace=False)
#     df1[str(val)] = df[combine].mean(axis=1)
#     #df1 = pd.concat([df1,df[combine].mean(axis=1)],keys=[val])

# df1.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision3.csv", index=False)

# print("Finished")

# df1 = pd.DataFrame({'1': [1, 2, 3, 4], 
#                     '2': [5, 6, 7, 8], 
#                     '3': [9, 10, 11, 12], 
#                     '4': [13, 14, 15, 16],
#                     '5': [17, 18, 19, 20], 
#                     '6': [21, 22, 23, 24], 
#                     '7': [25, 26, 27, 28]})
# df2 = df1.copy()
# # df2 should have columns 1,2,5 which are the mean of df1 columns [1],[2,3,4],[5,6,7]
# new_cols = [1, 2, 5, 8]
# for i, val in enumerate(new_cols):
#     if val == 8:
#         break
#     #All the column names are integers as str
#     combine = list(map(str, range(new_cols[i], new_cols[i+1])))
#     df2 = df2.drop(combine, axis=1, inplace=False)
#     df2[str(val)] = df1[combine].mean(axis=1)
# print(df2)



new_cols = list(range(350, 702, 3)) + list(range(707, 1398, 6)) + \
    list(range(1407, 2098, 10)) + list(range(2112, 2488, 15)) + [2501]
cols = list(map(int, list(df.columns)[1:]))
combos = []
for i, val in enumerate(new_cols):
    if val != 2501:
        combos.append(list(map(str, range(new_cols[i], new_cols[i+1]))))
df1 = df.assign(**{
    str(maincol): df.loc[:, combo].mean(axis=1)
    for maincol, combo in zip(new_cols, combos)
}).loc[:, map(str, new_cols[:-1])]
df1 = pd.concat([df.Species,df1],axis=1)    
df1.to_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Calibrated_data_reduced_precision.csv", index=False)

print("Finished")


# import pandas
# df1 = pandas.DataFrame({
#     '1': [1, 2, 3, 4], 
#     '2': [5, 6, 7, 8], 
#     '3': [9, 10, 11, 12], 
#     '4': [13, 14, 15, 16],
#     '5': [17, 18, 19, 20], 
#     '6': [21, 22, 23, 24], 
#     '7': [25, 26, 27, 28],
# })

# # df2 should have columns 1,2,5 which are the mean of df1 columns [1],[2,3,4],[5,6,7]

# new_cols = [1, 2, 5, 8]
# combos = []
# for i, val in enumerate(new_cols):
#     if val != 8:
#         #All the column names are integers as str
#         combos.append(list(map(str, range(new_cols[i], new_cols[i+1]))))
# print(combos)
# df2 = df1.assign(**{
#     str(maincol): df1.loc[:, combo].mean(axis="columns")
#     for maincol, combo in zip(new_cols, combos)
# }).loc[:, map(str, new_cols[:-1])]
# print(df2)
# print(df1)
