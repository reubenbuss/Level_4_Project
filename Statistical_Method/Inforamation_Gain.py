import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import matplotlib.pyplot as plt

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
df = pd.read_csv(file)

y=df.Species

#x=df.drop(["Species"],axis=1,inplace=False)
#x=df[["1892","671","1652","720"]]
x=df[list(map(str,list(range(350,2501,50))))]


importance = mutual_info_classif(x,y)
feat_importance = pd.Series(importance, x.columns[0: len(x.columns)])
#print(feat_importance.sort_values(ascending = False))
feat_importance = feat_importance.nlargest(n=10,keep = "first")
#feat_importance = feat_importance.sort_values(ascending=False).head(10)
feat_importance = feat_importance.sort_values(ascending = True)
feat_importance.plot(kind='barh', color='teal',
    title = "Information Gain Top 10 with Wavelength Jumps of 50")

#print(feat_importance.nlargest(n=3,keep = "first"))