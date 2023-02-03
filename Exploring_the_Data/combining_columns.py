import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv") 
x_df = df.drop(["Species"],axis=1,inplace=False)
y_ser = df.Species

