
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Black Data Cleaned.csv"
df = pd.read_csv(file)

x = df.iloc[:,815]
m = x.mean()
std = x.std()
y = (x-m)/std
fig = plt.figure(dpi=2400)
fig = sm.qqplot(x)
plt.title("Q-Q Plot of Normally Distributed 800nm Black Data")
plt.show()
