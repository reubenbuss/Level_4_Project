import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap
from matplotlib import cm


file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack.csv"
data = pd.read_csv(file)
data.drop(["SpectraID","WhiteReference","ContactProbe",
    "FibreOptic","SPAD_1","SPAD_2","SPAD_3","SPAD_Ave",
    "Location","Lat","Long","StandAge","StandHealth",
    "SurfaceDescription"],axis = 1,inplace=True)

y=data.Species

#all
x=data.drop(["Species"],axis=1,inplace=False)
#first 40
#x=data[["350","356","713","351","1890","1888","1891","375","1893","2399","1899","355","1651","360","723","2241","1675","717","352","729","368","1998","1892","1831","353","1894","728","512","1882","1878","1123","2014","690","664","661","366","2132","1901","1886","1648"]]
#first100
#x=data[['350', '356', '351', '713', '1890', '1888', '1891', '375', '1893', '2399', '1899', '355', '1651', '360', '723', '717', '1675', '2241', '352', '368', '729', '1998', '1892', '353', '1831', '1894', '728', '512', '1123', '1882', '1878', '2014', '366', '690', '664', '661', '1648', '721', '737', '1886', '513', '354', '2132', '1901', '1619', '357', '378', '1879', '361', '367', '722', '369', '725', '727', '1975', '377', '718', '1108', '1652', '736', '930', '1889', '2304', '671', '1426', '1884', '2500', '2472', '2375', '1342', '2244', '659', '662', '1407', '673', '391', '649', '1900', '732', '2498', '1984', '747', '362', '726', '2022', '1877', '2003', '669', '672', '731', '2491', '390', '1986', '1001', '2469', '1887', '2035', '1658', '1895']]
#x=data[["1890","1998","1658","1901"]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
fig = plt.figure(dpi=2400)

def Max_Depth(x_train,x_test,y_train,y_test):
    plt.title("Optimising Depth of LightGBM Classifier")
    plt.ylabel("Test Dataset Score")
    plt.xlabel("Depth of Tree")
    X = [3,4,5,6,7,8]
    Y =[]
    for i in X:
        model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=i,random_state=42)
        model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)]
                ,eval_metric='logloss',verbose = -1)
        Y.append(model.score(x_test,y_test))
    plt.scatter(X,Y,c="green")
    plt.show()

def Num_Leaves(x_train,x_test,y_train,y_test):
    plt.title("Optimising Number of Leaves of LightGBM Classifier")
    plt.ylabel("Test Dataset Score")
    plt.xlabel("Number of Leaves in Tree")
    X = [2,4,6,8,10,12,14,16]
    Y =[]
    for i in X:
        model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=4,num_leaves=i,random_state=42)
        model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)]
                ,eval_metric='logloss',verbose = -1)
        Y.append(model.score(x_test,y_test))
    plt.scatter(X,Y,c="green")
    plt.show()    

#Num_Leaves(x_train,x_test,y_train,y_test)

def Min_Data_In_Leaf(x_train,x_test,y_train,y_test):
    plt.title("Optimising the Minimum Data in each Leaf of LightGBM Classifier")
    plt.ylabel("Test Dataset Score")
    plt.xlabel("Minimum Data in each Leaf in Tree")
    X = range(10,31,2)
    Y =[]
    for i in X:
        model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=4,num_leaves=8,min_data_in_leaf=i,random_state=42)
        model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)]
                ,eval_metric='logloss',verbose = -1)
        Y.append(model.score(x_test,y_test))
    plt.scatter(X,Y,c="green")
    plt.show()    

#Min_Data_In_Leaf(x_train,x_test,y_train,y_test)

def Min_Data_In_Leaf(x_train,x_test,y_train,y_test):
    plt.title("Optimising the Minimum Data in each Leaf of LightGBM Classifier")
    plt.ylabel("Test Dataset Score")
    plt.xlabel("Minimum Data in each Leaf in Tree")
    X = range(10,31,2)
    Y =[]
    for i in X:
        model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=4,
        num_leaves=8,min_data_in_leaf=20,,random_state=42)
        model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)]
                ,eval_metric='logloss',verbose = -1)
        Y.append(model.score(x_test,y_test))
    plt.scatter(X,Y,c="green")
    plt.show()    
