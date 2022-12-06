from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap
from dtreeviz.trees import dtreeviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack.csv"
data = pd.read_csv(file)
data.drop(["SpectraID","WhiteReference","ContactProbe",
    "FibreOptic","SPAD_1","SPAD_2","SPAD_3","SPAD_Ave",
    "Location","Lat","Long","StandAge","StandHealth",
    "SurfaceDescription"],axis = 1,inplace=True)

y=data.Species
y = LabelEncoder().fit_transform(y)
global y
#all
#x=data.drop(["Species"],axis=1,inplace=False)
#first 40
#x=data[["350","356","713","351","1890","1888","1891","375","1893","2399","1899","355","1651","360","723","2241","1675","717","352","729","368","1998","1892","1831","353","1894","728","512","1882","1878","1123","2014","690","664","661","366","2132","1901","1886","1648"]]
#first100
#x=data[['350', '356', '351', '713', '1890', '1888', '1891', '375', '1893', '2399', '1899', '355', '1651', '360', '723', '717', '1675', '2241', '352', '368', '729', '1998', '1892', '353', '1831', '1894', '728', '512', '1123', '1882', '1878', '2014', '366', '690', '664', '661', '1648', '721', '737', '1886', '513', '354', '2132', '1901', '1619', '357', '378', '1879', '361', '367', '722', '369', '725', '727', '1975', '377', '718', '1108', '1652', '736', '930', '1889', '2304', '671', '1426', '1884', '2500', '2472', '2375', '1342', '2244', '659', '662', '1407', '673', '391', '649', '1900', '732', '2498', '1984', '747', '362', '726', '2022', '1877', '2003', '669', '672', '731', '2491', '390', '1986', '1001', '2469', '1887', '2035', '1658', '1895']]
x=data[["1890","1998","1658","1901"]]
global x 

def Sklearn():
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    model = DecisionTreeClassifier()
    model = model.fit(x_train,y_train)
    print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))
    return model

def LightGBM():
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
    model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
            verbose=20,eval_metric='logloss',feature_name = list(x.columns),)

    print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
    print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))
    return model



def shap_plot(model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    #shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test.iloc[0,:],matplotlib=True)
    cmap = ListedColormap(["black","red","green"])
    shap.summary_plot(shap_values, x, class_names=model.classes_, color=cmap, show = False)               
    plt.title("Shap Feature Importance",fontsize = 20)
    plt.ylabel("Fetaures", fontsize = 16)
    plt.xlabel("mean(|SHAP value|)",fontsize = 16)
    plt.show()
#shap_plot(model,x)


def dtree(model):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    viz = dtreeviz(model, 
               x_data=x_train,
               y_data=y_train,
               target_name='Species',
               feature_names=list(x.columns), 
               class_names=["Black","Red","White"], 
               title="Sklearn-DecisionTreeClassifier")
    viz.view()

def grid_search(model):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15),}
    # decision tree model
    dtree_model=DecisionTreeClassifier()
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(model, param_grid)
    #fit model to data
    dtree_gscv.fit(X, y)
    return dtree_gscv.best_params_

model = Sklearn()
dtree(model)

print("Finished")
