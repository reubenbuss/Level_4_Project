import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap
import pandas as pd

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")

y=df.Species
x=df.drop(["Species"],axis=1)

def algorithm(x,y,top,secs):
    '''
    function that performs a lighgbm classification on the whole dataset
    then shap selects the top important features 
    running lightgbm classification on the top features
    removing the top features 
    then repeating the test 
    returning all the top features, order of selection and their accuracys
    '''
    wavelengths = []
    order = []
    scores = []
    TOP = top
    for i in range(0,secs):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
        model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10)
        model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
                eval_metric='logloss',feature_name = list(x.columns),
                callbacks = [lgb.log_evaluation(period=-1),lgb.early_stopping(stopping_rounds=10)])
        print(f'Training accuracy {model.score(x_train,y_train)}')
        print(f'Testing accuracy {model.score(x_test,y_test)}')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        vals= np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(x.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        top_10 = feature_importance.iloc[0:TOP,0]
        top_10 = list(top_10)
        x_new = x[top_10]
        x_train,x_test,y_train,y_test = train_test_split(x_new,y,test_size=0.2)
        model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10)
        model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
                eval_metric='logloss',feature_name = list(x_new.columns),
                callbacks = [lgb.log_evaluation(period=-1),lgb.early_stopping(stopping_rounds=10)])
        score = model.score(x_test,y_test)
        print(f'Training accuracy {model.score(x_train,y_train)}')
        print(f'Testing accuracy {score}')

        x = x.drop(top_10,axis=1,inplace=False)
        wavelengths += top_10
        order += [1-i*0.01]*TOP
        scores += [score]*TOP

    wavelengths = list(map(int, wavelengths))
    plt.scatter(wavelengths,scores,cmap='viridis',c=order)
    plt.colorbar()
    plt.show()

    results = pd.DataFrame({"Wavelengths":wavelengths,"Scores":scores,"Order":order})
    max1 = results.loc[results["Scores"] == results["Scores"].max()]
    print(max1)
    return list(max1["Wavelengths"])

new_wavelengths = list(map(str,[1992, 1649, 1997, 1912, 2016, 1973, 2034, 1662, 2461, 2020, 1991, 476, 2246, 1671, 725, 719, 2181, 1683, 419, 391, 2357, 2065, 2354, 429, 1458, 1707, 430, 639, 2139, 434, 1736, 1712, 2258, 405, 421, 427, 1581, 696, 2154, 1586]))
x = df[new_wavelengths]
print(algorithm(x,y,4,10))
print("Finished")
