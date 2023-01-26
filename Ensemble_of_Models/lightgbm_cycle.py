import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap
import pandas as pd

import lightgbm as lgb

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack Non Erroneous Data.csv")
species_labels=df.Species
reflectance_data=df.drop(["Species"],axis=1)

def classifier(features,labels,state=42):
    '''
    function that performs a lighgbm classification on selected dataset
    returning the accuracy on test data
    '''
    x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=state)
    model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10,random_state=state)
    model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
            eval_metric='logloss',feature_name = list(features.columns),
            callbacks = [lgb.log_evaluation(period=-1),lgb.early_stopping(stopping_rounds=10)])
    print(f'Training accuracy {model.score(x_train,y_train)}')
    print(f'Testing accuracy {model.score(x_test,y_test)}')
    return model.score(x_test,y_test)



classifier(df[["482","702","2147","2377"]],species_labels)

print("Finshed")
