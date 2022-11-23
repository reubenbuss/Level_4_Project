import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Loading the Data : 
data = pd.read_csv("SVMtrain.csv")
data.drop(['Sex'],axis=1,inplace=True)
#print(data.info())
#print(data.head())


'''
print("x dataset:---")
print(x.head())
print("y dataset:---")
print(y.head())
'''

# To define the input and output feature
x = data.drop(['Embarked','PassengerId'],axis=1)
y = data.Embarked
# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=20,eval_metric='logloss')

print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))
lgb.plot_importance(model,max_num_features = 20)