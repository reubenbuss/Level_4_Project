import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
df = pd.read_csv(file)

y=df.Species
lb = LabelEncoder()
y = lb.fit_transform(y)

x=df.drop(["Species"],axis=1,inplace=False)
#x=df[["1892","671","1652","720"]]
#x=df[list(map(str,list(range(350,2501,50))))]

xs = StandardScaler().fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(xs,y,test_size=0.2,random_state=42)

X_trains_df=pd.DataFrame(x_train,columns=x.columns)
from sklearn.feature_selection import RFE
svc_lin=SVC(kernel='linear')
svm_rfe_model=RFE(estimator=svc_lin,n_features_to_select=50)
svm_rfe_model_fit=svm_rfe_model.fit(X_trains_df,y_train)
feat_index = pd.Series(data = svm_rfe_model_fit.ranking_, index = x.columns)
signi_feat_rfe = feat_index[feat_index==1].index
print('Significant features from RFE',signi_feat_rfe)

# def f_importances(coef, names):
#     d = zip(*sorted(zip(coef,names)))
#     print(d)
#     #d = pd.DataFrame(d).nlargest(10, 1)

#     #plt.barh(range(len(d[1])), d[0], align='center')
#     #plt.yticks(range(len(d[1])), d[1])
#     #plt.show()
# features_names = x.columns
# svm = svm.SVC(kernel='linear')
# svm.fit(x_train, y_train)
# f_importances(svm.coef_[0], features_names)