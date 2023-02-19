import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
df = pd.read_csv(file)

y=df.Species
lb = LabelEncoder()
y_fit = lb.fit_transform(y)
print(y_fit)
x=df.drop(["Species"],axis=1,inplace=False)
#x=df[list(map(str,list(range(350,401))))]


#SVM 
xs = StandardScaler().fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(xs,y_fit,test_size=0.2,random_state=42)
X_trains_df=pd.DataFrame(x_train,columns=x.columns)
svc_lin=SVC(kernel='linear')
svm_rfe_model=RFE(estimator=svc_lin,n_features_to_select=100)
svm_rfe_model_fit=svm_rfe_model.fit(X_trains_df,y_train)
feat_index = pd.Series(data = svm_rfe_model_fit.ranking_, index = x.columns)
signi_feat_rfe = feat_index[feat_index==1].index
#print(signi_feat_rfe)

#Correlation
def remove_correllated_features(x,y,threshold):
    to_drop=[]
    cor_matrix = x.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    length = len(x.columns)
    for i in range(1,length):
        for j in range(0,i):
            if upper_tri.iloc[j,i] > threshold: #row then column
                df = pd.DataFrame(list(zip(x.iloc[:,i],x.iloc[:,j],y)),columns = [x.columns[i],x.columns[j],"y"])
                new_cor_matrix = df.corr().abs()
                print(new_cor_matrix)
                print(df)
                if new_cor_matrix.iloc[0,2] > new_cor_matrix.iloc[1,2]:
                    to_drop.append(x.columns[j])
                else: 
                    to_drop.append(x.columns[i])
    x_new = x.drop(to_drop, axis=1,inplace=False)
    return x_new

#x=df[signi_feat_rfe]
x=remove_correllated_features(df[["350","450"]],y_fit,0.90)
print(x.columns)

#LightGBM
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
          callbacks = [lgb.log_evaluation(period=1),lgb.early_stopping(stopping_rounds=10)],
          eval_metric='logloss',feature_name = list(x.columns),)

print('Training accuracy {0:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {0:.4f}'.format(model.score(x_test,y_test)))


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)
sv = explainer(x_test)
#shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test.iloc[0,:],matplotlib=True)
cmap = ListedColormap(["red","green","black"])


shap.summary_plot(shap_values, x_test, class_names=model.classes_, color=cmap, show = False)               
plt.title("SHAP Feature Importance Top 4 from 350nm to 750nm",fontsize = 20)
plt.ylabel("Fetaures", fontsize = 16)
plt.xlabel("mean(|SHAP value|)",fontsize = 16)
plt.show()

print("Finsihed")
