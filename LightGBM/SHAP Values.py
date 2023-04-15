import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap
import graphviz

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
data = pd.read_csv(file)
data = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv")


# data.drop(["SpectraID","WhiteReference","ContactProbe",
#     "FibreOptic","SPAD_1","SPAD_2","SPAD_3","SPAD_Ave",
#     "Location","Lat","Long","StandAge","StandHealth",
#     "SurfaceDescription"],axis = 1,inplace=True)

y=data.Species

#all
#x=data.drop(["Species"],axis=1,inplace=False)
#389,521,569,761
x=data[["389","521",'569',"761"]]
#x=data[["1441","680","725","366"]]
#x=data[["750","680","351","390","716","511"]]
#x=data[list(map(str,list(range(350,751))))]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10,random_state=0)
model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
          verbose=20,eval_metric='logloss',feature_name = list(x.columns),)

print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)
sv = explainer(x_test)
#shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test.iloc[0,:],matplotlib=True)
cmap = ListedColormap(["red","green","black"])
vals= np.abs(shap_values).mean(0)

feature_importance = pd.DataFrame(list(zip(x.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
feature_importance.head()
shap.summary_plot(shap_values, x_test, class_names=model.classes_, color=cmap, show = False)               
#plt.title("SHAP Feature Importance Top 4 from 350nm to 750nm",fontsize = 20)
plt.ylabel("Fetaures", fontsize = 16)
plt.xlabel("mean(|SHAP value|)",fontsize = 16)
plt.show()


# exp = shap.Explanation(sv.values[:,:,1], 
#                   sv.base_values[:,1], 
#                   data=x.values, 
#                   feature_names=x.columns)
# idx = 2
# shap.waterfall_plot(exp[idx],show=False)
# plt.title("SHAP Waterfall of Random Sample",fontsize = 20)
# #plt.ylabel("Fetaures", fontsize = 16,rotation = 90 ,loc="center")
# #plt.xlabel("Feature Influence",fontsize = 16,loc = "left")
# plt.show()

print("Finished")
