import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap

df = pd.read_csv(r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Coffee\Categorised_Coffee_Data.csv")

x = df.iloc[:,246:1178]
y=df.Species


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
          verbose=20,eval_metric='logloss',feature_name = list(x.columns),)

print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(x_test)
# sv = explainer(x_test)
# cmap = ListedColormap(["red","green","blue","brown"])


# shap.summary_plot(shap_values, x_test, class_names=model.classes_, color=cmap, show = False)               
# plt.title("SHAP Feature Importance Top 4 from 350nm to 750nm",fontsize = 20)
# plt.ylabel("Fetaures", fontsize = 16)
# plt.xlabel("mean(|SHAP value|)",fontsize = 16)
# plt.show()