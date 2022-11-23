import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\RedWhiteBlack.csv"
data = pd.read_csv(file)
data.drop(["SpectraID","WhiteReference","ContactProbe",
    "FibreOptic","SPAD_1","SPAD_2","SPAD_3","SPAD_Ave",
    "Location","Lat","Long","StandAge","StandHealth",
    "SurfaceDescription"],axis = 1,inplace=True)

y=data.Species

#all
#x=data.drop(["Species"],axis=1,inplace=False)
#first 40
#x=data[["350","356","713","351","1890","1888","1891","375","1893","2399","1899","355","1651","360","723","2241","1675","717","352","729","368","1998","1892","1831","353","1894","728","512","1882","1878","1123","2014","690","664","661","366","2132","1901","1886","1648"]]
#first100
x=data[['350', '356', '351', '713', '1890', '1888', '1891', '375', '1893', '2399', '1899', '355', '1651', '360', '723', '717', '1675', '2241', '352', '368', '729', '1998', '1892', '353', '1831', '1894', '728', '512', '1123', '1882', '1878', '2014', '366', '690', '664', '661', '1648', '721', '737', '1886', '513', '354', '2132', '1901', '1619', '357', '378', '1879', '361', '367', '722', '369', '725', '727', '1975', '377', '718', '1108', '1652', '736', '930', '1889', '2304', '671', '1426', '1884', '2500', '2472', '2375', '1342', '2244', '659', '662', '1407', '673', '391', '649', '1900', '732', '2498', '1984', '747', '362', '726', '2022', '1877', '2003', '669', '672', '731', '2491', '390', '1986', '1001', '2469', '1887', '2035', '1658', '1895']]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=20,eval_metric='logloss')

print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))
lgb.plot_importance(model,max_num_features = 20)
features = x_train.columns
importances = model.feature_importances_
feature_importance = pd.DataFrame({'features':features,'importance':importances}).sort_values('importance', ascending=False).reset_index(drop=True)

#feature_importance
#get list of top features
#x = feature_importance.iloc[0:99,0]
#print(list(x))

'''
#scatter
features = features.astype(np.float)
plt.scatter(features,importances)
plt.xlim(300,2500)
plt.show()
'''

