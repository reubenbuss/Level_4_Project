import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import shap

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
data = pd.read_csv(file)

# data.drop(["SpectraID","WhiteReference","ContactProbe",
#     "FibreOptic","SPAD_1","SPAD_2","SPAD_3","SPAD_Ave",
#     "Location","Lat","Long","StandAge","StandHealth",
#     "SurfaceDescription"],axis = 1,inplace=True)

y=data.Species

#all
#x=data.drop(["Species"],axis=1,inplace=False)
#first 40
#x=data[["350","356","713","351","1890","1888","1891","375","1893","2399","1899","355","1651","360","723","2241","1675","717","352","729","368","1998","1892","1831","353","1894","728","512","1882","1878","1123","2014","690","664","661","366","2132","1901","1886","1648"]]
#first100
#x=data[['350', '356', '351', '713', '1890', '1888', '1891', '375', '1893', '2399', '1899', '355', '1651', '360', '723', '717', '1675', '2241', '352', '368', '729', '1998', '1892', '353', '1831', '1894', '728', '512', '1123', '1882', '1878', '2014', '366', '690', '664', '661', '1648', '721', '737', '1886', '513', '354', '2132', '1901', '1619', '357', '378', '1879', '361', '367', '722', '369', '725', '727', '1975', '377', '718', '1108', '1652', '736', '930', '1889', '2304', '671', '1426', '1884', '2500', '2472', '2375', '1342', '2244', '659', '662', '1407', '673', '391', '649', '1900', '732', '2498', '1984', '747', '362', '726', '2022', '1877', '2003', '669', '672', '731', '2491', '390', '1986', '1001', '2469', '1887', '2035', '1658', '1895']]
#x=data[["1890","1998","1658","1901"]]
#x=data[["350","400","450","500","550","600","650","700","750","800"]]
x=data[list(map(str,list(range(350,751))))]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = lgb.LGBMClassifier(learning_rate=0.4,max_depth=5,num_leaves=10,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test)],
          verbose=0,eval_metric='logloss',feature_name = list(x.columns),)

print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)
#shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], x_test.iloc[0,:],matplotlib=True)

#print(model.classes_) = ['Black' 'Red' 'White']
#cmap = ListedColormap(["black","red","green"])

shap_values_black = shap_values[0]
shap_values_black = np.abs(shap_values_black).mean(0)
shap_values_black = list(enumerate(shap_values_black,350))
shap_values_black = [i for i in shap_values_black if i[1]>0.2]

shap_values_red = shap_values[1]
shap_values_red = np.abs(shap_values_red).mean(0)
shap_values_red = list(enumerate(shap_values_red,350))
shap_values_red = [i for i in shap_values_red if i[1]>0.2]

shap_values_white = shap_values[2]
shap_values_white = np.abs(shap_values_white).mean(0)
shap_values_white = list(enumerate(shap_values_white,350))
shap_values_white = [i for i in shap_values_white if i[1]>0.2]

fig = plt.figure(dpi = 2400)
plt.xlim(300,800)

#plt.ylim(0,1.1)
# plt.scatter(*zip(*shap_values_black), c="black",label = "Black Mangrove")
# plt.scatter(*zip(*shap_values_red), c="red", label = "Red mangrove")
# plt.scatter(*zip(*shap_values_white), c="green", label = "White Mangrove")

b_i,b_v = zip(*shap_values_black)
r_i,r_v = zip(*shap_values_red)
w_i,w_v = zip(*shap_values_white)
plt.hist([b_i,r_i,w_i],weights = [b_v,r_v,w_v],bins = 50,color = ["black","red","green"],stacked = True,label = ["Black Mangrove", "Red Mangrove","White Mangrove"])

plt.legend(loc="upper left")
plt.title("SHAP feature importance 350nm-750nm")
plt.xlabel("Features")
plt.ylabel("mean(|SHAP value|)")
plt.show()


print("Finished")
