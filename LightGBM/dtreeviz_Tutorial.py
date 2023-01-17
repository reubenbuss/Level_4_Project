from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_boston
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from dtreeviz.trees import *

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
data = pd.read_csv(file)

y=data.Species
lb = LabelEncoder()
y = lb.fit_transform(y)

#x=data.drop(["Species"],axis=1,inplace=False)
x=data[["750","680","351","390"]]
x=data[["1892","671","1652","720"]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# fit the classifier
clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

viz = dtreeviz(clf, 
               x_data=X_train,
               y_data=y_train,
               target_name='Mangrove Species',
               feature_names=x.columns, 
               class_names=list(["Black Mangrove","Red Mangrove","White Mangrove"]), 
               title="Decision Tree",
               colors = {"classes":[None,None,None,["Black","Red","Green"]]})
viz
