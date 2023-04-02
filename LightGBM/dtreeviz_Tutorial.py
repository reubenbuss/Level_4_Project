from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_boston
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from dtreeviz.trees import *

rp_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_5_best_outliers_removed.csv")

y=rp_df.Species
lb = LabelEncoder()
y = lb.fit_transform(y)

print(list(rp_df.columns))

x=rp_df[['1997','2442','677','1637']]
x = rp_df.drop(['Species'],axis=1,inplace=False)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# fit the classifier
clf = tree.DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)

viz = dtreeviz(clf, 
               x_data=X_train,
               y_data=y_train,
               target_name='Mangrove Species',
               feature_names=x.columns, 
               class_names=list(["Black Mangrove","Red Mangrove","White Mangrove"]), 
               colors = {"classes":[None,None,None,["Black","Red","Green"]]})
viz
