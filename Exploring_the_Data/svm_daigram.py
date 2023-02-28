import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
  

clean_df = pd.read_csv(
    r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\Mangrove_data_reduced_precision_3_best.csv")
labels = {"Black": "black", "White": "green",
          "Red": "red", "na": 'blue', 'Mud': 'brown'}
species_dictionary = {"Black": 0, "Red": 1, "White": 2}
  
X = clean_df[['677','1997']].to_numpy()
y = clean_df.Species
colour_labels = [labels[x] for x in y.to_numpy()]

def label_to_float(labels_df):
    '''
    Tranforms the str labels to flaot labels
    '''
    new_labels=[]
    for i in range(0, len(labels_df)):
        new_labels.append(species_dictionary[labels_df.iat[i]])
    return new_labels

y = label_to_float(y)
####
# model = SVC(kernel='linear',C=100)
# model.fit(X, y)
# print(model.support_vectors_)

# plt.scatter(X[:,0],X[:,1],c=colour_labels)
# plt.ylim(0.015,0.1)
# plt.xlim(0,0.08)


# plt.show()
####
# def make_meshgrid(x, y, h=.01):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy

# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out

# model = SVC(kernel='linear',C=10000)
# clf = model.fit(X, y)
# y_pred = model.predict(X)

# fig, ax = plt.subplots(dpi=240)
# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)

# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","red","green"])
# #cmap = matplotlib.colors.ListedColormap(["black","red","green"])
# my_cmap = cmap(np.arange(cmap.N))
# my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
# my_cmap = matplotlib.colors.ListedColormap(my_cmap)
# plot_contours(ax, clf, xx, yy, cmap=my_cmap, alpha=0.8)
# ax.scatter(X0, X1, c=colour_labels, s=20, edgecolors='black',linewidth=0.5)
# ax.set_ylabel('1997nm')
# ax.set_xlabel('677nm')
# # ax.set_xticks(())
# # ax.set_yticks(())
# #ax.legend()
# plt.xlim(0.015,0.1)
# plt.ylim(0,0.08)
# plt.show()

#####
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

z = X[:,0]*10


ax.scatter(X[:,0],X[:,1],z,c=colour_labels)
plt.xlim(0.015,0.1)
plt.ylim(0,0.08)
#ax.set_zlim(0,0.01)
ax.set_xlabel('667nm')
ax.set_ylabel('1997nm')
ax.set_zlabel('kernel')
plt.show()
