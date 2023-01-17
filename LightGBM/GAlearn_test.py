import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc
from galearn import *
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt 


path_submissions = '/'

target_name = 'target'
scores_folds = {}

def rmspe(y_true, y_pred):
    return -np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

file = r"C:\Users\reube\OneDrive - Durham University\Documents\Year 4\Project\Data\BlackRedWhite Data Cleaned.csv"
df = pd.read_csv(file)

y=df.Species
x=df.drop(["Species"],axis=1,inplace=False)

params = dict()
params['boosting_type'] = ['gbdt', 'dart']
params['n_jobs'] = [-1]
params['num_leaves'] = np.arange(50,300)
params['learning_rate'] = np.linspace(0.0001, 0.5, 1000)
params['colsample_bytree'] = np.linspace(0.5, 1, 1000)
params['subsample'] = np.linspace(0.25, 0.9, 1000)
params['n_estimators'] = np.arange(70, 300)
params['min_child_weight'] = np.linspace(0.00001, 50, 1000)
params['min_child_samples'] = np.arange(3, 50)
params['reg_alpha'] = np.linspace(0.001, 1, 1000)
params['reg_lambda'] = np.linspace(0.0001, 0.5, 1000)
params['random_state'] = [42]

scorer = make_scorer(rmspe)

features = [col for col in x.columns if col not in {"time_id", "target", "row_id"}]

best = simulate(params, scorer, 10,
                        lgb.LGBMClassifier,
                        x, y,
                        selection = 'tournament',
                        p_cross = 0.8,
                        p_mutate = 0.5,
                        sim_ann = True)

print(best)