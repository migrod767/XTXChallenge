import pandas as pd
import pickle
from datetime import datetime
import tensorflow_estimator as tfE

N = 2999999
N_Leaves = 50
modelFile = 'XTX_Save_Model//Des_Tree_' + '2.9mm' +'_NLeaves'+str(N_Leaves)+ '.sav'
splitSize = 0.999999
X = pd.DataFrame()
y = pd.DataFrame()
train_x = pd.DataFrame()
test_x = pd.DataFrame()
train_y = pd.DataFrame()
test_y = pd.DataFrame()

def load_data():
    global train_x, test_x, train_y, test_y, X, y, N, splitSize

    filepath = 'Data//SampleXTXData' + str(N) + '.csv'
    dataset = pd.read_csv(filepath)
    dataset.fillna(0, inplace=True)

    print("Data Loaded")
    print(datetime.now() - startTime)

    y = dataset.iloc[:, 60].astype(float)
    X = dataset.iloc[:, 0:60].astype(float)
    print(X.shape)
    print(y.shape)
    print("")

    train_x = X
    train_y = y

    test_x = train_x
    test_y = train_y

def regmodel(N_treestmp,):
    regresor = tfE.estimator.BoostedTreesRegressor(n_trees=N_treestmp)

