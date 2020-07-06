# TUTORIAL URL: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Pandas is used for data manipulation
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import _hist_gradient_boosting
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import pickle
import os
import json
import math

N = 2999999
N_ref = N
cwd = os.getcwd()
modelFile = "XTX_Save_Model//RF_SK_N{0}.sav".format(N)
splitSize = 0.0000001


def load_data():
    global train_features, test_features, train_labels, test_labels, N, splitSize, N_ref
    # Read in data as a dataframe
    filepath = "Data//SampleXTXData{0}.csv".format(N)
    dataset = pd.read_csv(filepath)
    dataset.fillna(0, inplace=True)

    print("Data Loaded")
    print(datetime.now() - startTime)

    # Extract features and labels
    labels = dataset.iloc[:, 60].astype(float)
    features = dataset.iloc[:, 0:60].astype(float)
    print(labels.shape)
    print(features.shape)
    print("")
    del dataset

    # Convert to numpy arrays

    features = np.array(features)
    labels = np.array(labels)

    # Training and Testing Sets
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=splitSize,
                                                                                random_state=42)

    del features, labels

    print("Data Splited")
    print(datetime.now() - startTime)
    print("Train Data")
    print(train_features.shape)
    print(train_labels.shape)
    N_ref = train_labels.shape[0]
    print("Test Data")
    print(test_features.shape)
    print(test_labels.shape)
    print("")


def train_tune_model():
    # Random Forest Model
    rf = RandomForestRegressor(random_state=42)

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(rf.get_params())
    print("")

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=5)]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=8)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x * N_ref) for x in np.linspace(start=0.015, stop=0.035, num=7)]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   # 'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid,
                                   n_iter=100, cv=4, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(train_features, train_labels)

    temp_params = rf_random.best_params_
    pprint(temp_params)
    print_params(temp_params)
    print("N_ref is : {0}".format(N_ref))

    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors_num = (predictions - test_labels) ** 2
        errors_den = test_labels ** 2
        mape = np.sum(errors_num) / np.sum(errors_den)
        accuracy = 1 - mape
        print('Model Performance')
        print('Accuracy = {:0.2f}%.'.format(accuracy))

        return accuracy

    base_model = RandomForestRegressor(n_estimators=10, random_state=42)
    base_model.fit(train_features, train_labels)
    base_accuracy = evaluate(base_model, test_features, test_labels)
    print("Base Accuracy: {0}".format(base_accuracy))

    best_random = rf_random.best_estimator_
    # Saves model on .sav file
    pickle.dump(best_random, open(modelFile, 'wb'))

    random_accuracy = evaluate(best_random, test_features, test_labels)
    print("Random Accuracy: {0}".format(random_accuracy))

    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))


def train_rf():
    global train_features, train_labels

    # Number of trees in random forest
    n_estimators = 200
    # Number of features to consider at every split
    max_features = 8
    # Maximum number of levels in tree
    max_depth = 30
    # Minimum number of samples required at each leaf node
    min_samples_leaf = 75000

    # for e in n_estimators:
    modelPath = "XTX_Save_Model//RF_MD{2}_NE{0}_MF{1}_ML{3}K.sav".format(n_estimators, max_features,
                                                                         max_depth,
                                                                         int(min_samples_leaf / 1000))

    rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                               max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                               n_jobs=-1)

    rf.fit(train_features, train_labels)

    pickle.dump(rf, open(modelPath, 'wb'))

    print("Modelo END")
    print(datetime.now() - startTime)

    n_estimators = 300

    max_depth = 15

    # for e in min_samples_leaf:
    modelPath = "XTX_Save_Model//RF_NE{0}_ML{3}K_MF{1}_MD{2}.sav".format(n_estimators, max_features,
                                                                         max_depth,
                                                                         int(min_samples_leaf / 1000))

    rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                               max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                               n_jobs=-1)

    rf.fit(train_features, train_labels)

    pickle.dump(rf, open(modelPath, 'wb'))

    print("Modelo END")
    print(datetime.now() - startTime)


def train_HGB():
    # ensemble.HistGradientBoostingRegressor

    global train_features, train_labels

    # Number of trees in random forest
    n_estimatorsV = [50, 100, 200]
    # Number of features to consider at every split
    max_depthV = 10
    # Minimum number of samples required at each leaf node
    min_samples_leafV = 75000
    # loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}
    lossV = 'huber'
    # learning rate shrinks the contribution of each tree by learning_rate.
    learning_rateV = 0.025

    for n_estimatorsV in n_estimatorsV:
        modelPath = "XTX_Save_Model//HGB_NE{0}_MF{1}_LR{4}_LF{3}_ML{2}K.sav".format(n_estimatorsV,
                                                                                         max_depthV,
                                                                                         int(min_samples_leafV / 1000),
                                                                                         lossV, learning_rateV)

        rf = _hist_gradient_boosting(learning_rate=learning_rateV, n_estimators=n_estimatorsV,
                                     max_depth=max_depthV,
                                     min_samples_leaf=min_samples_leafV, verbose=1,
                                     loss=lossV)

        rf.fit(train_features, train_labels)

        pickle.dump(rf, open(modelPath, 'wb'))

        print("Model END")
        print(learning_rateV)
        print(datetime.now() - startTime)


def train_GBR():
    global train_features, train_labels

    # Number of trees in random forest
    n_estimatorsV = 100
    # Number of features to consider at every split
    max_featuresV = 45
    # Maximum number of levels in tree
    max_depthV = 20
    # Minimum number of samples required at each leaf node
    min_samples_leafV = [40000, 50000,100000,110000]
    # loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}
    lossV = 'ls'
    # learning rate shrinks the contribution of each tree by learning_rate.
    learning_rateV = 0.025


    for min_samples_leafV in min_samples_leafV:
        modelPath = "XTX_Save_Model//GB_MD{2}_NE{0}_LF{4}_MF{1}_ML{3}K_LR{5}.sav".format(n_estimatorsV,
                                                                                               max_featuresV,
                                                                                               max_depthV,
                                                                                               int(min_samples_leafV / 1000),
                                                                                               lossV, learning_rateV)

        rf = GradientBoostingRegressor(learning_rate=learning_rateV, n_estimators=n_estimatorsV,
                                       max_features=max_featuresV, max_depth=max_depthV,
                                       min_samples_leaf=min_samples_leafV, verbose=1,
                                       loss=lossV)

        rf.fit(train_features, train_labels)

        pickle.dump(rf, open(modelPath, 'wb'))

        print("Model END")
        print(learning_rateV)
        print(datetime.now() - startTime)


def print_params(params):
    global N, splitSize

    printText = json.dumps(params)

    filepath = open("XTX_DT_REG_Results//Params_tuning_RF" + str(N) + ".txt", "w+")
    filepath.write(printText)
    filepath.close()


def train_GBR_parallel():
    global train_features, train_labels
    import multiprocessing as mp

    def para_trainer(learning_rateV1, n_estimatorsV1, max_featuresV1, max_depthV1, min_samples_leafV1, lossV1):

        modelPath = "XTX_Save_Model//GB_L{4}_MD{2}_NE{0}_MF{1}_ML{3}K.sav".format(learning_rateV1, n_estimatorsV1,
                                                                                  max_featuresV1, max_depthV1,
                                                                                  int(min_samples_leafV1 / 1000),
                                                                                  lossV1)

        rf = GradientBoostingRegressor(learning_rate=learning_rateV1, n_estimators=n_estimatorsV1,
                                       max_features=max_featuresV1, max_depth=max_depthV1,
                                       min_samples_leaf=min_samples_leafV1, verbose=1,
                                       loss=lossV1)

        rf.fit(train_features, train_labels)

        pickle.dump(rf, open(modelPath, 'wb'))

        print("Model END")
        print(lossV)
        print(datetime.now() - startTime)

    pool = mp.Pool(4)

    # learning rate shrinks the contribution of each tree by learning_rate.
    learning_rateV = 0.1
    # Number of trees in random forest
    n_estimatorsV = 100
    # Number of features to consider at every split
    max_featuresV = 32
    # Maximum number of levels in tree
    max_depthV = 15  # TODO Peding RF results
    # Minimum number of samples required at each leaf node
    min_samples_leafV = 75000  # TODO Peding RF results
    # loss : {‘ls’, ‘lad’, ‘huber’, ‘quantile’}
    lossV = ['ls', 'lad', 'huber', 'quantile']

    for lossV2 in lossV:
        pool.apply_async(para_trainer,
                         args=(learning_rateV, n_estimatorsV, max_featuresV, max_depthV, min_samples_leafV, lossV2))

    lossV = 'ls'

    learning_rateV = [0.1, 0.05, 0.2]

    for learning_rateV2 in learning_rateV:
        pool.apply_async(para_trainer,
                         args=(learning_rateV2, n_estimatorsV, max_featuresV, max_depthV, min_samples_leafV, lossV))

    pool.close()


def predictions():
    global modelFile

    modelFile = "XTX_Save_Model//RF_NE100_MF8_MD15_ML75K.sav"
    load_Regresor = pickle.load(open(modelFile, 'rb'))

    N = 100000
    filepath = "Data//SampleXTXData{0}.csv".format(N)
    dataset = pd.read_csv(filepath)
    dataset.fillna(0, inplace=True)

    labels = dataset.iloc[:, 60].astype(float)
    features = dataset.iloc[:, 0:60].astype(float)

    features = np.array(features)
    labels = np.array(labels)

    print("Data Loaded")
    print(datetime.now() - startTime)
    print("")

    sumA = 0
    sumB = 0

    for i in range(0, N):
        if (i % int(N * 0.05)) == 0:
            print("Checking {0}".format(i))
            print(datetime.now() - startTime)
            print("")

        newX = features[i, :]
        realY = labels[i]

        newY = load_Regresor.predict([newX])
        newY = newY[0]

        sumA = sumA + (realY - newY) ** 2
        sumB = sumB + realY ** 2

    modelScore = 1 - (sumA / sumB)

    print("Model Score is: {0}".format(modelScore))
    print(datetime.now() - startTime)
    print("")



startTime = datetime.now()
print("Script Start")

load_data()

# train_GBR_parallel()



train_GBR()

#train_HGB()

# train_rf()

# train_tune_model()

# predictions()

print("End Script")
print(datetime.now() - startTime)
