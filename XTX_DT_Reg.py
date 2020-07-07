from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split


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


def print_TXT(timespend1, modelscore1):
    global N, splitSize

    printText = "Desicion Tree Model SkLearn \n " \
                "Time spend: " + str(timespend1) + "\n " \
                   "Model Score: " + str(modelscore1) + "\n " \
                     "Split Size: " + str(splitSize) + "\n " \
                        "N: " + str(N) + "\n "

    filepath = open("XTX_DT_REG_Results//Score" + str(N) + ".txt", "w+")
    filepath.write(printText)
    filepath.close()


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
    #(train_x, test_x, train_y, test_y) = \
    #    train_test_split(X, y, train_size=splitSize, random_state=1)

    print("Data Splited")
    print(datetime.now() - startTime)
    print("Train Data")
    print(train_y.shape)
    print(train_x.shape)
    print("Test Data")
    print(test_y.shape)
    print(test_x.shape)
    print("")


def regmodel(N_leavestemp):
    global test_x, test_y, X, y
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state=0,
                                      min_samples_leaf=N_leavestemp)
    print("Model created")
    print(datetime.now() - startTime)
    print("")

    # fit the regressor with X and Y data

    regressor.fit(train_x, train_y)
    # regressor.fit(X, y)

    print("Trained")
    print(datetime.now() - startTime)
    print("")

    ModelScore = regressor.score(test_x, test_y)
    print("Accuracy: ", ModelScore)
    # print("Accuracy: ", regressor.score(X, y))
    timeSpend = datetime.now() - startTime

    #Saves model on .sav file
    pickle.dump(regressor, open(modelFile, 'wb'))

    print_TXT(timeSpend, ModelScore)

    print("")

def numpy_predictor():
    global modelFile
    load_Regresor = pickle.load(open(modelFile, 'rb'))

    N = 5000
    filepath = "Data//XTXData{}K.csv".format(str(N/1000))
    dataset = pd.read_csv(filepath)
    dataset.fillna(0, inplace=True)

    y = dataset.iloc[:, 60].astype(float)
    X = dataset.iloc[:, 0:60].astype(float)

    X_Np = X.to_numpy()
    #X_Np = X.as_matrix()

    print("Data Loaded")
    print(datetime.now() - startTime)
    print("")

    sumA = 0
    sumB = 0

    for i in range(0, N):
        if (i % 100000) == 0:
            print("Checking " + str(i))
            print(datetime.now() - startTime)
            print("")

        newX = [X_Np[i, :]]
        realY = y.iloc[i]

        newY = load_Regresor.predict(newX)
        newY = newY[0]

        sumA = sumA + (realY - newY) ** 2
        sumB = sumB + realY ** 2

    modelScore = 1 - (sumA / sumB)

    print("Model Score \n is: " + str(modelScore))
    print(datetime.now() - startTime)
    print("")

    timeSpend = datetime.now() - startTime
    print_TXT(timeSpend, modelScore)


def main_predictor():
    global modelFile
    load_Regresor = pickle.load(open(modelFile, 'rb'))

    N = 1000000
    filepath = "Data//XTXData{}K.csv".format(str(N/1000))
    dataset = pd.read_csv(filepath)
    dataset.fillna(0, inplace=True)

    y = dataset.iloc[:, 60].astype(float)
    X = dataset.iloc[:, 0:60].astype(float)

    print("Data Loaded")
    print(datetime.now() - startTime)
    print("")

    sumA = 0
    sumB = 0

    for i in range(0, N):
        if (i % 100000) == 0:
            print("Checking "+str(i))
            print(datetime.now() - startTime)
            print("")

        newX = [X.iloc[i, :]]
        realY = y.iloc[i]

        newY = load_Regresor.predict(newX)
        newY = newY[0]

        sumA = sumA + (realY - newY) ** 2
        sumB = sumB + realY ** 2

    modelScore = 1 - (sumA / sumB)

    print("Model Score \n is: " + str(modelScore))
    print(datetime.now() - startTime)
    print("")

    timeSpend = datetime.now() - startTime
    print_TXT(timeSpend, modelScore)

def plot_tree():
    from sklearn.tree import export_graphviz
    load_regresor = pickle.load(open(modelFile, 'rb'))
    print("Model loaded")
    print(datetime.now() - startTime)
    print("")

    export_graphviz(load_regresor, out_file='tree_XTX_Model.dot')
    print("Model plotted")
    print(datetime.now() - startTime)
    print("")

def main_train():
    global modelFile
    load_data()
    N_Leaves = 75000
    print("Model 1")
    modelFile = 'XTX_Save_Model//Des_Tree_'  + '_N' + str(N_Leaves) + '.sav'
    regmodel(N_Leaves)

    N_Leaves = 25000
    print("Model 1")
    modelFile = 'XTX_Save_Model//Des_Tree_' + '_N' + str(N_Leaves) + '.sav'
    regmodel(N_Leaves)

    N_Leaves = 50000
    N_Sample = 50000
    print("Model 3")
    modelFile = 'XTX_Save_Model//Des_Tree_' + '_S' + str(N_Sample) + '_N' + str(N_Leaves) + '.sav'
    #regmodel(N_Leaves, N_Sample)

    #numpy_predictor()


startTime = datetime.now()
print("Script Start")

main_train()
#plot_tree()
#main_predictor()
#numpy_predictor()

print(datetime.now() - startTime)
