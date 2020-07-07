import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_data(N, startTime):
    filepath = "Data//XTXData{}K.csv".format(str(N/1000))
    dataset = pd.read_csv(filepath)
    dataset.fillna(0, inplace=True)

    print("Data Loaded")
    print(datetime.now() - startTime)

    y = dataset.iloc[:, 60].astype(float)
    X = dataset.iloc[:, 0:60].astype(float)
    print(X.shape)
    print(y.shape)
    print("")

    print("Data Splited")
    print(datetime.now() - startTime)

    return X, y

def exploration(X, y):
    print("X Columns")
    print(X.columns)
    print('\n')

    print('Y Unique Values')
    y_list = y.unique()
    y_list.sort()
    print(y_list)
    print('\n')

    print(y.value_counts())
    print('\n')

    ts = X['bidSize0'].plot.hist(bins=100)
    ts.plot()
    plt.show()
    #TODO apply log transform to bidSize and askSize



if __name__ == '__main__':
    N = 1000000
    startTime = datetime.now()

    X, y = load_data(N, startTime)
    exploration(X, y)

