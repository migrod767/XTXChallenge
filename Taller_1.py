import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from keras.utils import to_categorical

def load_data(N, splitSize):
    filepath = os.path.join("Data", "XTXData1000.0K.csv") #"Data\XTXData{}K.csv".format(str(N/1000))
    dataset = pd.read_csv(filepath, sep=",")
    dataset.fillna(0, inplace=True)

    y = dataset.iloc[:, 60].astype(float)
    X = dataset.iloc[:, 0:60].astype(float)
    print(X.shape)
    print(y.shape)
    print("")

    X = X.to_numpy()
    y = y.to_numpy()

    y_encoded = to_categorical(y)

    (train_x, test_x, train_y, test_y) = \
        train_test_split(X, y, train_size=splitSize, random_state=222)
    return train_x, test_x, train_y, test_y

def nn_model(train_x, test_x, train_y, test_y):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(60, activation='sigmoid'),
        tf.keras.layers.Dense(40, activation='sigmoid'),
        tf.keras.layers.Dense(20, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=5)

    # Saves model on .sav file
    #pickle.dump(regressor, open(modelFile, 'wb'))

    model.evaluate(test_x, test_y, verbose=2)

if __name__ == '__main__':
    N = 10000
    splitSize = 0.2
    train_x, test_x, train_y, test_y = load_data(N, splitSize)
    nn_model(train_x, test_x, train_y, test_y)