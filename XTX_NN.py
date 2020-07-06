import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sys import getsizeof

N = 5000
modelFile = "XTX_Save_Model//RF_SK_N{0}.sav".format(N)
splitSize = 0.3
startTime = datetime.now()

def load_data():
    global N, splitSize, startTime, train_dataset, features, labels
    # Read in data as a dataframe
    filepath = "Data//SampleXTXData{0}.csv".format(N)
    dataset = pd.read_csv(filepath)
    dataset.fillna(0, inplace=True)
    print(dataset.head())

    print("Data Loaded")
    print(datetime.now() - startTime)

    # Extract features and labels
    column_names = list(dataset.columns)
    label_name = column_names[-1]
    column_names.pop()

    training_set, test_set = train_test_split(dataset, test_size=splitSize, random_state=42)

    traning_y = training_set.iloc[:,-1]
    training_set = training_set.drop(labels=label_name, axis=1)


    # Create Batches
    batch_size = 100

    train_dataset = tf.data.Dataset.from_tensor_slices((training_set.values, traning_y.values))
    train_dataset = train_dataset.batch(batch_size)


    features, labels = next(iter(train_dataset))


    print("Data Splited")
    print(datetime.now() - startTime)
    print("features", "\n")
    print(features, "\n")
    print("labels", "\n")
    print(labels, "\n")


def TF_NN():
    global train_dataset, features, labels

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation=tf.nn.relu, input_shape=(60,)),  # input shape required
        tf.keras.layers.Dense(1)
    ])

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loss(model, x, y):
        y_ = model(x)

        return loss_object(y_true=y, y_pred=y_)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)  # TODO cambiar el SGD

    loss_value, grads = grad(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                              loss(model, features, labels).numpy()))

    ## Note: Rerunning this cell uses the same model variables

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 200

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            """ Track progress
            """
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(y, model(x))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()

if __name__ == '__main__':
    load_data()
    TF_NN()