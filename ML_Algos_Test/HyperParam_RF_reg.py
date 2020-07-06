#TUTORIAL URL: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Pandas is used for data manipulation
import pandas as pd
import numpy as np

def load_data():
    global train_features, test_features, train_labels, test_labels
    # Read in data as a dataframe
    features = pd.read_csv('temps_extended.csv')

    # One Hot Encoding
    features = pd.get_dummies(features)

    # Extract features and labels
    labels = features['actual']
    features = features.drop('actual', axis = 1)

    # Names of six features accounting for 95% of total importance
    important_feature_names = ['temp_1', 'average', 'ws_1', 'temp_2', 'friend', 'year']

    # Update feature list for visualizations
    feature_list = important_feature_names[:]

    features = features[important_feature_names]
    features.head(5)

    # Convert to numpy arrays

    features = np.array(features)
    labels = np.array(labels)

    # Training and Testing Sets
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size = 0.25,
                                                                            random_state = 42)


load_data()

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=1,
                               random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_labels)

print(rf_random.best_params_)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


base_model = RandomForestRegressor(n_estimators=10, random_state=42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)
print("Base Accuracy: "+str(base_accuracy))

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)
print("Random Accuracy: "+str(random_accuracy))

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))