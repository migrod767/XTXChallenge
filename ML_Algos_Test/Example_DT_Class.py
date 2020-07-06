import pandas as pd
import numpy as np
import matplotlib as plt
from datetime import datetime
startTime = datetime.now()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('Data//IrisDataSet.csv', sep=',')
#df = pd.read_csv('Data//SampleXTXData.csv', sep=',')

print(df.describe())
#print(df.dtypes)

df.fillna(0, inplace=True)

all_inputs = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
all_classes = df['species'].values

#all_inputs = df[['askRate0','askRate1','askRate2','askRate3','askRate4','askRate5',\
#                  'askRate6','askRate7','askRate8','askRate9','askRate10','askRate11','askRate12',\
#                  'askRate13','askRate14','askSize0','askSize1','askSize2','askSize3','askSize4',\
#                  'askSize5','askSize6','askSize7','askSize8','askSize9','askSize10','askSize11',\
#                  'askSize12','askSize13','askSize14','bidRate0','bidRate1','bidRate2','bidRate3',\
#                  'bidRate4','bidRate5','bidRate6','bidRate7','bidRate8','bidRate9','bidRate10',\
#                  'bidRate11','bidRate12','bidRate13','bidRate14','bidSize0','bidSize1','bidSize2',\
#                  'bidSize3','bidSize4','bidSize5','bidSize6','bidSize7','bidSize8','bidSize9','bidSize10',\
#                 'bidSize11','bidSize12','bidSize13','bidSize14']].values
#all_classes = df['y'].values


(train_inputs, test_inputs, train_classes, test_classes) =\
    train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

estimator = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=3)
estimator.fit(train_inputs, train_classes)

print("Accuracy: ", estimator.score(test_inputs, test_classes))
print("--------------------")
treeStructure()
print("--------------------")
print(datetime.now() - startTime)

