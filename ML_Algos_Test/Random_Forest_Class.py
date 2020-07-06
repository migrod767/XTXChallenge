import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

np.random.seed(999)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df['is_train'] = np.random.uniform(0,1,len(df))<=0.65

train, test = df[df['is_train']==True],df[df['is_train']==False]
print('Number of observations in train',len(train))
print('Number of observations in test',len(test))

features = df.columns[:4]
num_species = pd.factorize(train['species'])[0]

classifier = RandomForestClassifier(n_jobs=2, random_state=0)
classifier.fit(train[features],num_species)
classifier.predict(test[features])

print(classifier.predict_proba(test[features])[0:10])
test_x = test[features]
test_y = test['species']

print(test_x)
print(test_y)

print("Accuracy: ", classifier.score(test[features], test['species']))


names = iris.target_names[classifier.predict(test[features])]

print(names[0:5])
print(test['species'].head())