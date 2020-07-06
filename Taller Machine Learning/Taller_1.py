from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

df = pd.read_csv(csv_file)

df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

tf.constant(df['thal'])

train_dataset = dataset.shuffle(len(df)).batch(2)
print(train_dataset)