#!/usr/bin/env python3

import pandas
import numpy as np
import tensorflow as tf

# read data
dfnm = 'iusdata_raw.csv'
lblc = 'LABEL_NEC'
dtac = ['DATA_SBR',    'DATA_SBL',
        'DATA_COLONR', 'DATA_COLONL',
        'DATA_LUMENR', 'DATA_LUMENL']
# read initial data frame
dataframe = pandas.read_csv(dfnm)
ynp = dataframe[lblc].to_numpy(dtype=np.int32)
xnp = dataframe[dtac].to_numpy(dtype=np.float32)
print(xnp.shape, ynp.shape)
# tensorflow dataset package
batchsize = 8
data = tf.data.Dataset.from_tensor_slices((xnp,ynp))
data = data.shuffle(buffer_size=ynp.shape[0],
                    reshuffle_each_iteration=True).batch(batchsize)
print(data)

# build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,
                                activation='sigmoid',
                                input_shape=[xnp.shape[-1]]))
model.summary()

# compile
mets = [tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.AUC(curve='ROC', name='roc'),
        tf.keras.metrics.AUC(curve='PR',  name='prc')]
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=mets)

# fit
tot  = ynp.shape[0]
pos  = np.sum(ynp)
neg  = tot - pos
w0   = (1.0 / neg) * (tot / 2.0)
w1   = (1.0 / pos) * (tot / 2.0)
cwts = {0:w0, 1:w1}
model.fit(data, epochs=5000, class_weight=cwts)
