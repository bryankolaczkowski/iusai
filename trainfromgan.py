#!/usr/bin/env python3

import sys
import pandas
import numpy as np
import tensorflow as tf

## class label generator
class LabelGen(object):
  """
  generate random binary labels 0|1
  """
  def __init__(self):
    self.rng = np.random.default_rng()
    return

  def __call__(self):
    return self

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    lbl = self.rng.integers(1, dtype=np.int32, endpoint=True)
    return (lbl,lbl)

## set up training data generator
batchsize = 64
traindata = tf.data.Dataset.from_generator(LabelGen(),
                    output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32),
                                      tf.TensorSpec(shape=(), dtype=tf.int32)))
traindata = traindata.batch(batchsize)
print(list(traindata.take(4).as_numpy_iterator()))

## load generator model
gen_mdl = tf.keras.models.load_model('genr_20211106_061524', compile=False)
gen_mdl.trainable = False
gen_mdl.summary()

class StripLabels(tf.keras.layers.Layer):
  """
  removes labels from generator output
  """
  def __init__(self, *args, **kwargs):
    super(StripLabels, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    return inputs[0]

## build classifier
class_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

## connect classifier to generator
inputs = tf.keras.Input(shape=(1,))
x = gen_mdl(inputs)
x = StripLabels()(x)
outputs = class_layer(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

## compile
mets = [tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.AUC(curve='ROC', name='roc'),
        tf.keras.metrics.AUC(curve='PR',  name='prc')]
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=mets)

## fit
model.fit(traindata, epochs=1000, steps_per_epoch=100)


### evaluate on real data
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
valdata = tf.data.Dataset.from_tensor_slices((xnp,ynp)).batch(ynp.shape[0])
# evaluation model
inputs  = tf.keras.Input(shape=(xnp.shape[1],))
outputs = class_layer(inputs)
emodl = tf.keras.Model(inputs=inputs, outputs=outputs)
emodl.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=mets)
print('EVALUATING MODEL')
emodl.summary()
# evaluate
emodl.evaluate(valdata)
