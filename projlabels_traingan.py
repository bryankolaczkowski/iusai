#!/usr/bin/env python3

import sys
import time
import distutils.util
import argparse

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn.preprocessing
import sklearn.decomposition

import pandas
import numpy as np

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import Model

### DATA #######################################################################

def getRealData(datafile, batchsize):
  lblc = 'LABEL_NEC'
  dtac = ['DATA_SBR',    'DATA_SBL',
          'DATA_COLONR', 'DATA_COLONL',
          'DATA_LUMENR', 'DATA_LUMENL']
  # read initial data frame
  dataframe = pandas.read_csv(datafile)
  ynp = dataframe[lblc].to_numpy(dtype=np.int32)
  xnp = dataframe[dtac].to_numpy(dtype=np.float32)
  print(xnp.shape, ynp.shape)
  # tensorflow dataset package
  data = tf.data.Dataset.from_tensor_slices((xnp,ynp))
  data = data.shuffle(buffer_size=ynp.shape[0],
                      reshuffle_each_iteration=True).batch(batchsize)
  print(data)
  return (xnp.shape[1], xnp.shape[0], xnp, ynp, data)

### PLOTTING DATA ##############################################################

class PlotCallback(tf.keras.callbacks.Callback):
  """
  plot generated data
  """
  def __init__(self, data, labels, log_dir='logs'):
    self.writer = tf.summary.create_file_writer(log_dir + '/gen')
    self.data   = data
    self.labels = labels
    # create data normalizer
    self.norm = sklearn.preprocessing.StandardScaler()
    self.norm.fit(self.data)
    # create PCA projection
    self.pca = sklearn.decomposition.PCA(n_components=2)
    self.pca.fit(self.norm.transform(self.data))
    # plot housekeeping
    self.colors = ['blue', 'red']
    return

  def plot_data(self, data):
    fig = plt.figure(figsize=(6,6))
    real_data = self.pca.transform(self.norm.transform(self.data))
    for color,label in zip(self.colors, [0,1]):
      plt.scatter(real_data[self.labels==label, 0],
                  real_data[self.labels==label, 1],
                  color=color,
                  marker='o',
                  alpha=0.8)
    fake_data = self.pca.transform(self.norm.transform(data))
    for color,label in zip(self.colors, [0,1]):
      plt.scatter(fake_data[self.labels==label, 0],
                  fake_data[self.labels==label, 1],
                  color=color,
                  marker='D',
                  alpha=0.4)
    n = sklearn.preprocessing.StandardScaler()
    n.fit(data)
    p = sklearn.decomposition.PCA(n_components=2)
    p.fit(data)
    fake_data = p.transform(n.transform(data))
    for color,label in zip(self.colors, [0,1]):
      plt.scatter(fake_data[self.labels==label, 0],
                  fake_data[self.labels==label, 1],
                  color=color,
                  marker='s',
                  alpha=0.1)
    return fig

  def plot_to_image(self, plot):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(plot)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

  def on_epoch_end(self, epoch, logs=None):
    # generate example datas
    ((dta,lbl),scr) = self.model(self.labels)
    fig = self.plot_data(dta)
    img = self.plot_to_image(fig)
    with self.writer.as_default():
      tf.summary.image('GenData', img, step=epoch)
    return

### GAN ########################################################################

@tf.function(experimental_relax_shapes=True)
def lrsact(x, alpha=0.4):
  """
  2-sided 'leaky-rectified' linear activation
  scales x by alpha*x whenever |x| > (1-alpha)
  """
  v  = 1.0 - alpha
  b  = v * v
  # leaky-rectify positive values
  c = tf.math.greater(x, v)
  r = tf.where(c, alpha*x+b, x)
  # leaky-rectify negative values
  c = tf.math.less(r, -v)
  r = tf.where(c, alpha*r-b, r)
  return r

class WassersteinLoss(Loss):
  """
  implements wasserstein loss function

  'earth mover' distance from:
    https://arxiv.org/pdf/1701.07875.pdf
    https://arxiv.org/pdf/1704.00028.pdf
  """
  def __init__(self):
    super(WassersteinLoss, self).__init__(name='wasserstein_loss')
    return

  def call(self, y_true, y_pred):
    return tf.math.reduce_mean(y_true * y_pred)

class GanOptimizer(Optimizer):
  """
  implements a generator,discriminator optimizer pair
  """
  def __init__(self,
               gen_optimizer='sgd',
               dis_optimizer='sgd',
               **kwargs):
    super(GanOptimizer, self).__init__(name='GanOptimizer', **kwargs)
    self.gen_optimizer = tf.keras.optimizers.get(gen_optimizer)
    self.dis_optimizer = tf.keras.optimizers.get(dis_optimizer)
    return

  def apply_gradients(self, grads_and_vars,
                      name=None, experimental_aggregate_gradients=True):
    raise NotImplementedError('GAN optimizer should call '
                              'apply_generator_gradients and '
                              'apply_discriminator_gradients instead')

  def apply_generator_gradients(self, grads_and_vars):
    return self.gen_optimizer.apply_gradients(grads_and_vars)

  def apply_discriminator_gradients(self, grads_and_vars):
    return self.dis_optimizer.apply_gradients(grads_and_vars)

  def get_config(self):
    config = super(GanOptimizer, self).get_config()
    config.update({
      'gen_optimizer' : tf.keras.optimizers.serialize(self.gen_optimizer),
      'dis_optimizer' : tf.keras.optimizers.serialize(self.dis_optimizer),
    })
    return config


class GAN(Model):
  """
  generative adversarial network
  """
  def __init__(self,
               generator,
               discriminator,
               augmentation,
               augmentprob,
               **kwargs):
    super(GAN, self).__init__(**kwargs)
    self.genr = generator
    self.disr = discriminator
    self.augm = augmentation
    self.augp = augmentprob
    return

  def compile(self,
              optimizer=GanOptimizer(),
              loss=WassersteinLoss(),
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              **kwargs):
    super(GAN, self).compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics,
                             loss_weights=loss_weights,
                             weighted_metrics=weighted_metrics,
                             run_eagerly=run_eagerly,
                             steps_per_execution=steps_per_execution,
                             **kwargs)
    return

  def call(self, inputs, training=None):
    """
    inputs should be 0|1 labels
    """
    gdta1,_ = self.genr(inputs, training=training)  # generate first  data
    gdta2,_ = self.genr(inputs, training=training)  # generate second data
    dsrsr = self.disr((gdta1, gdta2, inputs), training=training)
    return ((gdta1,inputs), dsrsr)

  def _augment(self, data):
    n = tf.random.normal(shape=tf.shape(data), stddev=self.augm)
    a = tf.random.stateless_binomial(shape=tf.shape(data),
                                     seed=tf.random.uniform(shape=(2,), minval=1, maxval=10000, dtype=tf.int32),
                                     counts=tf.ones(shape=tf.shape(data)[-1]),
                                     probs=[self.augp],
                                     output_dtype=tf.float32)
    return data + (n*a)

  def _calc_loss(self, qry_data, gnr_data, lbls, y, training=None):
    """
    calculates appropriate loss function
    """
    y_hat = self.disr((qry_data, gnr_data, lbls), training=training)
    return self.compiled_loss(y, y_hat)

  def _get_step_setup(self, inputs):
    """
    returns positive and negative labels
    """
    bs    = tf.shape(inputs)[0]  # batch size
    pones =  tf.ones((bs,1))     # positive labels
    nones = -tf.ones((bs,1))     # negative labels
    return pones, nones

  def train_step(self, inputs):
    """
    single training step; inputs are (data,labels)
    """
    data = inputs[0]
    lbls = inputs[1]

    pones, nones = self._get_step_setup(lbls)

    # train discriminator using real data
    with tf.GradientTape() as tape:
      disr_rl = self._calc_loss( \
                    qry_data=self._augment(data),
                    gnr_data=self._augment(self.genr(lbls, training=False)[0]),
                    lbls=lbls,
                    y=nones,
                    training=True)
    grds = tape.gradient(disr_rl, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train discriminator using fake data
    with tf.GradientTape() as tape:
      disr_fk = self._calc_loss( \
                    qry_data=self._augment(self.genr(lbls, training=False)[0]),
                    gnr_data=self._augment(self.genr(lbls, training=False)[0]),
                    lbls=lbls,
                    y=pones,
                    training=True)
    grds = tape.gradient(disr_fk, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train generator
    with tf.GradientTape() as tape:
      genr_ls = self._calc_loss( \
                    qry_data=self._augment(self.genr(lbls, training=True )[0]),
                    gnr_data=self._augment(self.genr(lbls, training=False)[0]),
                    lbls=lbls,
                    y=nones,
                    training=False)
    grds = tape.gradient(genr_ls, self.genr.trainable_weights)
    self.optimizer.apply_generator_gradients(zip(grds,
                                             self.genr.trainable_weights))

    return {'disr_rl' : disr_rl,
            'disr_fk' : disr_fk,
            'genr_ls' : genr_ls }

  def get_config(self):
    config = super(GAN, self).get_config()
    config.update({
      'generator'     : tf.keras.layers.serialize(self.genr),
      'discriminator' : tf.keras.layers.serialize(self.disr),
      'augmentation'  : self.augm,
      'augmentprob'   : self.augp,
    })
    return config

class GenEnc(tf.keras.layers.Layer):
  """
  initial encoding layer for generator
  """
  def __init__(self, dim, *args, **kwargs):
    super(GenEnc, self).__init__(*args, **kwargs)
    self.dimn = dim
    return

  def call(self, inputs):
    """
    inputs should be 0|1 labels
    """
    lb = tf.tile(tf.cast(inputs, dtype=tf.float32), [1,self.dimn])
    rn = tf.random.normal(shape=(tf.shape(inputs)[0],self.dimn))
    return tf.concat([rn,lb], axis=-1)

  def get_config(self):
    config = super(GenEnc, self).get_config()
    config.update({
      'dim'   : self.dimn,
    })
    return config

class ProjLabel(tf.keras.layers.Layer):
  """
  project label to given dimension
  """
  def __init__(self, dim, *args, **kwargs):
    super(ProjLabel, self).__init__(*args, **kwargs)
    self.dimn = dim
    return

  def call(self, inputs):
    """
    inputs should be 0|1 labels
    """
    return tf.tile(tf.cast(inputs, dtype=tf.float32), [1,self.dimn])

  def get_config(self):
    config = super(ProjLabel, self).get_config()
    config.update({
      'dim'   : self.dimn,
    })
    return config


## generator build ##
def generator_build(outdim, repdim=8, nblcks=4, nheads=4):
  # create input, gaussian random noise in data shape
  inputs  = tf.keras.Input(shape=[1], dtype=tf.int32, name='rndin')
  # initial data encoding
  out = GenEnc(outdim, name='encod')(inputs)
  out = tf.keras.layers.Flatten()(out)
  out = tf.keras.layers.Dense(units=outdim*repdim)(out)
  out = tf.keras.layers.Reshape((outdim,repdim))(out)
  # transformation(s)
  keydim = repdim // 2
  for i in range(nblcks):
    # attention
    tmp = tf.keras.layers.LayerNormalization()(out)
    tmp = tf.keras.layers.MultiHeadAttention(num_heads=nheads,
                                             key_dim=keydim)(tmp,tmp,tmp)
    out = tf.keras.layers.Add()([out,tmp])
    # feedforward
    tmp = tf.keras.layers.LayerNormalization()(out)
    tmp = tf.keras.layers.Dense(units=repdim, activation=lrsact)(tmp)
    tmp = tf.keras.layers.Dense(units=repdim)(tmp)
    out = tf.keras.layers.Add()([out,tmp])
  # output
  out = tf.keras.layers.Flatten()(out)
  outputs = tf.keras.layers.Dense(units=outdim)(out)
  return Model(inputs=inputs, outputs=(outputs,inputs))

## discriminator build ##
def discriminator_build(indim, repdim=8, nblcks=4, nheads=4, doutrt=0.2):
  # create input
  in1 = tf.keras.Input(shape=[indim], dtype=tf.float32, name='dtin1')
  in2 = tf.keras.Input(shape=[indim], dtype=tf.float32, name='dtin2')
  in3 = tf.keras.Input(shape=[1],     dtype=tf.int32,   name='dtin3')
  lpr = ProjLabel(indim)(in3)
  out = tf.keras.layers.Concatenate(name='concat')([in1,in2,lpr])
  # initial encoding
  out = tf.keras.layers.Flatten()(out)
  out = tf.keras.layers.Dense(units=indim*repdim)(out)
  out = tf.keras.layers.Reshape((indim,repdim))(out)
  # transformation(s)
  keydim = repdim // 2
  for i in range(nblcks):
    # attention
    tmp = tf.keras.layers.LayerNormalization()(out)
    tmp = tf.keras.layers.MultiHeadAttention(num_heads=nheads,
                                             key_dim=keydim,
                                             dropout=doutrt)(tmp,tmp,tmp)
    tmp = tf.keras.layers.Dropout(rate=doutrt)(tmp)
    out = tf.keras.layers.Add()([out,tmp])
    # feedforward
    tmp = tf.keras.layers.LayerNormalization()(out)
    tmp = tf.keras.layers.Dense(units=repdim, activation=lrsact)(tmp)
    tmp = tf.keras.layers.Dense(units=repdim)(tmp)
    tmp = tf.keras.layers.Dropout(rate=doutrt)(tmp)
    out = tf.keras.layers.Add()([out,tmp])
  # flatten and score
  outputs = tf.keras.layers.Flatten(name='flatn')(out)
  outputs = tf.keras.layers.Dense(units=1, name='outpt')(outputs)
  return Model(inputs=[in1,in2,in3], outputs=outputs)


### EXECUTIONS #################################################################

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
                description='generative adversarial network',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # data input
  group = parser.add_argument_group('data')
  group.add_argument('-f', '--file', dest='file',
                     help='numpy data file', metavar='data.npz')

  # model
  group = parser.add_argument_group('model')
  group.add_argument('--repdim', dest='repdim', type=int,
                     help='dimension of internal data representation',
                     metavar='N')
  group.add_argument('--blocks', dest='blocks', type=int,
                     help='number of transformer blocks', metavar='N')
  group.add_argument('--heads', dest='heads', type=int,
                     help='number of transformer heads', metavar='N')
  group.add_argument('--dropout', dest='dropout', type=float,
                     help='discriminator dropout rate', metavar='N')
  group.add_argument('--augmentsdev', dest='augmentsdev', type=float,
                     help='data augmentation stdev', metavar='N')
  group.add_argument('--augmentprob', dest='augmentprob', type=float,
                     help='data augmentation prob', metavar='N')

  # training regime
  group = parser.add_argument_group('training')
  group.add_argument('--batch_size', dest='batch_size', type=int,
                     help='training batch size', metavar='N')
  group.add_argument('--epochs', dest='epochs', type=int,
                     help='number of training epochs', metavar='N')
  group.add_argument('--learning_rate', dest='learn_rate', type=float,
                     help='base learning rate for generator', metavar='N')
  group.add_argument('--learning_rate_mult', dest='learn_rate_mult',
                     type=float,
                     help='discriminator learning rate multiplier',
                     metavar='N')

  parser.set_defaults(file='iusdata_raw.csv',

                      repdim=16,
                      blocks=8,
                      heads=4,
                      dropout=0.2,
                      augmentsdev=0.1,
                      augmentprob=0.8,

                      batch_size=8,
                      epochs=20000,
                      learn_rate=1.0e-5,
                      learn_rate_mult=0.1)

  args = parser.parse_args()


  (datadim, samples, npx, npy, data) = getRealData(args.file, args.batch_size)
  tf.print(datadim,samples,data)

  generator = generator_build(datadim,
                              args.repdim,
                              args.blocks,
                              args.heads)
  generator.summary()

  discriminator = discriminator_build(datadim,
                                      args.repdim,
                                      args.blocks,
                                      args.heads,
                                      args.dropout)
  discriminator.summary()

  # create optimizer
  steps_per_epoch = samples // args.batch_size
  decay_steps     = steps_per_epoch * 50  # reduce lr every 50 epochs
  total_steps     = steps_per_epoch * args.epochs
  decay_rate      = (1.0e-7/args.learn_rate)**(decay_steps/total_steps)

  gsch = tf.keras.optimizers.schedules.ExponentialDecay(\
                  initial_learning_rate=args.learn_rate,
                  decay_steps=decay_steps,
                  decay_rate=decay_rate,
                  staircase=True)
  dsch = tf.keras.optimizers.schedules.ExponentialDecay(\
                  initial_learning_rate=args.learn_rate * args.learn_rate_mult,
                  decay_steps=decay_steps*2,
                  decay_rate=decay_rate,
                  staircase=True)
  gopt = tf.keras.optimizers.SGD(learning_rate=gsch,
                                 momentum=0.8,
                                 nesterov=True)
  dopt = tf.keras.optimizers.SGD(learning_rate=dsch,
                                 momentum=0.8,
                                 nesterov=True)
  opt  = GanOptimizer(gopt, dopt)

  # compile gan
  gan = GAN(generator, discriminator, args.augmentsdev, args.augmentprob)
  gan.compile(optimizer=opt)

  # set up callbacks
  callbacks = []
  tbdir = 'tblog'
  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tbdir))
  callbacks.append(PlotCallback(npx, npy, log_dir=tbdir))

  # fit gan
  gan.fit(data, epochs=args.epochs, callbacks=callbacks)

  # save final generator model
  genfname = 'genr_' + time.strftime("%Y%m%d_%H%M%S")
  gan.genr.save(genfname)

  print('done.')
  sys.exit(0)
