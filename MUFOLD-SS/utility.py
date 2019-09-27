import tensorflow as tf
import gzip
from tensorflow.python.client import device_lib
import numpy as np
import os
import numpy as np
import subprocess
from keras.initializers import Ones, Zeros
from keras.layers import Layer
from copy import deepcopy
import h5py
import config

def check_gpu():
  print(device_lib.list_local_devices())
  if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
  else:
    print("Please install GPU version of TF")


def load_gz(path):
  f = gzip.open(path, 'rb')
  return np.load(f)

def decode(x): # used to take make the tokens
  return np.argmax(x)

class LayerNormalization(Layer):
  def __init__(self, eps: float = 1e-5, **kwargs) -> None:
    self.eps = eps
    super().__init__(**kwargs)

  def build(self, input_shape):
    self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
    self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
    super().build(input_shape)

  def call(self, x, **kwargs):
    u = K.mean(x, axis=-1, keepdims=True)
    s = K.mean(K.square(x - u), axis=-1, keepdims=True)
    z = (x - u) / K.sqrt(s + self.eps)
    return self.gamma * z + self.beta

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
      'eps': self.eps,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def load_casp(casp='casp10'):
  casp10 = None
  if casp == 'casp10':
    print("Loading Test data [ CASP10 ]...")
    casp10 = h5py.File(config.casp10_dataset_dir)

  elif casp == 'casp11':
    print("Loading Test data [ CASP11 ]...")
    casp10 = h5py.File(config.casp11_dataset_dir)

  # print (casp10.shape)
  datahot = casp10['features'][:, :, 0:21]  # sequence feature

  datapssm = casp10['features'][:, :, 21:42]  # profile feature
  labels = casp10['labels'][:, :, 0:8]  # secondary struture label

  lengths_cb = np.sum(np.sum(datahot, axis=2), axis=1).astype(int)

  testhot = datahot
  testlabel = labels
  testpssm = datapssm

  casp10_protein_one_hot_with_noseq = np.zeros(
    (int(testhot.shape[0]), int(testhot.shape[1]), int(testhot.shape[2] + 1)))
  # casp10_protein_one_hot_with_noseq = deepcopy(testhot)
  # casp10_protein_one_hot_with_noseq = np.array(a, order='F')


  test_hot = np.zeros((testhot.shape[0], testhot.shape[1]))
  for i in range(testhot.shape[0]):
    for j in range(testhot.shape[1]):
      casp10_protein_one_hot_with_noseq[i, j, :-1] = testhot[i, j, :]
      if np.sum(testhot[i, j, :]) == 0:
        test_hot[i, j] = 21.0
        casp10_protein_one_hot_with_noseq[i][j][-1] = 1.0
      else:
        test_hot[i, j] = np.argmax(testhot[i, j, :])

        #             if np.argmax(casp10_protein_one_hot_with_noseq[i,j]) != np.argmax(testhot[i,j]) :
        #               print('>>', i, j)
        #               print(casp10_protein_one_hot_with_noseq[i,j])
        #               print(testhot[i,j])

  casp10_pos = np.array(range(700))
  casp10_pos = np.repeat([casp10_pos], testpssm.shape[0], axis=0)
  #print(casp10_pos.shape)

  return datahot, test_hot, testpssm, testlabel, lengths_cb, casp10_protein_one_hot_with_noseq, casp10_pos


class StepDecay():
  def __init__(self, initAlpha=0.0005, factor=0.8, dropEvery=40):
    self.initAlpha = initAlpha
    self.factor = factor
    self.dropEvery = dropEvery

  def __call__(self, epoch):
    exp = np.floor((epoch + 1) / self.dropEvery)
    alpha = self.initAlpha * (self.factor ** exp)
    return float(alpha)
