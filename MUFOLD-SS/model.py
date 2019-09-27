import config
from keras.models import Model
from keras.layers import RepeatVector, Multiply, Flatten, Dot, Softmax, Lambda, Add, BatchNormalization, Dropout, concatenate
from keras.layers import Input,SpatialDropout1D, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, Masking, Bidirectional
from keras.regularizers import l2


def inceptionBlock(x):
  x = BatchNormalization()(x)
  conv1_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv1_1 = Dropout(config.conv_layer_dropout_rate)(conv1_1)
  conv1_1 = BatchNormalization()(conv1_1)

  conv2_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv2_1 = Dropout(config.conv_layer_dropout_rate)(conv2_1)
  conv2_1 = BatchNormalization()(conv2_1)
  conv2_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2_1)
  conv2_2 = Dropout(config.conv_layer_dropout_rate)(conv2_2)
  conv2_2 = BatchNormalization()(conv2_2)

  conv3_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv3_1 = Dropout(config.conv_layer_dropout_rate)(conv3_1)
  conv3_1 = BatchNormalization()(conv3_1)
  conv3_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_1)
  conv3_2 = Dropout(config.conv_layer_dropout_rate)(conv3_2)
  conv3_2 = BatchNormalization()(conv3_2)
  conv3_3 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_2)
  conv3_3 = Dropout(config.conv_layer_dropout_rate)(conv3_3)
  conv3_3 = BatchNormalization()(conv3_3)
  conv3_4 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_3)
  conv3_4 = Dropout(config.conv_layer_dropout_rate)(conv3_4)
  conv3_4 = BatchNormalization()(conv3_4)

  concat = concatenate([conv1_1, conv2_2, conv3_4])
  concat = BatchNormalization()(concat)

  return concat


def deep3iBLock(x):
  block1_1 = inceptionBlock(x)

  block2_1 = inceptionBlock(x)
  block2_2 = inceptionBlock(block2_1)

  block3_1 = inceptionBlock(x)
  block3_2 = inceptionBlock(block3_1)
  block3_3 = inceptionBlock(block3_2)
  block3_4 = inceptionBlock(block3_3)

  concat = concatenate([block1_1, block2_2, block3_4])
  concat = BatchNormalization()(concat)

  return concat


def get_model():
  pssm_input = Input(shape=(700, 21,), name='pssm_input')
  seq_input = Input(shape=(700, 22,), name='seq_input')

  main_input = concatenate([seq_input, pssm_input])

  block1 = deep3iBLock(main_input)

  block2 = deep3iBLock(block1)

  conv11 = Convolution1D(100, 11, activation='relu', padding='same', kernel_regularizer=l2(0.001))(block2)

  dense1 = TimeDistributed(Dense(units=256, activation='relu'))(conv11)
  dense1 = Dropout(config.dense_layer_dropout_rate)(dense1)

  main_output = TimeDistributed(Dense(units=8, activation='softmax', name='main_output'))(dense1)

  model = Model([pssm_input, seq_input], main_output)
  return model
