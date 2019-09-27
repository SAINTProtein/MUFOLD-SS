import keras.backend as K

def truncated_accuracy(y_true, y_predict):
  mask = K.sum(y_true, axis=2)
  y_pred_labels = K.cast(K.argmax(y_predict, axis=2), 'float32')
  y_true_labels = K.cast(K.argmax(y_true, axis=2), 'float32')
  is_same = K.cast(K.equal(
    y_true_labels, y_pred_labels), 'float32')
  num_same = K.sum(is_same * mask, axis=1)
  lengths = K.sum(mask, axis=1)
  return K.mean(num_same / lengths, axis=0)