from utility import *
import config
from precision_recall_F1score_utils import *
from model import *
from metric import *
import keras
from keras.callbacks import LambdaCallback
from keras import callbacks, backend
from keras.optimizers import Adam
from pprint import pprint
import gc

###................ load the model...........###
model = get_model()
if config.load_saved_model:
  best_model_file = config.best_model_file
  model = keras.models.load_model(best_model_file, custom_objects={'LayerNormalization':LayerNormalization,'shape_list':shape_list,'backend':backend,'MyLayer':WeightedSumLayer, 'truncated_accuracy':truncated_accuracy})

adam = Adam(lr=config.lr)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              sample_weight_mode='temporal',
              metrics=['accuracy', 'mae', truncated_accuracy])

if config.evaluate_on_cb513 or config.show_F1score_cb513:
  print('Testing on CB513...\n')
  ###............... load and process CB513 to validate ..............###

  dataindex = list(range(35, 56))
  labelindex = range(22, 30)
  print('Loading Test data [ CB513 ]...')
  cb513 = load_gz(config.cb513_dataset_dir)
  cb513 = np.reshape(cb513, (514, 700, 57))
  # print(cb513.shape)
  x_test = cb513[:, :, dataindex]
  y_test = cb513[:, :, labelindex]

  cb513_protein_one_hot = cb513[:, :, : 21]
  cb513_protein_one_hot_with_noseq = cb513[:, :, : 22]
  lengths_cb = np.sum(np.sum(cb513_protein_one_hot, axis=2), axis=1).astype(int)
  # print(cb513_protein_one_hot_with_noseq.shape)
  del cb513_protein_one_hot
  gc.collect()

  cb513_seq = np.zeros((cb513_protein_one_hot_with_noseq.shape[0], cb513_protein_one_hot_with_noseq.shape[1]))
  for j in range(cb513_protein_one_hot_with_noseq.shape[0]):
    for i in range(cb513_protein_one_hot_with_noseq.shape[1]):
      datum = cb513_protein_one_hot_with_noseq[j][i]
      cb513_seq[j][i] = int(decode(datum))

  cb_scores = model.evaluate([x_test, cb513_protein_one_hot_with_noseq], y_test)
  # print(cb_scores)
  print("Accuracy: " + str(round(cb_scores[3]*100, 2)) + "%, Loss: " + str(cb_scores[0])+'\n\n')
  ### for amino_acid_wise_precision_recall_F1
  if config.show_F1score_cb513:
    print('Calcualting Precision, Recall and F1 score...\n')
    y_pred_casp = model.predict([x_test, cb513_protein_one_hot_with_noseq], verbose=1)
    precision_recall_F1_dict = amino_acid_wise_precision_recall_F1(y_pred_casp, y_test, lengths=lengths_cb)
    print('Precision, Recall and F1-score (per amino-acid)')
    pprint(precision_recall_F1_dict[0])
    print('\n\n')
    print('False Negative(FN), False Positive(FP), True Positive counts(TP) counts (per amino-acid)')
    pprint(precision_recall_F1_dict[1])
    print('\n\n')


if config.evaluate_on_casp10 or config.show_F1score_casp10:
  print('Testing on CASP10...\n')
  casp_one_hot, casp_one_hot_token, casp_pssm, casp_label, lengths_casp, casp_protein_one_hot_with_noseq, casp_pos = load_casp(casp='casp10')
  casp_scores10 = model.evaluate([casp_pssm, casp_protein_one_hot_with_noseq], casp_label)
  print("Accuracy: " + str(round(casp_scores10[3]*100, 2)) + "%, Loss: " + str(casp_scores10[0])+'\n\n')
  ### for amino_acid_wise_precision_recall_F1
  if config.show_F1score_casp10:
    print('Calcualting Precision, Recall and F1 score...\n')
    y_pred_casp = model.predict([casp_pssm, casp_protein_one_hot_with_noseq], verbose=1)
    precision_recall_F1_dict = amino_acid_wise_precision_recall_F1(y_pred_casp, casp_label, lengths=lengths_casp)
    print('Precision, Recall and F1-score (per amino-acid)')
    pprint(precision_recall_F1_dict[0])
    print('\n\n')
    print('False Negative(FN), False Positive(FP), True Positive(TP) counts (per amino-acid)')
    pprint(precision_recall_F1_dict[1])
    print('\n\n')

if config.evaluate_on_casp11 or config.show_F1score_casp11:
  print('Testing on CASP11...\n')
  casp_one_hot, casp_one_hot_token, casp_pssm, casp_label, lengths_casp, casp_protein_one_hot_with_noseq, casp_pos = load_casp(casp='casp11')
  casp_scores11 = model.evaluate([casp_pssm, casp_protein_one_hot_with_noseq], casp_label)
  print("Accuracy: " + str(round(casp_scores11[3]*100, 2)) + "%, Loss: " + str(casp_scores11[0])+'\n\n')
  ### for amino_acid_wise_precision_recall_F1
  if config.show_F1score_casp11:
    print('Calcualting Precision, Recall and F1 score...\n')
    y_pred_casp = model.predict([casp_pssm, casp_protein_one_hot_with_noseq], verbose=1)
    precision_recall_F1_dict = amino_acid_wise_precision_recall_F1(y_pred_casp, casp_label, lengths=lengths_casp)
    print('Precision, Recall and F1-score (per amino-acid)')
    pprint(precision_recall_F1_dict[0])
    print('\n\n')
    print('False Negative(FN), False Positive(FP), True Positive(TP) counts (per amino-acid)')
    pprint(precision_recall_F1_dict[1])
    print('\n\n')

