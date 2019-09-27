import numpy as np
def tp_fp_fn_counter(predicted_pss, true_pss, lengths):
  pred_dict = {}
  secondary_labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
  for i in range(8):
    pred_dict[i] = {'label':secondary_labels[i], 'TP':0, 'FP':0, 'FN':0}

  for i in range(len(true_pss)):
    for j in range(lengths[i]):
      pred_ = np.argmax(predicted_pss[i][j])
      true_ = np.argmax(true_pss[i][j])
      if pred_ == true_ :
        pred_dict[pred_]['TP'] += 1
      else:
        pred_dict[pred_]['FP'] += 1
        pred_dict[true_]['FN'] += 1
  return pred_dict


def amino_acid_wise_precision_recall_F1(predicted_pss, true_pss, lengths):
  tp_fp_fn_dict = tp_fp_fn_counter(predicted_pss, true_pss, lengths)
  prec_recall_dict = {}
  secondary_labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
  for i in range(8):
    prec_recall_dict[i] = {'label': secondary_labels[i], 'precision': 0, 'recall': 0, 'F1': 0}

  for i in range(8):
    try:
      prec_recall_dict[i]['precision'] = tp_fp_fn_dict[i]['TP'] / (tp_fp_fn_dict[i]['TP'] + tp_fp_fn_dict[i]['FP'])
      prec_recall_dict[i]['recall'] = tp_fp_fn_dict[i]['TP'] / (tp_fp_fn_dict[i]['TP'] + tp_fp_fn_dict[i]['FN'])
      prec_recall_dict[i]['F1'] = 2 * (prec_recall_dict[i]['precision'] * prec_recall_dict[i]['recall']) / (
      prec_recall_dict[i]['precision'] + prec_recall_dict[i]['recall'])
    except:
      # print('All zero values. skipped.')
      pass

  return prec_recall_dict, tp_fp_fn_dict

