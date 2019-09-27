

## cb6133 dataset link: http://www.princeton.edu/~jzthree/datasets/ICML2014/
## cb513 dataset link: http://www.princeton.edu/~jzthree/datasets/ICML2014/
## casp10 and casp11 dataset link:

train_dataset_dir = "psp.npy.gz"
best_model_file = './mufold_ss.h5'

cb513_dataset_dir = "CB513.npy.gz"
casp10_dataset_dir = "casp10.h5"
casp11_dataset_dir = "casp11.h5"

load_saved_model = True
to_train = False
lr = 0.0005
conv_layer_dropout_rate = 0.4
dense_layer_dropout_rate = 0.5

evaluate_on_cb513 = True
evaluate_on_casp10 = True
evaluate_on_casp11 = True

show_F1score_cb513 = True
show_F1score_casp10 = True
show_F1score_casp11 = True

