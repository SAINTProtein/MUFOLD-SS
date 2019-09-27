# MUFOLD-SS
This is an implementation of of the paper [MUFOLD-SS: New Deep Inception-Inside-Inception Networks for Protein Secondary Structure Prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6120586/)

# Download the dataset
##### CB6133 dataset and CB513 dataset
Please download the files *cullpdb+profile_5926_filtered.npy.gz* and *cb513+profile_split1.npy.gz* from [this website](http://www.princeton.edu/~jzthree/datasets/ICML2014/) for training the model on the filtered version of CB6133 dataset(duplicates removed) and testing it on CB513 dataset, then rename the files as **psp.npy.gz** and **CB513.npy.gz** respectively:
##### CASP10 and CASP11 dataset
To test the model on CASP10 and CASP11 benchmark-datasets, please download the files from [this website](https://drive.google.com/drive/folders/1404cRlQmMuYWPWp5KwDtA7BPMpl-vF-d).

Finally, put all the downloaded files in the *./MUFOLD-SS* folder.

# Setup:
The implementation is in python3. Keras 2.2.4 and Tensorflow 1.13.1 were used to build this project. To intall these dependencies please run the following commands(You can use [Anaconda](https://www.anaconda.com/) or [Pip](https://pip.pypa.io/en/stable/installing/) for installation):
##### For pip
> pip install tensorflow-gpu

> pip install keras

##### For Anaconda
> conda install tensorflow-gpu

> conda install keras

# Training the model
To train the model after cloing the repository please run the following commands:
> cd MUFOLD-SS

> python3 train.py

Various parameters can be changed for training or evaluating the model in the file *./MUFOLD-SS/config.py*

# Running pretrained model
##### Evaluation
To run the pretrained model please download the *mufold_ss.h5* from [this link](https://drive.google.com/open?id=1NZ8MO94W-YPfRx60wj2p2bEeYxmEC1LZ). and place it in the *./MUFOLD-SS* folder.
Then run the following command:
> python3 evaluate.py
