# Temporal_hierarchical_dictionary_HMM
This code is for the paper: Temporal Hierarchical Dictionary Guided Decodingfor Online Gesture Segmentation and Recognition. We propose a novel hybrid HMM-DNN framework for online segmentation and recognition of skeleton-based human gestures. The network is tested on the four datasets, MSRA, OAD gesture dataset, DHG gesture dataset and Chalearn 2014 dataset. We report state-of-the-art performances on all these datasets.

Code written by Chen Haoyu, University of Oulu.

The original code is written in Theano, we re-implemented it with Keras-Tensorflow.

We train and evaluate on Ubuntu 16.04, it will also work for Windows and OS.

## Code structure
* step1_generateEntropyMap is used for generating THD-HMM with entropy maps
* step2_THD_HMM-LSTM is used for training the BiLSTM with the generated THD-HMM for recognition and segmentation

## Quick start (example with OAD dataset)
* We already prepare the pretrained THD-HMM dictionary for OAD dataset, which can be used for training and validing the networkds to recognize the gestures directly. So you can skip step1_generateEntropyMap and move to step2_THD_HMM-LSTM directly.

* 1. Download the dataset and put it into folder ./step2_THD_HMM-LSTM, OAD can be download from [here](http://www.icst.pku.edu.cn/struct/Projects/OAD.html)

* 2. run the code with:
 `python main.py`
 
* 3. check the experiment results in the folder `./experi/`.

## Train your own THD-HMM (example with Chalearn 2014 dataset)

* 1. Download the dataset and put it into folder ./step1_THD_HMM-LSTM, Chalearn dataset can be download from [here](http://sunai.uoc.edu/chalearnLAP/), note that we use Track 3: gesture recognition, for validating. 

* 2. run the code `main_THD.m` with matlab.

* 3. check the generated THD-HMM and calculated Entropy maps in the folder `./template/`.

## Environments
Ubuntu 16.04 <br>
Python 3.6.5 <br>
Keras 2.3.1  <br>
Tensorflow==1.15.0 <br>
