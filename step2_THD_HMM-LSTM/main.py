#-------------------------------------------------------------------------------
# Name:        THD-HMM for gesture recognition
# Purpose:     The main script to run the project
# Copyright (C) 2018 Haoyu Chen <chenhaoyucc@icloud.com>,
# author: Chen Haoyu
# @center of Machine Vision and Signal Analysis,
# Department of Computer Science and Engineering,
# University of Oulu, Oulu, 90570, Finland
#-------------------------------------------------------------------------------

#from BiRNN_Training import train_BiLSTM,BiRNN
from THDutils import loader, datasaver, packagePara, tester
from Att_BiLSTM_training import train_att_BiLSTM

import os, shutil
#from keras.models import load_model
# train_list = [1, 2]
# test_list = [0]
train_list = [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51, 54, 57, 58]
test_list = [0, 10, 13, 17, 21, 26, 27, 28, 29, 36, 40, 41, 42, 43, 44, 45, 52, 53, 55, 56]
ges_list = ['drinking', 'eating', 'gargling', 'opening cupboard', 'opening microwave oven',
'washing hands', 'wiping', 'writing', 'sweeping', 'Throwing trash']

PC = 'chen'
if PC == 'chen':
    outpath = './experi/'
if PC == 'csc':
    outpath = '/wrk/chaoyu/DHori_experidata/'

if __name__ == '__main__':
    parsers = {
            'STATE_NUM' :10,
            'training_steps':14,
            'segmentation': False,
            'online':True,
            'outpath':outpath,
            'Trainingmode':True,
            'Testingmode': True,
            'THD': False,
            'interpolate_train': False,
            'interpolate_test': False,
            # Data folder (Training data)
            'data': "./OADdataset/data/",
            'train_list': train_list,
            'test_list': test_list,
            'ges_list': ges_list,
            'normalization': False,
            'MovingPose':True,
            #used joint list
            'used_joints': ['HipCenter', 'ShoulderCenter', 'Head',
               'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
               'ShoulderRight','ElbowRight','WristRight','HandRight'],
            'class_count' : 10,
            'maxframe':200,
            # get total state number of the THD model
            # Training Parameters
            'netmodel':4,
            'batch_size':64,
            'LSTM_step' : 5,
            'njoints':25,
            # get total state number of the THD model
            'featureNum' : 525}

    if parsers['Trainingmode']:
        parsers['experi_ID'] = 'state' + str(parsers['STATE_NUM']) + '_Bilstm' + str(parsers['netmodel']) + '_epoch'+ str(parsers['training_steps'])+'/'
        if parsers['segmentation']:
            parsers['experi_ID'] = 'seg_' + parsers['experi_ID']
        if parsers['THD']:
            parsers['experi_ID'] = 'THD_' + parsers['experi_ID']
        expripath = parsers['outpath'] + parsers['experi_ID']
        print(expripath)
        if os.path.exists(expripath):
            shutil.rmtree(expripath)
        os.mkdir(expripath)
        mode = 'train'
        #
        Prior, Transition_matrix,Feature_all,Targets_all,norm_name = loader(parsers)

        HMM_file_train, train_Feature_file = datasaver(parsers, Prior, Transition_matrix,Feature_all,Targets_all)

        #HMM_file = 'flat10Prior_Transition_matrix_complete.mat'
        #Feature_file = 'flat10feature154Feature_all_complete.pkl'
        #norm_name = 'origstep_10_state201_fea_154SK_normalization_complete.pkl'

        model1name, best_threshold = train_att_BiLSTM(parsers, train_Feature_file)
        #model1name,best_threshold = train_torch_BiLSTM(parsers, train_Feature_file, parsers['netmodel'])
        #model2name,best_threshold = train_torch_BiLSTM(parsers, Feature_file,3)
        #model1name = 'snapshot_acc_96.8750_loss_0.123691_iter_51000_model.pt'
        model2name = []
        packagePara(parsers,model1name,norm_name,HMM_file_train)

    if parsers['Testingmode']:
        parsers['experi_ID'] = 'state' + str(parsers['STATE_NUM']) + '_Bilstm' + str(parsers['netmodel']) + '_epoch'+ str(parsers['training_steps'])+'/'
        if parsers['segmentation']:
            parsers['experi_ID'] = 'seg_' + parsers['experi_ID']
        if parsers['THD']:
            parsers['experi_ID'] = 'THD_' + parsers['experi_ID']
        expripath = parsers['outpath'] + parsers['experi_ID']
        best_threshold = 0
        #MODEL4 = load_model(filepath = parsers['outpath']+ parsers['experi_ID'] +'/model.ckpt')
        #filepath = parsers['outpath']+ parsers['experi_ID'] +'/model.ckpt'
        tester(parsers,best_threshold)
