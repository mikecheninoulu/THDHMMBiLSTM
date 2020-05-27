#-------------------------------------------------------------------------------
# Name:        THD-HMM for gesture recognition utilities
# Purpose:     provide some toolkits
# Copyright (C) 2018 Haoyu Chen <chenhaoyucc@icloud.com>,
# author: Chen Haoyu

# center of Machine Vision and Signal Analysis,
# Department of Computer Science and Engineering,
# University of Oulu, Oulu, 90570, Finland

# this code is based on the Starting Kit for ChaLearn LAP 2014 Track3
# and Di Wu: stevenwudi@gmail.com DBN implement for CVPR
# thanks for their opensource
#-------------------------------------------------------------------------------
""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

"""
import shutil
import gc
import copy
import numpy
import random
import cv2
from PIL import Image, ImageDraw
import os
from functools import partial
from scipy.ndimage.filters import gaussian_filter
import time
import pickle
import re
from sklearn import preprocessing
import scipy.io as sio
from Att_BiLSTM_training import test_att_BiLSTM
from keras.models import *
from statistics import mode
from scipy import interpolate
from scipy.io import loadmat
import sys

def print_toolbar(rate, annotation=''):

    toolbar_width = 30
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")

'''
load training dataset
'''
def loader(parsers):

    # start counting tim
    time_tic = time.time()

    #counting feature number
    HMM_state_feature_count = 0

    #how many joints are used
    njoints = parsers['njoints']

    #HMM temporal steps
    Time_step_NO = parsers['STATE_NUM']#3

    frame_compens = int((parsers['LSTM_step']-1)/2 + 2)
    #feature dimension of the LSTM input
    featurenum = parsers['featureNum'] * parsers['LSTM_step']

    ges_list = parsers['ges_list']

    #get sample list
    Sample_list = parsers['train_list']

    if parsers['THD']:
        statePath = loadmat('THD_OAD_10.mat')
        THD = statePath['statePath']-1
        # print(THD)
        dictionaryNum = THD.max()+1
        # print(dictionaryNum)
    else:
        dictionaryNum = parsers['class_count']*parsers['STATE_NUM']


    if parsers['segmentation']:
        dictionaryNum = dictionaryNum+1

    #using normalization
    if parsers['normalization']:
        limbmodelPath = 'all_limb_lengths.mat'
        limbmodel = sio.loadmat(limbmodelPath)
        limblist = limbmodel['limblenlist']
    else:
        limblist= []

    '''pre-allocating the memory '''
    #gesture features
    Feature_all_states = numpy.zeros(shape=(800000, featurenum), dtype=numpy.float32)
    Targets_all_states = numpy.zeros(shape=(800000, dictionaryNum), dtype=numpy.uint8)

    # HMM pror and transition matrix
    Prior = numpy.zeros(shape=(dictionaryNum))
    Transition_matrix = numpy.zeros(shape=(dictionaryNum,dictionaryNum))

    #start traversing samples
    for sampleID in Sample_list:

        print("\t Processing file " + str(sampleID))
        label_List = []
        # Get the list of actions for this frame
        f = open(parsers['data'] + str(sampleID) + "/label/label.txt", "r")
        label_List = f.read().splitlines()
        #print(label_List)

        '''gather all the ges info in this sample'''
        ges_info_list = []
        gestureID = -1
        # Iterate for each action in this sample
        for label_info in label_List:
            #print(label_info)
            # Get the gesture ID, and start and end frames for the gesture
            if label_info in ges_list:
                gestureID = ges_list.index(label_info)
            else:
                ges_info_list.append([int(label_info.split(' ')[0]),int(label_info.split(' ')[1]),int(gestureID)])

        print(ges_info_list)

        '''traverse all ges in samples'''
        #get the skeletons of the gesture
        for ges_info in ges_info_list:
            [startFrame, endFrame, gestureID] = ges_info
            if parsers['online']:
                startFrame = startFrame-16
            frame_count = endFrame-startFrame+1+8
            # print(startFrame)
            # print(endFrame)
            # print(gesID)
            # print()
            '''1 extract skeleton features of this ges'''
            '''process the gesture parts'''
            skeleton_all = numpy.zeros(shape=(frame_count, njoints*3), dtype=numpy.float32)
            for numFrame in range(startFrame-4, endFrame+4):
                # Get the Skeleton object for this frame
                f = open(parsers['data'] + str(sampleID) + "/skeleton/" + str(numFrame) + ".txt", "r")
                label_List = f.read().splitlines()
                for joint in range(njoints):
                    skeleton_all[numFrame-startFrame+4][joint*3:joint*3+3] = [float(label_List[joint].split(' ')[0]),
                    float(label_List[joint].split(' ')[1]),float(label_List[joint].split(' ')[2])]

                #print(skeleton_all.shape)

                '''interpolate the gesture'''
                if parsers['interpolate_train']:
                    skeleton_all_new = numpy.zeros(shape=(parsers['maxframe'], njoints*3), dtype=numpy.float32)
                    for joint_num in range(njoints*3):
                        x = numpy.linspace(0, 1, frame_count)
                        y = skeleton_all[:,joint_num]

                        x_new = numpy.linspace(0, 1, parsers['maxframe'])
                        tck = interpolate.splrep(x, y)
                        y_bspline = interpolate.splev(x_new, tck)
                        skeleton_all_new[:,joint_num] = y_bspline
                        skeleton_all = skeleton_all_new

            #get the temporal features for LSTM of the gesture
            #print(frame_count)
            sample_feature, feature_count = extract_temporal_movingPose(skeleton_all,frame_compens,parsers['LSTM_step'],njoints)
            #print(feature_count)
            #generatre the corresponding labels
            if parsers['THD']:
                sample_label = extract_THD_HMMstate_label(Time_step_NO, feature_count, dictionaryNum, gestureID,THD)
            else:
                sample_label = extract_HMMstate_label(Time_step_NO, feature_count, dictionaryNum, gestureID)

            #assign seg_length number of features to current gesutre
            Feature_all_states[HMM_state_feature_count:HMM_state_feature_count + feature_count, :] = sample_feature
            #assign seg_length number of labels to corresponding features
            Targets_all_states[HMM_state_feature_count:HMM_state_feature_count + feature_count, :] = sample_label
            #update feature count
            HMM_state_feature_count = HMM_state_feature_count + feature_count


            '''2 extract HMM transition info of this ges'''
            for frame in range(frame_count):
                # print(gestureID)
                if parsers['THD']:
                    state_no_1,state_no_2 = THD_HMMmatrix(gestureID, frame, frame_count, Time_step_NO,THD)
                else:
                    state_no_1,state_no_2 = HMMmatrix(gestureID, frame, frame_count, Time_step_NO)# print(state_no_1)
                # print(state_no_2)
                ## we allow first two states add together:
                Prior[state_no_1] += 1
                Transition_matrix[state_no_1, state_no_2] += 1

    # save the feature file:
    Feature_all,Targets_all,SK_normalizationfilename = process_feature(parsers,Feature_all_states,Targets_all_states,HMM_state_feature_count)

    print ("Processing data done with consuming time %d sec" % int(time.time() - time_tic))

    return Prior, Transition_matrix, Feature_all,Targets_all,SK_normalizationfilename


'''
sparse data list
'''
def process_feature(parsers, Feature_all,Targets_all,action_feature_count):

    # save the feature file:
    print ('total training samples: ' + str(action_feature_count))
    Feature_all = Feature_all[0:action_feature_count, :]
    Targets_all = Targets_all[0:action_feature_count, :]


    #random the samples
    rand_num = numpy.random.permutation(Feature_all.shape[0])
    Feature_all = Feature_all[rand_num]
    Targets_all  = Targets_all[rand_num]


    #[train_set_feature_normalized, Mean1, Std1]  = preprocessing.scale(train_set_feature)
    scaler = preprocessing.StandardScaler().fit(Feature_all)
    Mean1 = scaler.mean_
    Std1 = scaler.scale_
    Feature_all = normalize(Feature_all,Mean1,Std1)

    # save the normalization files
    SK_normalizationfilename = parsers['outpath']+ parsers['experi_ID'] +'SK_normalization.pkl'

    f = open(SK_normalizationfilename,'wb')
    pickle.dump( {"Mean1": Mean1, "Std1": Std1 },f)
    f.close()

    return Feature_all,Targets_all, SK_normalizationfilename

def interpolate_skeleton(parsers, skeleton_all, njoints, intp):

    if intp:
        #compen for Movingpose
        skeleton_all_intped = numpy.zeros(shape=(parsers['maxframe'], njoints*3), dtype=numpy.float32)

        for joint_num in range(njoints*3):

            x = numpy.linspace(0, 1, skeleton_all.shape[0])
            y = skeleton_all[:,joint_num]

            x_new = numpy.linspace(0, 1, parsers['maxframe'])
            tck = interpolate.splrep(x, y)
            y_bspline = interpolate.splev(x_new, tck)

            skeleton_all_intped[:,joint_num] = y_bspline

            skeleton = skeleton_all_intped

    else:
        skeleton = skeleton_all

    return skeleton

'''
using DNN to get HMM emission probability
'''
def emission_prob(modelname, parsers, Feature, Mean1, Std1,best_threshold,modeltype,iflog=True):
    Feature_normalized = normalize(Feature, Mean1, Std1)
    #print Feature_normalized.max()
    #print Feature_normalized.min()

    # feed to  Network
    y_pred_4 = test_att_BiLSTM(modelname, parsers, Feature_normalized, best_threshold)
    #y_pred_4 = test_torch_BiLSTM(modelname, parsers, Feature_normalized,best_threshold, modeltype, iflog)

    # if iflog:
    log_observ_likelihood = numpy.log(y_pred_4.T)
    # else:
    #     log_observ_likelihood = y_pred_4.T
    return log_observ_likelihood

'''
save features
'''
def datasaver(parsers, Prior, Transition_matrix,Feature_all,Targets_all):

    '''define name'''

    Prior_Transition_matrix_filename = parsers['outpath']+ parsers['experi_ID'] + str(parsers['STATE_NUM']) + 'Prior_Transition_matrix.mat'
    Feature_filename = parsers['outpath'] + parsers['experi_ID'] + str(parsers['STATE_NUM']) + 'feature'+ str(parsers['featureNum']) +'Feature_all.pkl'

    '''HMM transition state'''
    #save HMM transition matrix
    sio.savemat( Prior_Transition_matrix_filename, {'Transition_matrix':Transition_matrix, 'Prior': Prior})

    '''feature storage'''
    # save the skeleton file:
    f = open(Feature_filename, 'wb')
    pickle.dump({"Feature_all": Feature_all, "Targets_all": Targets_all }, f,protocol=4)
    f.close()

    return Prior_Transition_matrix_filename, Feature_filename

'''
package Parameters into one file
'''
def packagePara(parsers, model1name, norm_name,HMM_file):

    Paras = {'model1':model1name,
             'norm_para':norm_name,
             'HMM_model':HMM_file,
             }
    path = parsers['outpath'] + parsers['experi_ID'] + '/Paras.pkl'
    afile = open(path, 'wb')
    pickle.dump(Paras, afile)
    afile.close()

'''
test result
'''
def tester(parsers,best_threshold):

    if parsers['THD']:
        statePath = loadmat('THD_OAD_10.mat')
        THD = statePath['statePath']-1
        # print(THD)
        dictionaryNum = THD.max()+1
        # print(dictionaryNum)
    else:
        dictionaryNum = parsers['class_count']*parsers['STATE_NUM']

    MODEL4 = load_model(parsers['outpath']+ parsers['experi_ID'] +'/my_model.h5')
    #print(ges_info_list)
    correct_count = 0.0
    total_count = 0.0
    acc_total = 0.0

    time_tic = time.time()
    datacheck = []

    #njoints = len(parsers['used_joints'])

    #dictionaryNum = parsers['class_count']*parsers['STATE_NO'] + 1

    #Time_step_NO = parsers['STATE_NO']

    #featurenum =  njoints*3

    if parsers['normalization']:
        limbmodelPath = 'all_limb_lengths.mat'
        limbmodel = sio.loadmat(limbmodelPath)
        limblist = limbmodel['limblenlist']
    else:
        limblist= []

    frame_compens = int((parsers['LSTM_step']-1)/2 + 2)
    #feature dimension of the LSTM input
    featurenum = parsers['featureNum'] * parsers['LSTM_step']

    path = parsers['outpath']+ parsers['experi_ID'] + '/Paras.pkl'
    file2 = open(path, 'rb')
    Paras = pickle.load(file2)
    file2.close()

    ### load the pre-store normalization constant
    f = open(Paras['norm_para'],'rb')
    SK_normalization = pickle.load(f)
    Mean1 = SK_normalization ['Mean1']
    Std1 = SK_normalization['Std1']

    ## Load networks
    modelname1 = Paras['model1']

    ## Load Prior and transitional Matrix
    # dic=sio.loadmat('Prior_Transition_matrix.mat')
    dic=sio.loadmat(Paras['HMM_model'])
    Transition_matrix = dic['Transition_matrix']
    Prior = dic['Prior']

    ## Load trained networks
    njoints = parsers['njoints']

    #get sample list
    ges_list = parsers['ges_list']

    #get sample list
    Sample_list = parsers['test_list']

    if parsers['segmentation']:
        dictionaryNum = dictionaryNum+1

    #using normalization
    if parsers['normalization']:
        limbmodelPath = 'all_limb_lengths.mat'
        limbmodel = sio.loadmat(limbmodelPath)
        limblist = limbmodel['limblenlist']
    else:
        limblist= []

    #start traversing samples
    for sampleID in Sample_list:

        # Create the object to access the sample
        print("\t Processing file " + str(sampleID))
        label_List = []
        # Get the list of actions for this frame
        f = open(parsers['data'] + str(sampleID) + "/label/label.txt", "r")
        label_List = f.read().splitlines()
        #print(label_List)

        '''gather all the ges info in this sample'''
        ges_info_list = []
        gestureID = -1
        # Iterate for each action in this sample
        for label_info in label_List:
            #print(label_info)
            # Get the gesture ID, and start and end frames for the gesture
            if label_info in ges_list:
                gestureID = ges_list.index(label_info)
            else:
                ges_info_list.append([int(label_info.split(' ')[0]), int(label_info.split(' ')[1]), int(gestureID)])

        print(ges_info_list)
        '''traverse all ges in samples'''
        #get the skeletons of the gesture


        for ges_info in ges_info_list:
            time_single = time.time()
            [startFrame, endFrame, gestureID] = ges_info
            if parsers['online']:
                startFrame = startFrame -16

            frame_count = endFrame-startFrame+1+8
            # print(ges_info)

            '''1 extract skeleton features of this ges'''
            '''process the gesture parts'''
            skeleton_all = numpy.zeros(shape=(frame_count, njoints*3), dtype=numpy.float32)

            for numFrame in range(startFrame-4,endFrame+4):
                # Get the Skeleton object for this frame
                f = open(parsers['data'] + str(sampleID) + "/skeleton/" + str(numFrame) + ".txt", "r")
                label_List = f.read().splitlines()
                for joint in range(njoints):
                    skeleton_all[numFrame-startFrame+4][joint*3:joint*3+3] = [float(label_List[joint].split(' ')[0]),
                    float(label_List[joint].split(' ')[1]),float(label_List[joint].split(' ')[2])]

            '''interpolate the gesture'''
            if parsers['interpolate_test']:
                skeleton_all_new = numpy.zeros(shape=(parsers['maxframe'], njoints*3), dtype=numpy.float32)
                for joint_num in range(njoints*3):
                    x = numpy.linspace(0, 1, frame_count)
                    y = skeleton_all[:,joint_num]

                    x_new = numpy.linspace(0, 1, parsers['maxframe'])
                    tck = interpolate.splrep(x, y)
                    y_bspline = interpolate.splev(x_new, tck)
                    skeleton_all_new[:,joint_num] = y_bspline
                    skeleton_all = skeleton_all_new

            #get the temporal features for LSTM of the gesture
            sample_feature, feature_len = extract_temporal_movingPose(skeleton_all, frame_compens,parsers['LSTM_step'], njoints)
            #ratio = 0.8
            #visiblenumber = int(ratio* (frame_count))

            # sample_feature1 = copy.copy(sample_feature)
            # observ_likelihood1 = emission_prob(modelname1, parsers, sample_feature1, Mean1, Std1, best_threshold, parsers['netmodel'],False)

            sample_feature2 = copy.copy(sample_feature)
            log_observ_likelihood1 = emission_prob(MODEL4, parsers, sample_feature2, Mean1, Std1, best_threshold, parsers['netmodel'])

    #            attentionweight = numpsy.zeros(shape= (log_observ_likelihood1.shape[1]))
    #            log_observ_likelihood2 = emission_prob(modelname2, parsers,sample_feature[0:visiblenumber,:], Mean1, Std1,best_threshold,3)
    #            for fra in range(log_observ_likelihood1.shape[1]):
    #                attentionweight[fra] = numpy.std(observ_likelihood1[:,fra])
    #
            # attentionweight = sum(observ_likelihood1)
            #     #        log_observ_likelihood[-1, 0:5] = 0
            #     #        log_observ_likelihood[-1, -5:] = 0
            #
            # mask1 = attentionweight < 50
            # mask2 = attentionweight > -50
            #
            # mask = mask1*mask2
            log_observ_likelihood = log_observ_likelihood1#[:,mask]# + log_observ_likelihood1
            #print("\t Viterbi path decoding " )
            #do it in log space avoid numeric underflow
            # print(log_observ_likelihood.shape)
            #print(visiblenumber)
            #print(Transition_matrix)
            [path, _, global_score] = viterbi_path_log(numpy.log(Prior), numpy.log(Transition_matrix), log_observ_likelihood)
            # print(path[0:50])
            if parsers['THD']:
                pred_label = viterbi_get_gesture_THD(path, THD, parsers['STATE_NUM'])
            else:
                pred_label = viterbi_get_gesture(path, parsers['STATE_NUM'])
            # print(path)
            # print(THD[:,-1])
            # print(pred_label)
            # print(gestureID)
            if gestureID == pred_label:
                correct_count = correct_count + 1
            total_count = total_count +1
            print("Used time %d sec, processing speed %f fps" %(int(time.time() - time_single),frame_count/float(time.time() - time_single)))

    print ("Processing testing data done with consuming time %d sec" % int(time.time() - time_tic))

    acc_total = correct_count/total_count

    print(parsers['experi_ID']+". The score for this prediction is " + "{:.12f}".format(acc_total))

    numpy.savetxt(parsers['outpath']+ parsers['experi_ID'] +'_score_'+ str(acc_total) +'.txt', [])
    # print(parsers['experi_ID']+". The score for this prediction is " + "{:.12f}".format(acc_overal))

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def correct_pos(pos, mt_switch, t_value, max_shift):
    nfrms = len(mt_switch)
    found = False
    shift = 0
    while not found and shift < max_shift:
        if mt_switch[max(0, pos - shift)] == t_value:
            found = True
            pos -= shift
        elif mt_switch[min(pos+shift, nfrms-1)] == t_value:
            found = True
            pos += shift
        elif pos-shift <= 0 and pos+shift >= nfrms-1:
            found = True
        shift +=1
    return pos

def check_gesture_length(gl, min_gl, max_gl):
    if gl < min_gl or gl > max_gl:
        return False
    else:
        return True

'''
record pose features
'''

def extract_skeleton(parsers, used_joints, smp, startFrame, endFrame,limblist):

    if limblist == []:
        #get the skeleton sequence of that gesture
        joinsCor = Extract_jointCor_UNnormalized(used_joints, smp, startFrame, endFrame)
        #get the skeleton sequence and normalizating
        joinsCor_norm = joinsCor
    else:
        #get the skeleton sequence of that gesture
        joinsCor = Extract_jointCor_UNnormalized(used_joints, smp, startFrame, endFrame)
        #get the skeleton sequence and normalizating
        joinsCor_norm = Extract_joinsCor_Normalized(joinsCor, used_joints, startFrame, endFrame, limblist)

    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))

    for joints in range(len(used_joints)):
        Skeleton_matrix[:, joints*3: (joints+1)*3] = joinsCor_norm[used_joints[joints]][:,:]
    return Skeleton_matrix,  Skeleton_matrix.shape[0]


def extract_temporal_skeleton(skeleton_all,LSTM_step, njoints):
    frame_count =  skeleton_all.shape[0]-LSTM_step+1
    Cordimension = skeleton_all.shape[1]
    absconnection_num = njoints * (njoints-1)/2
    feature_dim = Cordimension + absconnection_num
    feature_n  = numpy.zeros(shape=(frame_count, (feature_dim)*LSTM_step))

    #absolute pose
    FeatureNum = 0
    F_abs = numpy.zeros(shape=(skeleton_all.shape[0], njoints * (njoints-1)/2))

    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            all_X = skeleton_all[:, joints1*3] - skeleton_all[:, joints2*3]
            all_Y = skeleton_all[:, joints1*3+1] - skeleton_all[:, joints2*3+1]
            all_Z = skeleton_all[:, joints1*3+2] - skeleton_all[:, joints2*3+2]

            Abs_distance = numpy.sqrt(all_X**2 + all_Y**2 + all_Z**2)
            F_abs[:, FeatureNum] = Abs_distance
            FeatureNum += 1


    feature_all = numpy.concatenate((skeleton_all, F_abs), axis=1)

    for frame in range(frame_count):

        for step in range(LSTM_step):

            feature_n[frame, step*feature_dim:(step+1)*feature_dim] = feature_all[frame+step,:]
            #feature_n[frame, step*feature_dim:(step+1)*feature_dim] = feature_all[frame+step,:]

            #feature_n= Extract_moving_pose_Feature(Skeleton_matrix, len(used_joints))


    return feature_n, feature_n.shape[0]

def extract_temporal_movingPose(skeleton_all, frame_compens , LSTM_step, njoints):

    frame_count =  skeleton_all.shape[0] - frame_compens*2

    feature_all = Extract_moving_pose_Feature(skeleton_all, njoints)

    feature_dim = feature_all.shape[1]
    #print('featuredim')
    # print(feature_dim)
    # print(LSTM_step)
    # print(frame_count)

    feature_n  = numpy.zeros(shape=(frame_count, feature_dim*LSTM_step))

    for frame in range(frame_count):

        for step in range(LSTM_step):

            feature_n[frame, step*feature_dim:(step+1)*feature_dim] = feature_all[frame+step,:]

            #feature_n= Extract_moving_pose_Feature(Skeleton_matrix, len(used_joints))


    return feature_n, feature_n.shape[0]

def extract_Feature(parsers, used_joints, smp, startFrame, endFrame,limblist):

    if limblist == []:
        #get the skeleton sequence of that gesture
        joinsCor = Extract_jointCor_UNnormalized(used_joints, smp, startFrame, endFrame)
        #get the skeleton sequence and normalizating
        joinsCor_norm = joinsCor
    else:
        #get the skeleton sequence of that gesture
        joinsCor = Extract_jointCor_UNnormalized(used_joints, smp, startFrame, endFrame)
        #get the skeleton sequence and normalizating
        joinsCor_norm = Extract_joinsCor_Normalized(joinsCor, used_joints, startFrame, endFrame, limblist)

    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))

    for joints in range(len(used_joints)):
        Skeleton_matrix[:, joints*3: (joints+1)*3] = joinsCor_norm[used_joints[joints]][:,:]

    if parsers['movingPosefeature']:
        feature_n= Extract_moving_pose_Feature(Skeleton_matrix, len(used_joints))
    else:
        feature_n= Extract_feature_Accelerate(Skeleton_matrix, len(used_joints))

    return feature_n, feature_n.shape[0]


def extract_HMMstate_label(STATE_NO, action_count, dictionaryNum, gestureID):
    # label the features
    target = numpy.zeros(shape=(action_count, dictionaryNum))
    # HMM states force alignment
    for i in range(STATE_NO):
        # get feature index of the current time step
        begin_feature_index = int(numpy.round(action_count * i / STATE_NO) + 1)
        end_feature_index = int(numpy.round(action_count * (i + 1) / STATE_NO))
        # get feature length of the current time step
        seg_length = end_feature_index - begin_feature_index + 1
        labels = numpy.zeros(shape=(dictionaryNum, 1))
        # assign the one hot labels
        labels[ i + STATE_NO*gestureID] = 1
        #print(begin_feature_index-1)
        #print(end_feature_index)
        target[begin_feature_index-1:end_feature_index,:] = numpy.tile(labels.T, (seg_length, 1))
    return target

def extract_THD_HMMstate_label(STATE_NO, action_count, dictionaryNum, gestureID,THD):
    # label the features
    target = numpy.zeros(shape=(action_count, dictionaryNum))
    # HMM states force alignment
    for i in range(STATE_NO):
        # get feature index of the current time step
        begin_feature_index = int(numpy.round(action_count * i / STATE_NO) + 1)
        end_feature_index = int(numpy.round(action_count * (i + 1) / STATE_NO))
        # get feature length of the current time step
        seg_length = end_feature_index - begin_feature_index + 1
        labels = numpy.zeros(shape=(dictionaryNum, 1))
        # assign the one hot labels
        THD_index = THD[gestureID,i]
        labels[THD_index] = 1
        #print(begin_feature_index-1)
        #print(end_feature_index)
        target[begin_feature_index-1:end_feature_index,:] = numpy.tile(labels.T, (seg_length, 1))
    return target


def extract_nonActionlabel (action_count, dictionaryNum):
    target_n = numpy.zeros(shape=(action_count, dictionaryNum))
    target_n[:,-1] = 1
    return target_n

'''
record HMM transition matrix
'''
def HMMmatrix(gestureID, frame, frame_count,STATE_NO):
    state_no_1 = numpy.floor(STATE_NO*(frame*1.0/(frame_count+3)))
    state_no_1 = int(state_no_1+STATE_NO*(gestureID))
    state_no_2 = numpy.floor(STATE_NO*((frame+1)*1.0/(frame_count+3)))
    state_no_2 = int(state_no_2+STATE_NO*(gestureID))
    return state_no_1,state_no_2

'''
record THD HMM transition matrix
'''
def THD_HMMmatrix(gestureID, frame, frame_count,STATE_NO,THD):
    state_no_1_index = int(numpy.floor(STATE_NO*(frame*1.0/(frame_count+3))))
    THD_index = THD[gestureID,state_no_1_index]
    state_no_1 = THD_index
    state_no_2_index = int(numpy.floor(STATE_NO*((frame+1)*1.0/(frame_count+3))))
    THD_index = THD[gestureID,state_no_2_index]
    state_no_2 = THD_index
    return state_no_1,state_no_2

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
  """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

def zero_mean_unit_variance(Data):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    Mean = numpy.mean(Data, axis=0)
    Data  -=  Mean

    Std = numpy.std(Data, axis = 0)
    index = (numpy.abs(Std<10**-5))
    Std[index] = 1
    Data /= Std
    return [Data, Mean, Std]

def normalize(Data, Mean, Std):
    Data -= Mean
    Data /= Std
    return Data

def Extract_jointCor_UNnormalized( used_joints, smp, startFrame, endFrame):
    """
    Extract original features
    """

    joinsCor = dict();

    for joinIdx in used_joints:
        joinsCor[joinIdx] = numpy.zeros(shape=(endFrame-startFrame+1, 3))

    frame_num = 0

    for numFrame in range(startFrame,endFrame+1):
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)

        for joint in used_joints:
            joinsCor[joint][frame_num] = skel.joins[joint][0]

        frame_num += 1

#     for joints in range(len(used_joints)):
#         for i in range(3):
#             joinsCor[used_joints[joints]][:,i] = gaussian_filter(joinsCor[used_joints[joints]][:,i],sigma=1)
#
    return joinsCor

def Extract_joinsCor_Normalized(joinsCor, used_joints, startFrame, endFrame, limblist):
    """
    Extract skeleton and normalize them
    """
    SkeletonConnectionMap = (['HipCenter', 'ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                             ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                             ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'])

    joinsCor_norm = dict();
    for joinIdx in used_joints:
        joinsCor_norm[joinIdx]=numpy.zeros(shape=(endFrame-startFrame+1, 3))


    frame_num = 0
    for numFrame in range(startFrame,endFrame+1):
        #hip
        joinsCor_norm['HipCenter'][frame_num,:] = [0.0,0.0,0.0]
        for idx, link in enumerate(SkeletonConnectionMap):
            p1 = joinsCor[link[1]][frame_num,:]
            p2 = joinsCor[link[0]][frame_num,:]
            p1 = numpy.asarray(p1)
            p2 = numpy.asarray(p2)
            if numFrame == 0:
                if numpy.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2) != 0:
                    joinsCor_norm[link[1]][frame_num,:] = joinsCor_norm[link[0]][frame_num,:] + limblist[idx, 0] * (p1-p2)/numpy.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
            else:
                if numpy.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2) == 0:
                    joinsCor_norm[link[1]][frame_num,:] = joinsCor_norm[link[1]][frame_num-1,:]
                else:
                    joinsCor_norm[link[1]][frame_num,:] = joinsCor_norm[link[0]][frame_num,:] + limblist[idx, 0] * (p1-p2)/numpy.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        frame_num += 1

    return joinsCor_norm

def Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame):
    """
    Extract original features
    """
    frame_num = 0
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))

    for numFrame in range(startFrame,endFrame+1):
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        frame_num += 1


    if numpy.allclose(sum(sum(numpy.abs(Skeleton_matrix))),0):
        valid_skel = False
    else:
        valid_skel = True

    return Skeleton_matrix, valid_skel



def Extract_feature_Realtime(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], njoints * (njoints-1)/2*3))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1

    #F_cp
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[1:,joints1*3:(joints1+1)*3]-Pose[0:-1,joints2*3:(joints2+1)*3]
            FeatureNum += 1

    #Instead of initial frame as in the paper Eigenjoints-based action recognition using
    #naive-bayes-nearest-neighbor, we use final frame because it's better initiated
    # F_cf

    Features = numpy.concatenate( (Fcc[0:-1, :], Fcp), axis = 1)
    return Features

def Extract_feature_Accelerate(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], njoints * (njoints-1)/2*3))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1

    #F_cp --joint velocities
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-2, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[2:,joints1*3:(joints1+1)*3]-Pose[0:-2,joints2*3:(joints2+1)*3]
            FeatureNum += 1

    #F_ca --joint accelerations
    FeatureNum = 0
    Fca= numpy.zeros(shape=(Pose.shape[0]-4, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fca[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[4:,joints1*3:(joints1+1)*3] + Pose[0:-4,joints2*3:(joints2+1)*3] - 2 * Pose[2:-2,joints2*3:(joints2+1)*3]
            FeatureNum += 1

    Features = numpy.concatenate( (Fcc[0:-4, :], Fcp[0:-2,:], Fca), axis = 1)
    return Features


def Extract_moving_pose_Feature(Skeleton_matrix_Normalized, njoints):

    #pose
    F_pose = Skeleton_matrix_Normalized

    #velocity
    F_velocity = Skeleton_matrix_Normalized[2:,:] - Skeleton_matrix_Normalized[0:-2,:]

    #accelerate
    F_accelerate = Skeleton_matrix_Normalized[4:,:] + Skeleton_matrix_Normalized[0:-4,:] - 2 * Skeleton_matrix_Normalized[2:-2,:]

    #absolute pose
    FeatureNum = 0
    F_abs = numpy.zeros(shape=(Skeleton_matrix_Normalized.shape[0], int(njoints * (njoints-1)/2)))

    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            all_X = Skeleton_matrix_Normalized[:, joints1*3] - Skeleton_matrix_Normalized[:, joints2*3]
            all_Y = Skeleton_matrix_Normalized[:, joints1*3+1] - Skeleton_matrix_Normalized[:, joints2*3+1]
            all_Z = Skeleton_matrix_Normalized[:, joints1*3+2] - Skeleton_matrix_Normalized[:, joints2*3+2]

            Abs_distance = numpy.sqrt(all_X**2 + all_Y**2 + all_Z**2)
            F_abs[:, FeatureNum] = Abs_distance
            FeatureNum += 1

#     print(F_pose.shape)
#     print(F_velocity.shape)
#     print(F_accelerate.shape)
#     print(F_abs.shape)

    Features = numpy.concatenate((F_pose[2:-2, :], F_velocity[1:-1,:], F_accelerate, F_abs[2:-2, :]), axis = 1)
    return Features


def viterbi_path_log(prior, transmat, observ_likelihood):
    """ Viterbi path decoding
    Wudi first implement the forward pass.
    Future works include forward-backward encoding
    input: prior probability 1*N...
    transmat: N*N
    observ_likelihood: N*T
    """
    T = observ_likelihood.shape[-1]
    N = observ_likelihood.shape[0]

    path = numpy.zeros(T, dtype=numpy.int32)
    global_score = numpy.zeros(shape=(N,T))
    predecessor_state_index = numpy.zeros(shape=(N,T), dtype=numpy.int32)

    t = 1
    global_score[:, 0] =  observ_likelihood[:, 0]
    # need to  normalize the data

    for t in range(1, T):
        for j in range(N):
            temp = global_score[:, t-1] + transmat[:, j] + observ_likelihood[j, t]
            global_score[j, t] = max(temp)
            predecessor_state_index[j, t] = temp.argmax()

    path[T-1] = global_score[:, T-1].argmax()

    for t in range(T-2, -1, -1):
        path[t] = predecessor_state_index[ path[t+1], t+1]


    return [path, predecessor_state_index, global_score]


def viterbi_colab_flat(path, global_score, state_no = 5, threshold=-3, mini_frame=15, cls_num = 10):
    """
    Clean the viterbi path output according to its global score,
    because some are out of the vocabulary
    """
    # just to accommodate some frame didn't start right from the begining
    all_label = state_no * cls_num # 20 vocabularies
    start_label = numpy.concatenate((range(0,all_label,state_no), range(1,all_label,state_no),range(2,all_label,state_no)))
    end_label   = numpy.concatenate((range(state_no-3,all_label,state_no), range(state_no-2,all_label,state_no),range(state_no-1,all_label,state_no)))
#     start_label = range(0,all_label,state_no)
#     end_label   = range(4,all_label,state_no)
    begin_frame = []
    end_frame = []
    pred_label = []

    frame = 1
    while(frame < path.shape[-1]-1):
        if path[frame-1]==all_label and path[frame] in start_label:
            begin_frame.append(frame)
            # python integer divsion will do the floor for us :)
            pred_label.append( path[frame]/state_no + 1)
            while(frame < path.shape[-1]-1):
                if path[frame] in end_label and path[frame+1]==all_label:
                    end_frame.append(frame)
                    break
                else:
                    frame += 1
        frame += 1

    end_frame = numpy.array(end_frame)
    begin_frame = numpy.array(begin_frame)
    pred_label= numpy.array(pred_label)
    # risky hack! just for validation file 663
    if len(begin_frame)> len(end_frame):
        begin_frame = begin_frame[:-1]
        pred_label = pred_label[:-1]

    elif len(begin_frame)< len(end_frame):# risky hack! just for validation file 668
        end_frame = end_frame[1:]
    ## First delete the predicted gesture less than 15 frames
    frame_length = end_frame - begin_frame
    ## now we delete the gesture outside the vocabulary by choosing
    ## frame number small than mini_frame
    mask = frame_length > mini_frame
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]


    Individual_score = []
    for idx, g in enumerate(begin_frame):
            score_start = global_score[path[g], g]
            score_end = global_score[path[end_frame[idx]], end_frame[idx]]
            Individual_score.append(score_end - score_start)
    ## now we delete the gesture outside the vocabulary by choosing
    ## score lower than a threshold
    Individual_score = numpy.array(Individual_score)
    frame_length = end_frame - begin_frame
    # should be length independent
    Individual_score = Individual_score/frame_length

    order = Individual_score.argsort()
    ranks = order.argsort()

    mask = Individual_score > threshold
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]
    Individual_score = Individual_score[mask]


    return [pred_label, begin_frame, end_frame, Individual_score, frame_length]


def viterbi_get_gesture(path, state_no = 5):
    """
    Clean the viterbi path output according to its global score,
    because some are out of the vocabulary
    """
    # just to accommodate some frame didn't start right from the beginin
    # python integer divsion will do the floor for us :)
    #print(path[-1])
    pred_label = int(path[-1]/state_no)

    return pred_label
def viterbi_get_gesture_THD(path, THD, state_no = 5):
    """
    Clean the viterbi path output according to its global score,
    because some are out of the vocabulary
    """
    # just to accommodate some frame didn't start right from the beginin
    # python integer divsion will do the floor for us :)

    pred_label = numpy.where(THD[:,-1] == path[-1])

    return pred_label[0]

def skeletonToImage(framenum, joinsCor, width,height,bgColor):
    """ Create an image for the skeleton information """
    SkeletonConnectionMap = (['HipCenter','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                             ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                             ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'])
    im = Image.new('RGB', (width, height), bgColor)
    draw = ImageDraw.Draw(im)

    for link in SkeletonConnectionMap:

        joinsCor[link[1]][framenum,:]
        print (joinsCor[link[0]][framenum,:])

    for link in SkeletonConnectionMap:

        print (joinsCor[link[1]][framenum,:])
        print (joinsCor[link[0]][framenum,:])
        p = list(numpy.asarray(joinsCor[link[1]][framenum,0:2]*100)+200)
        p.extend(list(numpy.asarray(joinsCor[link[0]][framenum,0:2]*100)+200))
        draw.line(p, fill=(255,0,0), width=5)
    for node in ['HipCenter','ShoulderCenter','Head','ShoulderLeft','ElbowLeft','WristLeft','HandLeft','ShoulderRight','ElbowRight','WristRight','HandRight']:
        p=list(numpy.asarray(joinsCor[node][framenum,0:2]*100)+200)
        r=5
        draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
    del draw
    image = numpy.array(im)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
