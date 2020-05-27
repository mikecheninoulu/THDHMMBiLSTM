% Copyright (C) 2018 Haoyu Chen <chenhaoyucc@icloud.com>,
% center of Machine Vision and Signal Analysis,
% Department of Computer Science and Engineering,
% University of Oulu, Oulu, 90570, Finland

% this code is used to generate the THD (temporal hierarchical dictionary)
% by minizing the relative entropy

clc;
clear;
close all;
clear;

warning('off','all')

dataset = "chalearn";

%set threshold for stage 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

if (dataset == "chalearn")
    datapath = './train/';
    stateNum = 10;
    gestureNum = 20;
    n_desired_frames = 60;

elseif(dataset == "OAD")
    datapath = '/home/haoyu/Documents/5_TIP/step2_THD_HMM-LSTM/OADdataset/data';
    stateNum = 10;
    gestureNum = 10;
    n_desired_frames = 100;
    
elseif(dataset == "MSRAG")
    datapath = '/research/tklab/personal/haoyu/MSRAG/';
    stateNum = 10;
    gestureNum = 10;
    n_desired_frames = 80;
end

% calculate the entropy mapaccording to all the training samples
Step_1_DistanceMap(datapath,gestureNum, n_desired_frames,stateNum,1);

% calculate the mass function map according to the distance of lie group
Step_2_Mass_entropy_Map(gestureNum,stateNum,1);  

% calculate the mass function map according to the distance of lie group

% construct the temporal hierarchical dictionary

% threshold = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
% Step_4_Hierarchicalbuilding(gestureNum, threshold,1);

threshold = [ 0.38, 15, 0, 10, 10, 1, 1, 1, 1, 1];
Step_4_Hierarchicalbuilding(gestureNum, threshold,1);

threshold = [ 0.46, 18, 0.02, 0, 2, 1, 1, 1, 1, 1];
Step_4_Hierarchicalbuilding(gestureNum, threshold,2);

threshold = [ 0.475, 15, 10, 15, 12.5, 5, 2, 1, 1, 1];
Step_4_Hierarchicalbuilding(gestureNum, threshold,3);

threshold = [ 0.48, 15, 5, 15, 15, 10, 5, 1, 1, 1];
Step_4_Hierarchicalbuilding(gestureNum, threshold,4);

Step_3_Mapdraw();

