function []= Step_1_DistanceMap(datapath,gestureNum, n_desired_frames,stateNum,displayMap)
% Copyright (C) 2017 Haoyu Chen <chenhaoyucc@icloud.com>,
% center of Machine Vision and Signal Analysis,
% Department of Computer Science and Engineering,
% University of Oulu, Oulu, 90570, Finland
% this step 1 is to calculate the distance with lie group features of each HMM state from
% the given gestures

disp('step 1 start')

%you need to change the bodymodel path here
disp('load body model')
load('./data/Chalearn/body_modelupper.mat'); 
used_joints = ["Hipcenter1","Spine2","Neck3","Head4","ShoulderLeft5","ElbowLeft6", "WristLeft7", "HandLeft8","ElbowRight9", "WristRight10", "ShoulderRight11","HandRight12" ];
n_joints = size(used_joints,2);

%read video database as subfolders
files = dir(datapath);
files(1:2) = [];
count = 1;

%these file contains damaged skeleton joints and will cause reading problem
%badfile = [60 75 108 124 164 217];%test
badfile = [21 64 126 240 266 355 417 464];%train

%initial feature maps%%%
statedistanceMap = zeros(gestureNum,gestureNum);
statedistanceMaps = cell(stateNum,1);

statecountMap = zeros(gestureNum,gestureNum);
statecountMaps = cell(stateNum,1);

% set zero to all the maps first
for e = 1:stateNum
    statedistanceMaps{e} = statedistanceMap;
end
for e = 1:stateNum
    statecountMaps{e} = statecountMap;
end

%%%process train files
for i = 1 : length(files)
    if ismember(i,badfile)
        continue
    else
        message = strcat('training sample processed:',string(100*i*1.0/length(files)),'%');
        disp(message);
        zipfilepath = strcat(datapath,files(i).name);
        unzip(zipfilepath,'./temp/')
        addpath('./temp/');
        
        filenames = strsplit(files(i).name,'.');
        filename = filenames{1};
        
        %get skeleton and label lists of the sequence
        samplelist = load(strcat('./temp/',filename,'_labels.csv'));
        skeletonlist = load(strcat('./temp/',filename,'_skeleton.csv'));
        
        sampleinfo = size(samplelist);
        labelNum = sampleinfo(1);
        
        %disp('initial feature maps')
        features = cell(labelNum, 1);
        action_labels = zeros(labelNum, 1); 

        
        %% process the gestures to extract lie group features
        for ins = 1:labelNum
            
            %get skeleton location of the ins gesture
            actionlabel = samplelist(ins,1);
            startf = samplelist(ins,2);
            endf = samplelist(ins,3);
            actionLen = endf - startf +1;
            
            %get the gesture to be processed
            tmpSkel = skeletonlist(startf:endf,:);
            
            %get normalizationed and reshaped skeleton
            corSkel = normalization_skel(tmpSkel,actionLen,n_joints);
            
            features{ins} = get_se3_lie_algebra_features(corSkel, body_model, n_desired_frames, 'relative_pairs');
            
            action_labels(ins) = actionlabel;
            
            count = count + 1;    
            
        end
        
        %% using the lie group features to calculate the distance
        statlen = 60/stateNum;
        
        for state = 1: stateNum
            statedistanceMap = statedistanceMaps{state};
            statecountMap = statecountMaps{state};

            % iterate the baseline
            for base_g = 1:labelNum
                for comp_g = 1: labelNum
                    
                    se3_in1_features = features{base_g};
                    se3_in2_features = features{comp_g};
                    current_dis = norm(se3_in1_features(:,(state-1)*statlen+1)-se3_in2_features(:,(state-1)*statlen+1));
                
                    %record the gesture and baseline diff to the map
                    statedistanceMap(action_labels(comp_g,1),action_labels(base_g,1)) = statedistanceMap(action_labels(comp_g,1),action_labels(base_g,1)) + current_dis;
                    statecountMap(action_labels(comp_g,1),action_labels(base_g,1)) = statecountMap(action_labels(comp_g,1),action_labels(base_g,1)) + 1;
                
                end
            end
           %save map of that state to final maps
           statedistanceMaps{state} = statedistanceMap;
           statecountMaps{state}= statecountMap;
                
        end 
        
        rmdir  temp s
     end
end
message = strcat('extract:',string(count),' gesture samples');
disp(message);

save('template/statedistanceMaps.mat','statedistanceMaps');
save('template/statecountMaps.mat','statecountMaps');

if displayMap == 1

    maxCoforstate = zeros(length(statedistanceMaps),1);
    for state = 1 : length(statedistanceMaps)
        maxCoforstate(state) = max(max(statedistanceMaps{state}));
    end

    maxEn = max(maxCoforstate);

    %%%draw the maps
    for state = 1 : length(statedistanceMaps)
        A = statedistanceMaps{state};
        A = A / maxEn;
        subplot(2,5,state);
    %     range = [0 maxV];
        imagesc(A,[0.1 0.8]);
        colormap(jet(256));
        title(strcat('state number:',string(state)));

    %     colormap(gray(256));
    %     colormap(autumn(256));
    %     colormap(lines(256));
    end
else
    
end

disp('step 1 done')