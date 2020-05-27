function []= Step_4_Hierarchicalbuilding(gestureNum, threshold, num)
% Copyright (C) 2017 Haoyu Chen <chenhaoyucc@icloud.com>,
% center of Machine Vision and Signal Analysis,
% Department of Computer Science and Engineering,
% University of Oulu, Oulu, 90570, Finland

% this step shows how to generate the hierarchical structure based on entropy map
% entropy map is obtained from previous steps

disp('step 4 start')
%read the entropy templates
fileList = load('./template/statedistanceMaps.mat');
statecountMaps = fileList.('statedistanceMaps');

stateNum = length(statecountMaps);

clusterPath = zeros(gestureNum,stateNum);
gesPath = zeros(gestureNum,stateNum);

maxCoforstate = zeros(length(statecountMaps),1);
for state = 1 : length(statecountMaps)
    maxCoforstate(state) = max(max(statecountMaps{state}));
end
maxEn = max(maxCoforstate);

%%%draw the maps
for state = 1 : length(statecountMaps)
    
    if state == 1 
        disp('clustering state:');
        disp(state);
        A = statecountMaps{state};
        Apro = A;
        
        % max the symmetry axis value for coding need
        for i = 1:gestureNum
            Apro(i,i) = 10000;
        end
        
        %make a tmp dictionary to cluster the gesture
        temp = 1:1:gestureNum;
        
        %initial the flag for the cluster number
        flag = 1;
        
        %for current state loop until there is no more gesture element
        while ~isempty(temp)
            %         disp('cluster:');
            %         disp(flag);
            %         disp('cluster size:')
            %         disp(length(temp));
            % form the bag in current temp dic
            tempBag= Apro(:,temp);
            
            %find the min value
            comV = min(min(tempBag));
            indeV = find(comV==tempBag);
            comL = fix(indeV(1)/size(tempBag,1));
            if comL == 0
                comL = 1;
            end
            basegesind = comL;
            
            %set baseges to the flag number
            clusterPath(temp(basegesind),state) = flag;
            gesPath(temp(basegesind),state) = temp(basegesind);
            
            %compare the rest to this base gesture
            tempBag = [];
            for ges = 1:length(temp)
                if Apro(temp(basegesind),temp(ges))<threshold(state)
                    clusterPath(temp(ges),state)= flag;
                    gesPath(temp(ges),state) = temp(ges);
                else
                    if temp(ges) ~= temp(basegesind)
                        tempBag = [tempBag;temp(ges)];
                    end
                end
            end
            temp = tempBag;
            flag = flag +1;
        end
    else
        disp('clustering state:');
        disp(state);
        A = statecountMaps{state};
        Apro = A;
        % interate for each cluster in last state
        for c = 1:max(clusterPath(:,state-1))
            temp = [];
            %get all the gesture in the same cluster in last stage
            for j = 1:gestureNum
                if clusterPath(j,state-1) == c
                    temp = [temp j];
                end
            end

            %for current state loop until there is no more gesture element
            while length(temp)~=0
                %         disp('cluster:');
                %         disp(flag);
                %         disp('cluster size:')
                %         disp(length(temp));
                % form the bag in current temp dic
                tempBag= Apro(:,temp);
                
                %find the min value
                comV = min(min(tempBag));
                indeV = find(comV==tempBag);
                comL = fix(indeV(1)/size(tempBag,1));
                if comL == 0
                    comL = 1;
                end
                basegesind = comL;
                
                %set baseges to the flag number
                clusterPath(temp(basegesind),state) = flag;
                gesPath(temp(basegesind),state) = temp(basegesind);
                
                %compare the rest to this base gesture
                
                tempBag = [];
                for ges = 1:length(temp)
                    if Apro(temp(basegesind),temp(ges))<threshold(state)
                        clusterPath(temp(ges),state)= flag;
                        gesPath(temp(ges),state) = temp(ges);
                    else
                        if temp(ges) ~= temp(basegesind)
                            tempBag = [tempBag;temp(ges)];
                        end
                    end
                end
                
                temp = tempBag;
                flag = flag +1;
            end    
        end      
    end
end

% set the HMM statepath
%since state 5 has differ all the gesture 6,7,8 stages are not assogned wit
%hstate. 9,10 stages are used for ending
statePath = zeros(gestureNum,stateNum);
statePath(:,1:5) = clusterPath(:,1:5);
statePath(:,6) = clusterPath(:,5);
statePath(:,7) = clusterPath(:,5);
statePath(:,8) = clusterPath(:,5);
statePath(:,9) = (max(clusterPath(:,5))+1:1:max(clusterPath(:,5))+gestureNum);
statePath(:,10) = statePath(:,9)+gestureNum;
%statePath = clusterPath;

statePath_todraw = zeros(gestureNum,stateNum);
statePath_todraw(:,1:5) = sort(clusterPath(:,1:5));
statePath_todraw(:,6) = sort(clusterPath(:,5));
statePath_todraw(:,7) = sort(clusterPath(:,5));
statePath_todraw(:,8) = sort(clusterPath(:,5));
statePath_todraw(:,9) = (max(clusterPath(:,5))+1:1:max(clusterPath(:,5))+gestureNum);
statePath_todraw(:,10) = statePath_todraw(:,9);
%statePath = clusterPath;

figure()
plot(statePath_todraw);
    %     range = [0 maxV];
imagesc(statePath_todraw,[0 200]);
colormap(jet(256));
colorbar

disp('clustering state finish!');
for st = 1: stateNum
	disp('final state path:');
	disp(st);
    disp(statePath(:,st)');
end

disp('clustering state finish!');
for st = 1: stateNum
	disp('final state path:');
	disp(st);
    disp(clusterPath(:,st)');
end

[B,I] = sort(statePath);

save(strcat('threshold',string(num),'stateNum',string(stateNum),'statePath.mat'),'statePath');

disp('step 4 done')
