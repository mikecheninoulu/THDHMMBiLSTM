function [meanentropy1]= Recalculate_entropy(stateentropyMaps1, THDenable, num)
% Copyright (C) 2017 Haoyu Chen <chenhaoyucc@icloud.com>,
% center of Machine Vision and Signal Analysis,
% Department of Computer Science and Engineering,
% University of Oulu, Oulu, 90570, Finland

% this step shows how to generate the hierarchical structure based on entropy map
% entropy map is obtained from previous steps

meanentropy1 = zeros(length(stateentropyMaps1),1);

if THDenable
    fileList = load(strcat('threshold',string(num),'stateNum',string(length(stateentropyMaps1)),'statePath.mat'));
    statePath = fileList.('statePath');
end

% calculate current entropy map all  
N = size(stateentropyMaps1{1},1);
baseseq = 1:N;
compseq = 1:N;

for state = 1 : length(stateentropyMaps1)
    tem_entro = 0;
    stateentropyMap = stateentropyMaps1{state};
    if THDenable
        baseseq = statePath(:,state);
        compseq = statePath(:,state);
    end
    
    for baseind = 1:N
        for compind = 1:N
            if baseseq(baseind)~= compseq(compind)
                tem_entro = tem_entro + stateentropyMap(baseind,compind);
            end
        end
    end
    
    meanentropy1(state) = tem_entro/(N*N);
end
