function []= Step_2_Mass_entropy_Map(gestureNum,stateNum,displayMap)
% Copyright (C) 2017 Haoyu Chen <chenhaoyucc@icloud.com>,
% center of Machine Vision and Signal Analysis,
% Department of Computer Science and Engineering,
% University of Oulu, Oulu, 90570, Finland
% this step 1 is to calculate the distance with lie group features of each HMM state from
% the given gestures

disp('step 2 start')

fileList = load('./template/statedistanceMaps.mat');
statedistanceMaps = fileList.('statedistanceMaps');

fileList = load('./template/statecountMaps.mat');
statecountMaps = fileList.('statecountMaps');

%initial feature maps%%%
statemassMap1 = zeros(gestureNum,gestureNum);
statemassMaps_sig = cell(stateNum,1);

statemassMap2 = zeros(gestureNum,gestureNum);
statemassMaps_Gaussian = cell(stateNum,1);

statemassMap3 = zeros(gestureNum,gestureNum);
statemassMaps_linear = cell(stateNum,1);

statemassMap4 = zeros(gestureNum,gestureNum);
statemassMaps_piecew = cell(stateNum,1);

% set zero to all the maps first
for e = 1:stateNum
    statemassMaps_sig{e} = statemassMap1;
    statemassMaps_Gaussian{e} = statemassMap2;
    statemassMaps_linear{e} = statemassMap3;
    statemassMaps_piecew{e} = statemassMap4;
end



%initial feature maps%%%
stateentropyMap1 = zeros(gestureNum,gestureNum);
stateentropyMaps_sig = cell(stateNum,1);

stateentropyMap2 = zeros(gestureNum,gestureNum);
stateentropyMaps_Gaussian = cell(stateNum,1);

stateentropyMap3 = zeros(gestureNum,gestureNum);
stateentropyMaps_linear = cell(stateNum,1);

stateentropyMap4 = zeros(gestureNum,gestureNum);
stateentropyMaps_piecew = cell(stateNum,1);

% set zero to all the maps first
for e = 1:stateNum
    stateentropyMaps_sig{e} = stateentropyMap1;
    stateentropyMaps_Gaussian{e} = stateentropyMap2;
    stateentropyMaps_linear{e} = stateentropyMap3;
    stateentropyMaps_piecew{e} = stateentropyMap4;
end

for state = 1: stateNum
    statedistanceMap = statedistanceMaps{state};
    statecountMap = statecountMaps{state};

    % iterate the baseline
    for x = 1:gestureNum
        for y = 1: gestureNum
            
            %calculate the mass function 1 to the map
            mass = (0.5/(1+exp(-(statedistanceMap(x,y)-10))) + 0.5)/statecountMap(x,y);
            statemassMap1(x,y) = mass;
            stateentropyMap1(x,y) = mass .*log2(mass);
            
            %calculate the mass function 2 to the map
            mass = (0.5 +exp(-0.5*(statedistanceMap(x,y)-20).^2))/statecountMap(x,y);
            statemassMap2(x,y) = mass;
            stateentropyMap2(x,y) = mass .*log2(mass);
            
            %calculate the mass function 3 to the map
            mass = (1/40*(statedistanceMap(x,y))+0.5)/statecountMap(x,y);
            statemassMap3(x,y) = min(mass,1);
            stateentropyMap3(x,y) = mass .*log2(mass);
            
            %calculate the mass function 4 to the map
            if statedistanceMap(x,y)<4
                mass = 0.5;
            elseif statedistanceMap(x,y)<9
                mass = 0.6;
            elseif statedistanceMap(x,y)<12
                mass = 0.7;
            elseif statedistanceMap(x,y)<20
                mass = 0.8;
            else 
                mass = 1;
            end
            mass = mass/statecountMap(x,y);
            statemassMap4(x,y) = mass;
            stateentropyMap4(x,y) = mass .*log2(mass);
            
        end
    end
   %save map of that state to final maps
   statemassMaps_sig{state} = statemassMap1;
   statemassMaps_Gaussian{state} = statemassMap2;
   statemassMaps_linear{state} = statemassMap3;
   statemassMaps_piecew{state} = statemassMap4;
   
   %save map of that state to final maps
   stateentropyMaps_sig{state} = -stateentropyMap1;
   stateentropyMaps_Gaussian{state} = -stateentropyMap2;
   stateentropyMaps_linear{state} = -stateentropyMap3;
   stateentropyMaps_piecew{state} = -stateentropyMap4;

end 

save('template/statemassMaps_sig.mat','statemassMaps_sig');
save('template/stateentropyMaps_sig.mat','stateentropyMaps_sig');

save('template/statemassMaps_Gaussian.mat','statemassMaps_Gaussian');
save('template/stateentropyMaps_Gaussian.mat','stateentropyMaps_Gaussian');

save('template/statemassMaps_linear.mat','statemassMaps_linear');
save('template/stateentropyMaps_linear.mat','stateentropyMaps_linear');

save('template/statemassMaps_piecew.mat','statemassMaps_piecew');
save('template/stateentropyMaps_piecew.mat','stateentropyMaps_piecew');




if displayMap == 1
   
    %% sigmoid
    fig1 = figure(1);  
    maxCoforstate = zeros(length(statemassMaps_sig),1);
    for state = 1 : length(statemassMaps_sig)
        maxCoforstate(state) = max(max(statemassMaps_sig{state}));
    end
    
    maxEn = max(maxCoforstate);

    %%%draw the maps
    for state = 1 : length(statemassMaps_sig)
        A = statemassMaps_sig{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([])
        yticks([])
    %     range = [0 maxV];
        imagesc(A,[0.5 1]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    %savefig('template/massfuncMaps_sig.fig')
    print(fig1,'template/massfuncMaps_sig','-dpng')
    print(fig1,'template/massfuncMaps_sig','-dsvg')
    
    fig2 = figure(2);
    maxCoforstate = zeros(length(stateentropyMaps_sig),1);
    for state = 1 : length(stateentropyMaps_sig)
        maxCoforstate(state) = max(max(stateentropyMaps_sig{state}));
    end

    %%%draw the maps
    for state = 1 : length(stateentropyMaps_sig)
        A = stateentropyMaps_sig{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([])
        yticks([])
    %     range = [0 maxV];
        imagesc(A,[0 0.5]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    
    L = findobj(fig1,'type','line');
    a = copyobj(L,findobj(fig2,'type','axes'));
    %savefig('template/entropyMaps_sig.fig')
    print(fig2,'template/entropyMaps_sig','-dpng')
    print(fig2,'template/entropyMaps_sig','-dsvg')
    
    
    %% gaussian
    fig3 = figure(3);
    maxCoforstate = zeros(length(statemassMaps_Gaussian),1);
    for state = 1 : length(statemassMaps_Gaussian)
        maxCoforstate(state) = max(max(statemassMaps_Gaussian{state}));
    end
    maxEn = max(maxCoforstate);

    %%%draw the maps
    for state = 1 : length(statemassMaps_Gaussian)
        A = statemassMaps_Gaussian{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([]);
        yticks([]);
    %     range = [0 maxV];
        imagesc(A,[0.5 1]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    %savefig('template/massfuncMaps_Gaussian.fig')
    print(fig3,'template/massfuncMaps_Gaussian','-dpng')
    print(fig3,'template/massfuncMaps_Gaussian','-dsvg')
    
    fig4 = figure(4);

    %%%draw the maps
    for state = 1 : length(stateentropyMaps_Gaussian)
        A = stateentropyMaps_Gaussian{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([]);
        yticks([]);
    %     range = [0 maxV];
        imagesc(A,[0 0.5]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    %savefig('template/entropyMaps_Gaussian.fig')
    print(fig4,'template/entropyMaps_Gaussian','-dpng')
    print(fig4,'template/entropyMaps_Gaussian','-dsvg')
    
    
    
    %% linear
    fig5 = figure(5);
    maxCoforstate = zeros(length(statemassMaps_linear),1);
    for state = 1 : length(statemassMaps_linear)
        maxCoforstate(state) = max(max(statemassMaps_linear{state}));
    end
    maxEn = max(maxCoforstate);

    %%%draw the maps
    for state = 1 : length(statemassMaps_linear)
        A = statemassMaps_linear{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([]);
        yticks([]);
    %     range = [0 maxV];
        imagesc(A,[0.5 1]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    %savefig('template/massfuncMaps_linear.fig')
    print(fig5,'template/massfuncMaps_linear','-dpng')
    print(fig5,'template/massfuncMaps_linear','-dsvg')
    
    fig6 = figure(6);
    %%%draw the maps
    for state = 1 : length(stateentropyMaps_linear)
        A = stateentropyMaps_linear{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([]);
        yticks([]);
    %     range = [0 maxV];
        imagesc(A,[0 0.5]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    
    %savefig('template/entropyMaps_linear.fig')
    print(fig6,'template/entropyMaps_linear','-dpng')
    print(fig6,'template/entropyMaps_linear','-dsvg')
    
    %% piecew
    fig7 = figure(7);
    maxCoforstate = zeros(length(statemassMaps_piecew),1);
    for state = 1 : length(statemassMaps_piecew)
        maxCoforstate(state) = max(max(statemassMaps_Gaussian{state}));
    end
    maxEn = max(maxCoforstate);

    %%%draw the maps
    for state = 1 : length(statemassMaps_piecew)
        A = statemassMaps_piecew{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([]);
        yticks([]);
    %     range = [0 maxV];
        imagesc(A,[0.5 1]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    %savefig('template/massfuncMaps_piecew.fig')
    print(fig7,'template/massfuncMaps_piecew','-dpng')
    print(fig7,'template/massfuncMaps_piecew','-dsvg')
    
    fig8 = figure(8);
    %%%draw the maps
    for state = 1 : length(stateentropyMaps_piecew)
        A = stateentropyMaps_piecew{state};
        %A = A / maxEn;
        subplot(2,5,state);
        xticks([]);
        yticks([]);
    %     range = [0 maxV];
        imagesc(A,[0 0.5]);
        colormap(jet(256));
        %title(strcat('state:',string(state)));
    end
    %savefig('template/entropyMaps_piecew.fig')
    print(fig8,'template/entropyMaps_piecew','-dpng')
    print(fig8,'template/entropyMaps_piecew','-dsvg')
else
    
end

disp('step 2 done')