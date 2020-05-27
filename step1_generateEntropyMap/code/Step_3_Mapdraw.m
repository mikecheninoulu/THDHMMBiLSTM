function []= Step_3_Mapdraw()
% Copyright (C) 2017 Haoyu Chen <chenhaoyucc@icloud.com>,
% center of Machine Vision and Signal Analysis,
% Department of Computer Science and Engineering,
% University of Oulu, Oulu, 90570, Finland

% this step shows how to draw entropy map of each state from the given gesture
% entropy map is obtained from previous steps

%read the DTW templates
fileList = load('./template/stateentropyMaps_sig.mat');
stateentropyMaps_sig = fileList.('stateentropyMaps_sig');

% fileList = load('./template/stateentropyMaps_Gaussian.mat');
% stateentropyMaps_Gaussian = fileList.('stateentropyMaps_Gaussian');

fileList = load('./template/stateentropyMaps_linear.mat');
stateentropyMaps_linear = fileList.('stateentropyMaps_linear');

fileList = load('./template/stateentropyMaps_piecew.mat');
stateentropyMaps_piecew = fileList.('stateentropyMaps_piecew');


meanentropy_sig =  Recalculate_entropy(stateentropyMaps_sig, 0,1);
% meanentropy_Gaussian =  Recalculate_entropy(stateentropyMaps_Gaussian, 0,1);
meanentropy_linear =  Recalculate_entropy(stateentropyMaps_linear, 0,1);
meanentropy_piecew =  Recalculate_entropy(stateentropyMaps_piecew, 0,1);

fig2 = figure(2);
hold on

a1 = plot(meanentropy_sig,'-o','LineWidth',2);M1 = "Projection Sigmoid";
% a2 = plot(meanentropy_Gaussian,'-p','LineWidth',2);M2 = "Projection function 2";
a3 = plot(meanentropy_linear,'-p','LineWidth',2);M3= "Projection ReLU";
a4 = plot(meanentropy_piecew,'-p','LineWidth',2);M4 = "Projection Step";
% legend([a1; a2; a3; a4], [M1; M2; M3; M4]);
legend([a1; a3; a4], [M1; M3; M4]);

ylabel('The relative entropy/bits');
xlabel('The time step of a gesture');
ylim([0 0.5]);
print(fig2,'template/projection','-dpdf')
hold off

%draw the entropy line
meanentropy_sig = Recalculate_entropy(stateentropyMaps_sig, 0,1)
newentropy1 = Recalculate_entropy(stateentropyMaps_sig, 1,1)
newentropy2 = Recalculate_entropy(stateentropyMaps_sig, 1,2)
newentropy3 = Recalculate_entropy(stateentropyMaps_sig, 1,3)
newentropy4 = Recalculate_entropy(stateentropyMaps_sig, 1,4)



fig1 = figure(1);
hold on

b = plot(meanentropy_sig,'-o','LineWidth',2,'Color', [0 0.2 0.2]);N = "The orignial relative entropy";
b1 = plot(newentropy1,'LineWidth',2, 'Color', [0 0.4 0.4]);N1 = "The threshold of 0.4 bits";
b2 = plot(newentropy2,'LineWidth',2, 'Color', [0 0.6 0.6]);N2 = "The threshold of 0.3 bits";
b3 = plot(newentropy3,'LineWidth',2, 'Color', [0 0.8 0.8]);N3 = "The threshold of 0.2 bits";
b4 = plot(newentropy4,'LineWidth',2, 'Color', [0 1 1]);N4 = "The threshold of 0.1 bits";
legend([b; b1; b2; b3; b4], [N; N1; N2; N3; N4]);
ylabel('The relative entropy/bits');
xlabel('The time step of a gesture');
ylim([0 0.5]);
hold off
print(fig1,'template/newentropy','-dpdf')




    
