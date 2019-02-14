%code for experiment on real (KITTI) dataset
%test how many image pairs can be processed in N*d steps of A*

clear;
close all;

%directory to the dataset
dirDataset = '/media/zhipeng/Data/Data/KITTI/odometry dataset/image sequences/dataset/sequences/';
sequence = '00/image_2/'; %sequence to test from 00-21 (each number has two folders "image_2" and "image_3")
corrFolder = [dirDataset sequence 'corr_frameGap5/']; %folder of the input correspondences
numImgForTest = 200; %number of images to test, maximum 4540

fileOut = 'kernel_1To200_32bit_v2_frameGap5.mat';
th = 0.04; %inlier threshold

finishedOrNot = zeros(numImgForTest,1);
runtime = zeros(numImgForTest,1);


for i = 1:numImgForTest
    disp(['reading data ', num2str(i) '...']);
    %read sift correspondences
    dataIdx = sprintf('%06.f', i);
    fileName = [corrFolder dataIdx '.mat'];
    load(fileName);
    data = linearData{1};
    X{i} = data.x;
    Y{i} = data.y;
    
%     figure;
%     plot_match(data.matches, [data.matches.X1; data.matches.X2],  1:size(data.matches.X1,2), 0, 1000);
%     pause;
end
% 
delete(gcp('nocreate'));
parpool(4);
spmd
    warning('off','all');
end


%testing all the consecutive pairs
parfor i = 1:numImgForTest
%for i = 169:169
    disp(['processing data ' num2str(i) '...']);
    
    %input data
    x = X{i};
    y = Y{i};
    %infinitestimal perturbation for removing degeneracy
    x = x+1e-10*(rand(size(x))-0.5);
    y = y+1e-10*(rand(size(y))-0.5);

  
%     x1 = data.x1;
%     x2 = data.x2;
% 
%     [x,y] = linearize_fund(x1,x2);
    
    N = numel(y); %number of correspondences
    d = 8;

    inputSize = 32*N*d;
    
    %run A*-tree search in N*d steps
    theta0 = randn(d,1);
    
    tic;
    [UNN(i), violationSet{i}] = maxconASTAR_kernel_zhipeng(x,y,theta0,th,inputSize);
    vSize(i) = length(violationSet{i});
    runtime(i,1) = toc;
    %[pk,wk,vk,nnum,xnum,bk,UNN] = maxconASTAR(x,y, theta0, th)
     

     if(UNN(i)<inputSize)
        finishedOrNot(i,1) = 1;
        disp(['UNN(' num2str(i) ') = ' num2str(UNN(i)) '; inputSize = ' num2str(inputSize)]);
     end
    
%      %test
%      if(vSize(i)>=3)
%          %read sift correspondences
%          dataIdx = sprintf('%06.f', i);
%          fileName = [corrFolder dataIdx '.mat'];
%          load(fileName);
%          data = linearData{1};
%          
%          close all;
%          figure;
%          Inls = 1:size(data.matches.X1,2);
%          Inls(violationSet{i}) = [];
%          plot_match(data.matches, [data.matches.X1; data.matches.X2],  Inls, 1, 1000);
%          pause;
%      end
end

NOFinished = sum(finishedOrNot > 0)
NONotFinished = sum(finishedOrNot == 0)

aveOutlier = mean(vSize(finishedOrNot>0))
maxOutlier = max(vSize(finishedOrNot>0))
aveLevel = mean(vSize(finishedOrNot == 0))
maxLevel = max(vSize(finishedOrNot==0))

save(fileOut);
