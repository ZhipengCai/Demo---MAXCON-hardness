%plot final outliers from ASTAR

clear;
close all;

%directory to the dataset
dirDataset = '/media/zhipeng/Data/Data/KITTI/odometry dataset/image sequences/dataset/sequences/';
sequence = '00/image_2/'; %sequence to test from 00-21 (each number has two folders "image_2" and "image_3")
corrFolder = [dirDataset sequence 'corr/']; %folder of the input correspondences

load('kernel_1To1000_32bit_v2.mat');%read experimemt result

dataToPlot = 1; %number of images to test, maximum 4540
idxOutlier = violationSet{dataToPlot};

%read image pair
disp(['reading data ' num2str(dataToPlot)]);
%read sift correspondences
dataIdx = sprintf('%06.f', dataToPlot);
fileName = [corrFolder dataIdx '.mat'];
load(fileName);
data = linearData{1};

figure;
idxInls = 1:size(data.matches.X1,2);
idxInls(idxOutlier) = [];
plot_match(data.matches, [data.matches.X1; data.matches.X2],  idxInls, 1, 1000);