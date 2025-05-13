clc; clear; close all;

% Load dataset variables from dataset_preparation.mat
load('dataset_preparation.mat');  

% Load Pre-Trained CNN (ResNet-18)
net = resnet18;

% Extract Features from Last Layer (Pool5) Before Fully Connected Layers
layerName = 'pool5';

% Extract Deep Features from CNN for ALL vs. Normal Dataset
trainFeatures_ALLIDB2 = activations(net, augmentedTrain_ALLIDB2, layerName, 'OutputAs', 'rows');
testFeatures_ALLIDB2 = activations(net, augmentedTest_ALLIDB2, layerName, 'OutputAs', 'rows');

% Extract Deep Features for Subtype Classification (Benign, Pre, Early, Pro)
trainFeatures_Subtypes = activations(net, augmentedTrain_Subtypes, layerName, 'OutputAs', 'rows');
testFeatures_Subtypes = activations(net, augmentedTest_Subtypes, layerName, 'OutputAs', 'rows');

disp('✅ Deep Features Extracted Successfully!');

% Save extracted features for next steps
save('deep_features.mat', 'trainFeatures_ALLIDB2', 'testFeatures_ALLIDB2', ...
     'trainFeatures_Subtypes', 'testFeatures_Subtypes');

disp('✅ Deep features saved as deep_features.mat');