clc; clear; close all;

%% 1️⃣ Load Fused Features
load('fused_features.mat');

disp('✅ Loaded Fused Features');

%% 2️⃣ Load Labels from Dataset
load('dataset_preparation.mat'); 

% ✅ Extract Labels from the Saved Variables
labels_ALLIDB2_train = labels_ALLIDB2_train;  % Train labels for ALL vs. Normal
labels_Subtypes_train = labels_Subtypes_train;  % Train labels for Subtypes

disp('✅ Labels Extracted Successfully');

%% 3️⃣ Apply ReliefF Feature Selection
selectedFeatureCount = 100; % Choose top 100 most relevant features (adjust as needed)

% Feature Selection for ALL vs. Normal
idx_ALLIDB2 = relieff(fused_ALLIDB2, labels_ALLIDB2_train, 10);
selectedFeatures_ALLIDB2 = fused_ALLIDB2(:, idx_ALLIDB2(1:selectedFeatureCount));

% Feature Selection for Subtypes
idx_Subtypes = relieff(fused_Subtypes, labels_Subtypes_train, 10);
selectedFeatures_Subtypes = fused_Subtypes(:, idx_Subtypes(1:selectedFeatureCount));

disp('✅ ReliefF Feature Selection Completed');

%% 4️⃣ Save Selected Features
save('selected_features.mat', 'selectedFeatures_ALLIDB2', 'selectedFeatures_Subtypes', 'labels_ALLIDB2_train', 'labels_Subtypes_train');

disp('✅ Selected Features Saved as selected_features.mat');
