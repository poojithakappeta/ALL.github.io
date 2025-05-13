clc; clear; close all;

% Load handcrafted and wavelet features
load('wavelet_features.mat');
load('handcrafted_features.mat');

disp('✅ Loaded Features for Fusion');

% Check size of stats_ALLIDB2 fields
numSamples = size(LL_train_ALLIDB2, 1);

% Ensure LBP has correct size by replicating it across all samples
lbp_ALLIDB2_fixed = repmat(lbp_ALLIDB2, numSamples, 1);
lbp_Subtypes_fixed = repmat(lbp_Subtypes, size(LL_train_Subtypes, 1), 1);

% Ensure statistical features have correct shape
handcrafted_ALLIDB2 = [stats_ALLIDB2.mean, stats_ALLIDB2.variance, ...
                        stats_ALLIDB2.skewness, stats_ALLIDB2.entropy, ...
                        lbp_ALLIDB2_fixed];

handcrafted_Subtypes = [stats_Subtypes.mean, stats_Subtypes.variance, ...
                         stats_Subtypes.skewness, stats_Subtypes.entropy, ...
                         lbp_Subtypes_fixed];

% Concatenate deep and handcrafted features
fused_ALLIDB2 = [LL_train_ALLIDB2, handcrafted_ALLIDB2];
fused_Subtypes = [LL_train_Subtypes, handcrafted_Subtypes];

disp('✅ Feature Fusion Completed!');

% Save the fused feature set
save('fused_features.mat', 'fused_ALLIDB2', 'fused_Subtypes');

disp('✅ Fused Features Saved as fused_features.mat');
