
%Denoising and feature compression

clc; clear; close all;

% Load extracted deep features
load('deep_features.mat');

% Display confirmation
disp('✅ Deep Features Loaded for Wavelet Pooling');

% Define Wavelet Type
waveletType = 'haar'; % Haar wavelet used for pooling

% Calculate new feature size after DWT (HALF of original feature dimension)
newFeatureSize_ALLIDB2 = floor(size(trainFeatures_ALLIDB2, 2) / 2);
newFeatureSize_Subtypes = floor(size(trainFeatures_Subtypes, 2) / 2);

% Initialize matrices with reduced feature size (same rows, half columns)
LL_train_ALLIDB2 = zeros(size(trainFeatures_ALLIDB2, 1), newFeatureSize_ALLIDB2);
LL_test_ALLIDB2 = zeros(size(testFeatures_ALLIDB2, 1), newFeatureSize_ALLIDB2);

LL_train_Subtypes = zeros(size(trainFeatures_Subtypes, 1), newFeatureSize_Subtypes);
LL_test_Subtypes = zeros(size(testFeatures_Subtypes, 1), newFeatureSize_Subtypes);

% Apply Wavelet Transform to Training Features (ALL vs. Normal)
for i = 1:size(trainFeatures_ALLIDB2, 1) % Loop over rows (samples)
    [LL, ~] = dwt(trainFeatures_ALLIDB2(i, :), waveletType); % Apply DWT row-wise
    LL_train_ALLIDB2(i, :) = LL(1:newFeatureSize_ALLIDB2); % Assign only LL coefficients
end

for i = 1:size(testFeatures_ALLIDB2, 1)
    [LL, ~] = dwt(testFeatures_ALLIDB2(i, :), waveletType);
    LL_test_ALLIDB2(i, :) = LL(1:newFeatureSize_ALLIDB2);
end

% Apply Wavelet Transform to Training Features (Subtypes)
for i = 1:size(trainFeatures_Subtypes, 1)
    [LL, ~] = dwt(trainFeatures_Subtypes(i, :), waveletType);
    LL_train_Subtypes(i, :) = LL(1:newFeatureSize_Subtypes);
end

for i = 1:size(testFeatures_Subtypes, 1)
    [LL, ~] = dwt(testFeatures_Subtypes(i, :), waveletType);
    LL_test_Subtypes(i, :) = LL(1:newFeatureSize_Subtypes);
end

disp('✅ Wavelet Pooling Applied Successfully!');

% Save wavelet features for the next step
save('wavelet_features.mat', 'LL_train_ALLIDB2', 'LL_test_ALLIDB2', ...
     'LL_train_Subtypes', 'LL_test_Subtypes');

disp('✅ Wavelet Features Saved as wavelet_features.mat');