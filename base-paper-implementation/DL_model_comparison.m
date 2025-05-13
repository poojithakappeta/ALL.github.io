clc; clear; close all;

%% 1️⃣ Load Test Data
load('selected_features.mat');  % Load the test data and labels
numSamples = size(selectedFeatures_ALLIDB2, 1);

%% 2️⃣ Preprocess Data (Reshape and Resize)
imageSize = [10, 10]; % 100 features → 10x10 "grayscale image"
trainDataReshaped = reshape(selectedFeatures_ALLIDB2', [imageSize(1), imageSize(2), 1, numSamples]);
trainDataRGB = repmat(trainDataReshaped, [1, 1, 3, 1]);

resizedData = zeros(224, 224, 3, numSamples, 'single'); % Preallocate
for i = 1:numSamples
    resizedData(:, :, :, i) = imresize(trainDataRGB(:, :, :, i), [224, 224]);
end

trainLabelsCat = categorical(labels_ALLIDB2_train);

%% 3️⃣ Split Data (Train: 70%, Test: 30%)
cv = cvpartition(numSamples, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

testData = resizedData(:, :, :, testIdx);
testLabels = trainLabelsCat(testIdx);

disp('✅ Test Data Loaded and Preprocessed');

%% 4️⃣ Load Trained Models
disp('🔄 Loading Trained Models...');
load('mobilenet_model.mat', 'netMobileNet');
load('resnet50_model.mat', 'netResNet50');
disp('✅ Models Loaded');

%% 5️⃣ Evaluate MobileNet
disp('📊 Evaluating MobileNet...');
predictions_MobileNet = classify(netMobileNet, testData);
confMat_MobileNet = confusionmat(testLabels, predictions_MobileNet);
accuracy_MobileNet = sum(diag(confMat_MobileNet)) / sum(confMat_MobileNet(:)) * 100;
disp(['✅ MobileNet Accuracy: ', num2str(accuracy_MobileNet), '%']);

%% 6️⃣ Evaluate ResNet-50
disp('📊 Evaluating ResNet-50...');
predictions_ResNet50 = classify(netResNet50, testData);
confMat_ResNet50 = confusionmat(testLabels, predictions_ResNet50);
accuracy_ResNet50 = sum(diag(confMat_ResNet50)) / sum(confMat_ResNet50(:)) * 100;
disp(['✅ ResNet-50 Accuracy: ', num2str(accuracy_ResNet50), '%']);

%% 7️⃣ Plot Bar Graph with Custom Colors
figure;
barGraph = bar([accuracy_MobileNet, accuracy_ResNet50], 'FaceColor', 'flat');

% Custom Colors (Modify as needed)
barGraph.CData(1, :) = [0 0.4470 0.7410];  % Blue for MobileNet
barGraph.CData(2, :) = [0.8500 0.3250 0.0980];  % Orange for ResNet-50

set(gca, 'xticklabel', {'MobileNet', 'ResNet-50'});
ylabel('Accuracy (%)');
title('Accuracy Comparison: MobileNet vs. ResNet-50');
grid on;

disp('✅ Accuracy Comparison Bar Graph Generated with Custom Colors');
