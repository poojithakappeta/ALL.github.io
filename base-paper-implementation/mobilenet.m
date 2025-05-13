clc; clear; close all;

%% 1️⃣ Load Selected Features
load('selected_features.mat');

numSamples = size(selectedFeatures_ALLIDB2, 1);
numFeatures = size(selectedFeatures_ALLIDB2, 2);

disp(['✅ Loaded Data: ', num2str(numSamples), ' samples, ', num2str(numFeatures), ' features each']);

%% 2️⃣ Reshape Features into Image-Compatible Format
imageSize = [10, 10]; 
trainDataReshaped = reshape(selectedFeatures_ALLIDB2', [imageSize(1), imageSize(2), 1, numSamples]);
trainDataRGB = repmat(trainDataReshaped, [1, 1, 3, 1]);

resizedData = zeros(224, 224, 3, numSamples, 'single');
for i = 1:numSamples
    resizedData(:, :, :, i) = imresize(trainDataRGB(:, :, :, i), [224, 224]);
end

disp('✅ Data Reshaped & Resized for MobileNet');

%% 3️⃣ Convert Labels into Categorical Format
trainLabelsCat = categorical(labels_ALLIDB2_train);

%% 4️⃣ Split Data (Train: 70%, Test: 30%)
cv = cvpartition(numSamples, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

trainData = resizedData(:, :, :, trainIdx);
testData = resizedData(:, :, :, testIdx);
trainLabels = trainLabelsCat(trainIdx);
testLabels = trainLabelsCat(testIdx);

disp('✅ Data Split into Train & Test');

%% 5️⃣ Load Pretrained MobileNet & Modify Input Layer
net = mobilenetv2;
lgraph = layerGraph(net);

inputLayerName = net.Layers(1).Name;
newInputLayer = imageInputLayer([224 224 3], 'Name', inputLayerName);
lgraph = replaceLayer(lgraph, inputLayerName, newInputLayer);

%% 6️⃣ Modify Fully Connected Layer for Classification
numClasses = numel(unique(trainLabels));
newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'Logits', newFcLayer);

newSoftmaxLayer = softmaxLayer('Name', 'softmax');
newClassificationLayer = classificationLayer('Name', 'output');

lgraph = replaceLayer(lgraph, 'Logits_softmax', newSoftmaxLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_Logits', newClassificationLayer);

disp('✅ MobileNet Layers Modified');

%% 7️⃣ Set Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {testData, testLabels}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% 8️⃣ Train MobileNet
disp('🚀 Training MobileNet...');
netMobileNet = trainNetwork(trainData, trainLabels, lgraph, options);
disp('✅ MobileNet Model Training Completed');

%% 9️⃣ Save the Model
save('mobilenet_model.mat', 'netMobileNet');
disp('✅ MobileNet Model Saved');

%% 🔟 Model Evaluation
predictions = classify(netMobileNet, testData);

confMat_MobileNet = confusionmat(testLabels, predictions);
disp('Confusion Matrix for MobileNet:');
disp(confMat_MobileNet);

accuracy_MobileNet = sum(diag(confMat_MobileNet)) / sum(confMat_MobileNet(:)) * 100;
disp(['MobileNet Accuracy: ', num2str(accuracy_MobileNet), '%']);

%% 🔟 Plot Confusion Matrix
figure;
confusionchart(confMat_MobileNet);
title('Confusion Matrix - MobileNet');

disp('✅ MobileNet Model Evaluation Completed');
