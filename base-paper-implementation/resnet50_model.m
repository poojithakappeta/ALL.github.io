clc; clear; close all;

%% 1️⃣ Load Selected Features
load('selected_features.mat');

numSamples = size(selectedFeatures_ALLIDB2, 1);  % 208 samples
numFeatures = size(selectedFeatures_ALLIDB2, 2); % 100 features

disp(['✅ Loaded Data: ', num2str(numSamples), ' samples, ', num2str(numFeatures), ' features each']);

%% 2️⃣ Reshape Features into Image-Compatible Format
imageSize = [10, 10]; % 100 features → 10x10 "grayscale image"

% Reshape features into 10x10 grayscale images
trainDataReshaped = reshape(selectedFeatures_ALLIDB2', [imageSize(1), imageSize(2), 1, numSamples]);

% Convert grayscale to 3-channel images (Replicate across RGB channels)
trainDataRGB = repmat(trainDataReshaped, [1, 1, 3, 1]);

% Resize images to 224x224 for ResNet-50
resizedData = zeros(224, 224, 3, numSamples, 'single'); % Preallocate
for i = 1:numSamples
    resizedData(:, :, :, i) = imresize(trainDataRGB(:, :, :, i), [224, 224]);
end

disp('✅ Data Reshaped & Resized for ResNet-50');

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

%% 5️⃣ Load Pretrained ResNet-50 & Modify Input Layer
net = resnet50;
lgraph = layerGraph(net);

% Get the original input layer name
inputLayerName = net.Layers(1).Name;
disp(['✅ Input Layer Name: ', inputLayerName]);

% Replace input layer to match new image size
newInputLayer = imageInputLayer([224 224 3], 'Name', inputLayerName);
lgraph = replaceLayer(lgraph, inputLayerName, newInputLayer);

%% 6️⃣ Modify Fully Connected Layer for Classification
numClasses = numel(unique(trainLabels)); % Get number of classes
newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, 'fc1000', newFcLayer);

% Modify Softmax and Classification Layers
newSoftmaxLayer = softmaxLayer('Name', 'softmax');
newClassificationLayer = classificationLayer('Name', 'output');

lgraph = replaceLayer(lgraph, 'fc1000_softmax', newSoftmaxLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassificationLayer);

disp('✅ ResNet-50 Layers Modified');

%% 7️⃣ Set Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 10, ...  % No increase in epochs as per request
    'InitialLearnRate', 0.001, ...
    'ValidationData', {testData, testLabels}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% 8️⃣ Train ResNet-50
disp('🚀 Training ResNet-50...');
netResNet50 = trainNetwork(trainData, trainLabels, lgraph, options);
disp('✅ ResNet-50 Model Training Completed');

%% 9️⃣ Save the Model
save('resnet50_model.mat', 'netResNet50');
disp('✅ ResNet-50 Model Saved as resnet50_model.mat');

%% 🔟 Model Evaluation
predictions = classify(netResNet50, testData);

% Compute Confusion Matrix
confMat_ResNet50 = confusionmat(testLabels, predictions);
disp('Confusion Matrix for ResNet-50:');
disp(confMat_ResNet50);

% Compute Accuracy, Precision, Recall, F1-score
accuracy_ResNet50 = sum(diag(confMat_ResNet50)) / sum(confMat_ResNet50(:)) * 100;
precision_ResNet50 = diag(confMat_ResNet50) ./ sum(confMat_ResNet50, 2);
recall_ResNet50 = diag(confMat_ResNet50) ./ sum(confMat_ResNet50, 1)';
f1_score_ResNet50 = 2 * (precision_ResNet50 .* recall_ResNet50) ./ (precision_ResNet50 + recall_ResNet50);

disp(['ResNet-50 Accuracy: ', num2str(accuracy_ResNet50), '%']);
disp(['Precision (ResNet-50): ', num2str(mean(precision_ResNet50), '%.2f')]);
disp(['Recall (ResNet-50): ', num2str(mean(recall_ResNet50), '%.2f')]);
disp(['F1-score (ResNet-50): ', num2str(mean(f1_score_ResNet50), '%.2f')]);

%% 🔟 Plot Confusion Matrix
figure;
confusionchart(confMat_ResNet50);
title('Confusion Matrix - ResNet-50');

disp('✅ ResNet-50 Model Evaluation Completed');
