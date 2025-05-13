clc; clear; close all;

%% 1️⃣ Load ALL-IDB2 Dataset (Extract Labels from Filenames)
datasetPath = fullfile('/MATLAB Drive/Dataset_ALL'); % Update path if needed
allIDB2Path = fullfile(datasetPath, 'ALL_IDB2', 'img'); % Ensure correct path

% ✅ Load ALL-IDB2 dataset
imds_ALLIDB2 = imageDatastore(allIDB2Path, 'FileExtensions', {'.tif', '.jpg', '.png'});

% ✅ Extract Labels from Filenames (`ImXXX_Y.jpg → Y=0 or 1`)
numImages = numel(imds_ALLIDB2.Files);
labels_ALLIDB2 = zeros(numImages, 1); % Preallocate

for i = 1:numImages
    [~, filename, ~] = fileparts(imds_ALLIDB2.Files{i});
    labels_ALLIDB2(i) = str2double(filename(end)); % Extract last character (0 or 1)
end

% ✅ Convert labels to categorica+l (0 = Healthy, 1 = ALL)
labels_ALLIDB2 = categorical(labels_ALLIDB2, [0 1], {'Healthy', 'ALL'});

% Print dataset details
disp('✅ ALL-IDB2 Dataset Loaded!');
disp(['Total images: ', num2str(numImages)]);
disp('Class distribution:');
tabulate(labels_ALLIDB2);

%% 2️⃣ Load Subtype Dataset (Folder-Based Labels)
subtypePath = fullfile(datasetPath, 'Original'); % Subtype images are stored here
imds_Subtypes = imageDatastore(subtypePath, 'IncludeSubfolders', true, ...
                               'LabelSource', 'foldernames', 'FileExtensions', {'.tif', '.jpg', '.png'});
labels_Subtypes = imds_Subtypes.Labels; % Extract subtype labels

% ✅ Balance dataset by taking an equal number of images per class
numClasses = categories(labels_Subtypes);
minImagesPerClass = min(countcats(labels_Subtypes));

balancedFiles = {};
balancedLabels = categorical([]);

for i = 1:numel(numClasses)
    classIdx = find(labels_Subtypes == numClasses{i});
    classFiles = imds_Subtypes.Files(classIdx);
    selectedIdx = randperm(numel(classFiles), minImagesPerClass);
    balancedFiles = [balancedFiles; classFiles(selectedIdx)];
    balancedLabels = [balancedLabels; repmat(categorical({numClasses{i}}), minImagesPerClass, 1)];
end

imds_Subtypes = imageDatastore(balancedFiles, 'Labels', balancedLabels);
labels_Subtypes = imds_Subtypes.Labels;

% Print subtype dataset details
disp('✅ Subtype Dataset Loaded!');
disp(['Total subtype images: ', num2str(numel(imds_Subtypes.Files))]);
disp('Subtype class distribution:');
tabulate(labels_Subtypes);

%% 3️⃣ Stratified Split (80% Training, 20% Testing)
[trainIdx, testIdx] = holdoutPartition(labels_ALLIDB2, 0.2);

train_ALLIDB2 = subset(imds_ALLIDB2, trainIdx);
test_ALLIDB2 = subset(imds_ALLIDB2, testIdx);
labels_ALLIDB2_train = labels_ALLIDB2(trainIdx);
labels_ALLIDB2_test = labels_ALLIDB2(testIdx);

% ✅ Split subtype dataset (Folder-based labels)
[train_Subtypes, test_Subtypes] = splitEachLabel(imds_Subtypes, 0.8, 'randomized');
labels_Subtypes_train = train_Subtypes.Labels;
labels_Subtypes_test = test_Subtypes.Labels;

disp('✅ Dataset Split Completed!');
disp(['Training images (ALL-IDB2): ', num2str(numel(train_ALLIDB2.Files))]);
disp(['Testing images (ALL-IDB2): ', num2str(numel(test_ALLIDB2.Files))]);
disp(['Training images (Subtypes): ', num2str(numel(train_Subtypes.Files))]);
disp(['Testing images (Subtypes): ', num2str(numel(test_Subtypes.Files))]);

%% 4️⃣ Preprocessing & Augmentation
imageSize = [224 224];

% Define augmentation options
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ... 
    'RandXReflection',true, ... 
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);

% ✅ Augment training datasets
augmentedTrain_ALLIDB2 = augmentedImageDatastore(imageSize, train_ALLIDB2, 'DataAugmentation', augmenter);
augmentedTrain_Subtypes = augmentedImageDatastore(imageSize, train_Subtypes, 'DataAugmentation', augmenter);

% ✅ Resize test datasets (No augmentation)
augmentedTest_ALLIDB2 = augmentedImageDatastore(imageSize, test_ALLIDB2);
augmentedTest_Subtypes = augmentedImageDatastore(imageSize, test_Subtypes);

disp('✅ Preprocessing & Augmentation Completed!');

%% 5️⃣ Save Variables for Feature Extraction
save('dataset_preparation.mat', 'augmentedTrain_ALLIDB2', 'augmentedTest_ALLIDB2', ...
     'augmentedTrain_Subtypes', 'augmentedTest_Subtypes', ...
     'labels_ALLIDB2_train', 'labels_ALLIDB2_test', ...
     'labels_Subtypes_train', 'labels_Subtypes_test');

disp('✅ Dataset variables & labels saved as dataset_preparation.mat');

%% Helper Function: Stratified Holdout Partition
function [trainIdx, testIdx] = holdoutPartition(labels, testRatio)
    uniqueClasses = unique(labels);
    trainIdx = [];
    testIdx = [];

    for i = 1:numel(uniqueClasses)
        classIdx = find(labels == uniqueClasses(i));
        numTest = round(testRatio * numel(classIdx));
        permIdx = randperm(numel(classIdx));
        
        testIdx = [testIdx; classIdx(permIdx(1:numTest))];
        trainIdx = [trainIdx; classIdx(permIdx(numTest+1:end))];
    end
end