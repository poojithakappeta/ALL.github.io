clc; clear; close all;

%% 1️⃣ Load Selected Features
load('selected_features.mat');
disp('Loaded Selected Features for Classification');

%% 2️⃣ Split Data (Train: 70%, Test: 30%)
cv_ALLIDB2 = cvpartition(size(selectedFeatures_ALLIDB2, 1), 'HoldOut', 0.3);
trainIdx_ALLIDB2 = training(cv_ALLIDB2);
testIdx_ALLIDB2 = test(cv_ALLIDB2);
trainData_ALLIDB2 = selectedFeatures_ALLIDB2(trainIdx_ALLIDB2, :);
testData_ALLIDB2 = selectedFeatures_ALLIDB2(testIdx_ALLIDB2, :);
trainLabels_ALLIDB2 = labels_ALLIDB2_train(trainIdx_ALLIDB2);
testLabels_ALLIDB2 = labels_ALLIDB2_train(testIdx_ALLIDB2);

cv_Subtypes = cvpartition(size(selectedFeatures_Subtypes, 1), 'HoldOut', 0.3);
trainIdx_Subtypes = training(cv_Subtypes);
testIdx_Subtypes = test(cv_Subtypes);
trainData_Subtypes = selectedFeatures_Subtypes(trainIdx_Subtypes, :);
testData_Subtypes = selectedFeatures_Subtypes(testIdx_Subtypes, :);
trainLabels_Subtypes = labels_Subtypes_train(trainIdx_Subtypes);
testLabels_Subtypes = labels_Subtypes_train(testIdx_Subtypes);

disp('Data Split into Training & Testing Sets');

%% 3️⃣ Train SVM Classifier
SVMModel_ALLIDB2 = fitcsvm(trainData_ALLIDB2, trainLabels_ALLIDB2, 'KernelFunction', 'linear');
SVMModel_Subtypes = fitcecoc(trainData_Subtypes, trainLabels_Subtypes); % Multi-class SVM
disp('SVM Classifier Trained');

%% 4️⃣ Save Trained Models
save('svm_model.mat', 'SVMModel_ALLIDB2', 'SVMModel_Subtypes');
disp('✅ SVM Models Saved as svm_model.mat');

%% 5️⃣ Predict Labels Using SVM
pred_ALLIDB2_SVM = predict(SVMModel_ALLIDB2, testData_ALLIDB2);
pred_Subtypes_SVM = predict(SVMModel_Subtypes, testData_Subtypes);

disp('Predictions Made on Test Data');

%% 6️⃣ Compute Confusion Matrix
confMat_ALLIDB2_SVM = confusionmat(testLabels_ALLIDB2, pred_ALLIDB2_SVM);
confMat_Subtypes_SVM = confusionmat(testLabels_Subtypes, pred_Subtypes_SVM);

disp('Confusion Matrix for ALL vs. Normal (SVM):');
disp(confMat_ALLIDB2_SVM);
disp('Confusion Matrix for Subtypes (SVM):');
disp(confMat_Subtypes_SVM);

%% 7️⃣ Compute SVM Accuracy, Precision, Recall, F1-score
SVM_accuracy_ALLIDB2 = sum(diag(confMat_ALLIDB2_SVM)) / sum(confMat_ALLIDB2_SVM(:)) * 100;
SVM_precision_ALLIDB2 = diag(confMat_ALLIDB2_SVM) ./ sum(confMat_ALLIDB2_SVM, 2);
SVM_recall_ALLIDB2 = diag(confMat_ALLIDB2_SVM) ./ sum(confMat_ALLIDB2_SVM, 1)';
SVM_f1_score_ALLIDB2 = 2 * (SVM_precision_ALLIDB2 .* SVM_recall_ALLIDB2) ./ (SVM_precision_ALLIDB2 + SVM_recall_ALLIDB2);

SVM_accuracy_Subtypes = sum(diag(confMat_Subtypes_SVM)) / sum(confMat_Subtypes_SVM(:)) * 100;
SVM_precision_Subtypes = diag(confMat_Subtypes_SVM) ./ sum(confMat_Subtypes_SVM, 2);
SVM_recall_Subtypes = diag(confMat_Subtypes_SVM) ./ sum(confMat_Subtypes_SVM, 1)';
SVM_f1_score_Subtypes = 2 * (SVM_precision_Subtypes .* SVM_recall_Subtypes) ./ (SVM_precision_Subtypes + SVM_recall_Subtypes);

disp(['SVM Accuracy (ALL vs. Normal): ', num2str(SVM_accuracy_ALLIDB2), '%']);
disp(['SVM Precision (ALL vs. Normal): ', num2str(mean(SVM_precision_ALLIDB2))]);
disp(['SVM Recall (ALL vs. Normal): ', num2str(mean(SVM_recall_ALLIDB2))]);
disp(['SVM F1-score (ALL vs. Normal): ', num2str(mean(SVM_f1_score_ALLIDB2))]);

disp(['SVM Accuracy (Subtypes): ', num2str(SVM_accuracy_Subtypes), '%']);
disp(['SVM Precision (Subtypes): ', num2str(mean(SVM_precision_Subtypes))]);
disp(['SVM Recall (Subtypes): ', num2str(mean(SVM_recall_Subtypes))]);
disp(['SVM F1-score (Subtypes): ', num2str(mean(SVM_f1_score_Subtypes))]);

%% 8️⃣ Plot Confusion Matrices
figure;
subplot(1,2,1);
confusionchart(confMat_ALLIDB2_SVM);
title('SVM Confusion Matrix - ALL vs. Normal');

subplot(1,2,2);
confusionchart(confMat_Subtypes_SVM);
title('SVM Confusion Matrix - ALL Subtypes');

disp('✅ SVM Model Training & Evaluation Completed');
save('svm_results.mat', 'SVM_accuracy_ALLIDB2', 'SVM_accuracy_Subtypes', ...
     'SVM_precision_ALLIDB2', 'SVM_precision_Subtypes', ...
     'SVM_recall_ALLIDB2', 'SVM_recall_Subtypes', ...
     'SVM_f1_score_ALLIDB2', 'SVM_f1_score_Subtypes');
disp('✅ SVM results saved.');
