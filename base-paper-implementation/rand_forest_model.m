clc; clear; close all;

%% 1️⃣ Load Selected Features
load('selected_features.mat');

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

%% 3️⃣ Train Random Forest Model
RFModel_ALLIDB2 = TreeBagger(100, trainData_ALLIDB2, trainLabels_ALLIDB2, 'OOBPrediction', 'on');
RFModel_Subtypes = TreeBagger(100, trainData_Subtypes, trainLabels_Subtypes, 'OOBPrediction', 'on');

%% 4️⃣ Predict Using RF Model
pred_ALLIDB2 = predict(RFModel_ALLIDB2, testData_ALLIDB2);
pred_Subtypes = predict(RFModel_Subtypes, testData_Subtypes);

% ✅ Ensure predictions match testLabels data type
if isnumeric(testLabels_ALLIDB2)
    pred_ALLIDB2 = str2double(pred_ALLIDB2);
else
    pred_ALLIDB2 = categorical(pred_ALLIDB2);
end

if isnumeric(testLabels_Subtypes)
    pred_Subtypes = str2double(pred_Subtypes);
else
    pred_Subtypes = categorical(pred_Subtypes);
end

%% 5️⃣ Compute Accuracy
RF_accuracy_ALLIDB2 = sum(pred_ALLIDB2 == testLabels_ALLIDB2) / length(testLabels_ALLIDB2) * 100;
RF_accuracy_Subtypes = sum(pred_Subtypes == testLabels_Subtypes) / length(testLabels_Subtypes) * 100;

disp(['Random Forest Accuracy (ALL vs. Normal): ', num2str(RF_accuracy_ALLIDB2), '%']);
disp(['Random Forest Accuracy (Subtypes): ', num2str(RF_accuracy_Subtypes), '%']);

%% 6️⃣ Compute Confusion Matrices
confMat_ALLIDB2 = confusionmat(testLabels_ALLIDB2, pred_ALLIDB2);
confMat_Subtypes = confusionmat(testLabels_Subtypes, pred_Subtypes);

disp('Confusion Matrix for ALL vs. Normal:');
disp(confMat_ALLIDB2);
disp('Confusion Matrix for Subtypes:');
disp(confMat_Subtypes);

%% 7️⃣ Compute Performance Metrics
% Precision, Recall, and F1-score for ALL_IDB2
RF_precision_ALLIDB2 = diag(confMat_ALLIDB2) ./ sum(confMat_ALLIDB2, 2);
RF_recall_ALLIDB2 = diag(confMat_ALLIDB2) ./ sum(confMat_ALLIDB2, 1)';
RF_f1_score_ALLIDB2 = 2 * (RF_precision_ALLIDB2 .* RF_recall_ALLIDB2) ./ (RF_precision_ALLIDB2 + RF_recall_ALLIDB2);

disp(['Precision (ALL vs. Normal): ', num2str(nanmean(RF_precision_ALLIDB2))]);
disp(['Recall (ALL vs. Normal): ', num2str(nanmean(RF_recall_ALLIDB2))]);
disp(['F1-score (ALL vs. Normal): ', num2str(nanmean(RF_f1_score_ALLIDB2))]);

% Precision, Recall, and F1-score for Subtypes
RF_precision_Subtypes = diag(confMat_Subtypes) ./ sum(confMat_Subtypes, 2);
RF_recall_Subtypes = diag(confMat_Subtypes) ./ sum(confMat_Subtypes, 1)';
RF_f1_score_Subtypes = 2 * (RF_precision_Subtypes .* RF_recall_Subtypes) ./ (RF_precision_Subtypes + RF_recall_Subtypes);

disp(['Precision (Subtypes): ', num2str(nanmean(RF_precision_Subtypes))]);
disp(['Recall (Subtypes): ', num2str(nanmean(RF_recall_Subtypes))]);
disp(['F1-score (Subtypes): ', num2str(nanmean(RF_f1_score_Subtypes))]);

%% 8️⃣ Save Random Forest Model
save('rf_model.mat', 'RFModel_ALLIDB2', 'RFModel_Subtypes');

disp('✅ Random Forest Models Saved as rf_model.mat');

%% 9️⃣ Plot Confusion Matrices
figure;
subplot(1,2,1);
confusionchart(confMat_ALLIDB2);
title('Confusion Matrix - ALL vs. Normal');

subplot(1,2,2);
confusionchart(confMat_Subtypes);
title('Confusion Matrix - ALL Subtypes');

disp('✅ Random Forest Model Evaluation Completed');
save('rf_results.mat', 'RF_accuracy_ALLIDB2', 'RF_accuracy_Subtypes', ...
     'RF_precision_ALLIDB2', 'RF_precision_Subtypes', ...
     'RF_recall_ALLIDB2', 'RF_recall_Subtypes', ...
     'RF_f1_score_ALLIDB2', 'RF_f1_score_Subtypes');
disp('✅ RF results saved.');
