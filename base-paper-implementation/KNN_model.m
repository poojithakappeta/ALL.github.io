clc; clear; close all;

%% 1⃣ Load Selected Features
load('selected_features.mat');

%% 2⃣ Split Data (Train: 70%, Test: 30%)
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

%% 3⃣ Train KNN Model
KNNModel_ALLIDB2 = fitcknn(trainData_ALLIDB2, trainLabels_ALLIDB2, 'NumNeighbors', 5);
KNNModel_Subtypes = fitcknn(trainData_Subtypes, trainLabels_Subtypes, 'NumNeighbors', 5);

%% 4⃣ Predict Using KNN Model
pred_ALLIDB2 = predict(KNNModel_ALLIDB2, testData_ALLIDB2);
pred_Subtypes = predict(KNNModel_Subtypes, testData_Subtypes);

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

%% 5⃣ Compute Accuracy
KNN_accuracy_ALLIDB2 = sum(pred_ALLIDB2 == testLabels_ALLIDB2) / length(testLabels_ALLIDB2) * 100;
KNN_accuracy_Subtypes = sum(pred_Subtypes == testLabels_Subtypes) / length(testLabels_Subtypes) * 100;

disp(['KNN Accuracy (ALL vs. Normal): ', num2str(KNN_accuracy_ALLIDB2), '%']);
disp(['KNN Accuracy (Subtypes): ', num2str(KNN_accuracy_Subtypes), '%']);

%% 6⃣ Compute Confusion Matrices
KNN_confMat_ALLIDB2 = confusionmat(testLabels_ALLIDB2, pred_ALLIDB2);
KNN_confMat_Subtypes = confusionmat(testLabels_Subtypes, pred_Subtypes);

disp('Confusion Matrix for ALL vs. Normal:');
disp(KNN_confMat_ALLIDB2);
disp('Confusion Matrix for Subtypes:');
disp(KNN_confMat_Subtypes);

%% 7⃣ Compute Performance Metrics
% Precision, Recall, and F1-score for ALL_IDB2
KNN_precision_ALLIDB2 = diag(KNN_confMat_ALLIDB2) ./ sum(KNN_confMat_ALLIDB2, 2);
KNN_recall_ALLIDB2 = diag(KNN_confMat_ALLIDB2) ./ sum(KNN_confMat_ALLIDB2, 1)';
KNN_f1_score_ALLIDB2 = 2 * (KNN_precision_ALLIDB2 .* KNN_recall_ALLIDB2) ./ (KNN_precision_ALLIDB2 + KNN_recall_ALLIDB2);

disp(['KNN Precision (ALL vs. Normal): ', num2str(nanmean(KNN_precision_ALLIDB2))]);
disp(['KNN Recall (ALL vs. Normal): ', num2str(nanmean(KNN_recall_ALLIDB2))]);
disp(['KNN F1-score (ALL vs. Normal): ', num2str(nanmean(KNN_f1_score_ALLIDB2))]);

% Precision, Recall, and F1-score for Subtypes
KNN_precision_Subtypes = diag(KNN_confMat_Subtypes) ./ sum(KNN_confMat_Subtypes, 2);
KNN_recall_Subtypes = diag(KNN_confMat_Subtypes) ./ sum(KNN_confMat_Subtypes, 1)';
KNN_f1_score_Subtypes = 2 * (KNN_precision_Subtypes .* KNN_recall_Subtypes) ./ (KNN_precision_Subtypes + KNN_recall_Subtypes);

disp(['KNN Precision (Subtypes): ', num2str(nanmean(KNN_precision_Subtypes))]);
disp(['KNN Recall (Subtypes): ', num2str(nanmean(KNN_recall_Subtypes))]);
disp(['KNN F1-score (Subtypes): ', num2str(nanmean(KNN_f1_score_Subtypes))]);

%% 8⃣ Save KNN Model
save('knn_model.mat', 'KNNModel_ALLIDB2', 'KNNModel_Subtypes');

disp('✅ KNN Models Saved as knn_model.mat');

%% 9⃣ Plot Confusion Matrices
figure;
subplot(1,2,1);
confusionchart(KNN_confMat_ALLIDB2);
title('Confusion Matrix - ALL vs. Normal');

subplot(1,2,2);
confusionchart(KNN_confMat_Subtypes);
title('Confusion Matrix - ALL Subtypes');

disp('✅ KNN Model Evaluation Completed');
save('knn_results.mat', 'KNN_accuracy_ALLIDB2', 'KNN_accuracy_Subtypes', ...
     'KNN_precision_ALLIDB2', 'KNN_precision_Subtypes', ...
     'KNN_recall_ALLIDB2', 'KNN_recall_Subtypes', ...
     'KNN_f1_score_ALLIDB2', 'KNN_f1_score_Subtypes');
disp('✅ KNN results saved.');
