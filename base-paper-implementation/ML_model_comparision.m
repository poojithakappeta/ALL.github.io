clc; clear; close all;

% Load saved results
load('svm_results.mat');
load('rf_results.mat');
load('knn_results.mat');

% Ensure variables exist
whos

% Model names
models = {'SVM', 'Random Forest', 'KNN'};

% Accuracy Comparison
accuracy_ALLIDB2 = [SVM_accuracy_ALLIDB2, RF_accuracy_ALLIDB2, KNN_accuracy_ALLIDB2];
accuracy_Subtypes = [SVM_accuracy_Subtypes, RF_accuracy_Subtypes, KNN_accuracy_Subtypes];

% 🔹 Plot Accuracy Comparison
figure;
barData = [accuracy_ALLIDB2; accuracy_Subtypes]';

b = bar(barData, 'grouped'); % Grouped bar plot
xticklabels(models);
ylabel('Accuracy (%)');
title('Accuracy Comparison: ALL vs Normal & Subtypes');

% 🔹 Assign Different Colors to Each Model
b(1).FaceColor = [0 0.4470 0.7410]; % Blue for ALL vs Normal
b(2).FaceColor = [0.8500 0.3250 0.0980]; % Orange for Subtypes

legend({'ALL vs Normal', 'Subtypes'}, 'Location', 'northwest');

disp('✅ Accuracy Comparison Completed');
