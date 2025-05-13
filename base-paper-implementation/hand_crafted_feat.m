clc; clear; close all;

disp('🚀 Starting Handcrafted Feature Extraction...');

%% 1️⃣ Load Wavelet-Transformed Features
try
    load('wavelet_features.mat');
    disp('✅ Wavelet Features Loaded Successfully!');
catch ME
    disp(['❌ Error loading wavelet features: ', ME.message]);
    return; % Stop execution if loading fails
end

%% 2️⃣ Check Feature Sizes
disp(['Size of LL_train_ALLIDB2: ', num2str(size(LL_train_ALLIDB2, 1)), ' x ', num2str(size(LL_train_ALLIDB2, 2))]);
disp(['Size of LL_train_Subtypes: ', num2str(size(LL_train_Subtypes, 1)), ' x ', num2str(size(LL_train_Subtypes, 2))]);

%% 3️⃣ Convert Features to 2D for GLCM Processing
numFeatures = size(LL_train_ALLIDB2, 2);
matrixSize = round(sqrt(numFeatures));

if matrixSize^2 > numFeatures
    disp('⚠️ Warning: Feature vector size is not a perfect square, cropping to fit.');
end

try
    LL_train_ALLIDB2_reshaped = reshape(LL_train_ALLIDB2(:, 1:matrixSize^2), [], matrixSize);
    LL_train_Subtypes_reshaped = reshape(LL_train_Subtypes(:, 1:matrixSize^2), [], matrixSize);
    disp('✅ Feature Reshaping Done!');
catch ME
    disp(['❌ Error reshaping features: ', ME.message]);
    return;
end

%% 4️⃣ Extract GLCM Features (Texture-Based)
offsets = [0 1; -1 1; -1 0; -1 -1]; % Define GLCM Offsets

try
    glcm_ALLIDB2 = graycomatrix(LL_train_ALLIDB2_reshaped, 'Offset', offsets);
    stats_ALLIDB2.glcm = graycoprops(glcm_ALLIDB2, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

    glcm_Subtypes = graycomatrix(LL_train_Subtypes_reshaped, 'Offset', offsets);
    stats_Subtypes.glcm = graycoprops(glcm_Subtypes, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

    disp('✅ GLCM Features Extracted Successfully!');
catch ME
    disp(['❌ Error extracting GLCM features: ', ME.message]);
    return;
end

%% 5️⃣ Extract LBP Features (Local Binary Patterns)
radius = 1;
numPoints = 8 * radius;

try
    lbp_ALLIDB2 = extractLBPFeatures(LL_train_ALLIDB2_reshaped, 'Upright', false, 'Radius', radius, 'NumNeighbors', numPoints);
    lbp_Subtypes = extractLBPFeatures(LL_train_Subtypes_reshaped, 'Upright', false, 'Radius', radius, 'NumNeighbors', numPoints);
    disp('✅ LBP Features Extracted Successfully!');
catch ME
    disp(['❌ Error extracting LBP features: ', ME.message]);
    return;
end

%% 6️⃣ Extract Statistical Features (Mean, Variance, Skewness, Entropy)
try
    stats_ALLIDB2.mean = mean(LL_train_ALLIDB2, 2);
    stats_ALLIDB2.variance = var(LL_train_ALLIDB2, 0, 2);
    stats_ALLIDB2.skewness = skewness(LL_train_ALLIDB2, 1, 2);
    stats_ALLIDB2.entropy = -sum(LL_train_ALLIDB2 .* log(abs(LL_train_ALLIDB2 + eps)), 2);

    stats_Subtypes.mean = mean(LL_train_Subtypes, 2);
    stats_Subtypes.variance = var(LL_train_Subtypes, 0, 2);
    stats_Subtypes.skewness = skewness(LL_train_Subtypes, 1, 2);
    stats_Subtypes.entropy = -sum(LL_train_Subtypes .* log(abs(LL_train_Subtypes + eps)), 2);

    disp('✅ Statistical Features Extracted Successfully!');
catch ME
    disp(['❌ Error extracting statistical features: ', ME.message]);
    return;
end

%% 7️⃣ Save Extracted Features
try
    save('handcrafted_features.mat', 'stats_ALLIDB2', 'stats_Subtypes', 'lbp_ALLIDB2', 'lbp_Subtypes');
    disp('✅ Handcrafted Features Saved as handcrafted_features.mat 🎉');
catch ME
    disp(['❌ Error saving handcrafted features: ', ME.message]);
end

disp('🚀 Handcrafted Feature Extraction Completed Successfully! ✅');
