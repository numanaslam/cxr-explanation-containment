%% train_models.m
% Train AlexNet, VGG16 and VGG19 on lung-segmented (ROI) chest X-ray images
% for binary classification (normal vs ptb), then save each trained model
% together with the EXACT train/validation split so that evaluation
% (evaluate_models.m) is reproducible and leak-free.
%
% Requires the pretrained-network support packages (Home > Add-Ons):
%   - Deep Learning Toolbox Model for AlexNet Network
%   - Deep Learning Toolbox Model for VGG-16 Network
%   - Deep Learning Toolbox Model for VGG-19 Network

clc; clear; close all;

%% ---- Config -------------------------------------------------------------
dataFolders = { ...
    'C:\numan\input\roi\normal\*.png'; ...
    'C:\numan\input\roi\ptb\*.png'};

splitSeed     = 42;      % fixed seed -> reproducible held-out set
trainFraction = 0.8;     % 80/20 split (matches the recovered script)
useGray2RGB   = true;    % grayscale source -> 3-channel net input.
                         % Use the SAME value in evaluate_models.m
                         % (it is saved into each .mat, see bottom).

% Training options.
% NOTE: the recovered script used InitialLearnRate 1e-4 and MiniBatchSize 10,
% but the manuscript (Table 2 / Methods) states 0.001 and 16. Reconcile these
% before reporting numbers. Defaults below match the recovered script.
miniBatch  = 10;
maxEpochs  = 300;
initialLR  = 1e-4;

% Architectures: {display name, pretrained loader, output .mat file}
archs = { ...
    'AlexNet', @alexnet, 'alexnet_net.mat'; ...
    'VGG16',   @vgg16,   'vgg16_net.mat';   ...
    'VGG19',   @vgg19,   'vgg19_net.mat'};

%% ---- Data + reproducible split -----------------------------------------
rng(splitSeed);
imds = imageDatastore(dataFolders, 'LabelSource', 'foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds, trainFraction, 'randomized');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('Classes: %s | train = %d, validation = %d\n', ...
    strjoin(categories(imdsTrain.Labels), ', '), ...
    numel(imdsTrain.Labels), numel(imdsValidation.Labels));

% Augmentation (applied to TRAINING data only)
pixelRange = [-30 30];
augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

%% ---- Train each architecture -------------------------------------------
for a = 1:size(archs, 1)
    name    = archs{a, 1};
    loadNet = archs{a, 2};
    outFile = archs{a, 3};

    fprintf('\n================  Training %s  ================\n', name);

    net       = loadNet();                  % load pretrained network
    inputSize = net.Layers(1).InputSize;    % [227 227 3] (AlexNet) or [224 224 3] (VGG)

    % Replace the final 3 layers (fc / softmax / classification) for 2 classes
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses, ...
            'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer];

    % Build datastores at THIS network's input size, with identical
    % preprocessing for train and validation.
    if useGray2RGB
        augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
            'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);
        augVal   = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
            'ColorPreprocessing', 'gray2rgb');
    else
        augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
            'DataAugmentation', augmenter);
        augVal   = augmentedImageDatastore(inputSize(1:2), imdsValidation);
    end

    valFreq = max(1, floor(numel(imdsTrain.Labels) / miniBatch));  % ~once per epoch

    options = trainingOptions('sgdm', ...
        'MiniBatchSize',        miniBatch, ...
        'MaxEpochs',            maxEpochs, ...
        'InitialLearnRate',     initialLR, ...
        'Shuffle',              'every-epoch', ...
        'ValidationData',       augVal, ...
        'ValidationFrequency',  valFreq, ...
        'OutputNetwork',        'best-validation-loss', ... % remove if pre-R2021b
        'ExecutionEnvironment', 'auto', ...                 % uses GPU if available
        'Verbose',              true, ...
        'Plots',                'training-progress');

    netTransfer = trainNetwork(augTrain, layers, options);

    % Quick validation accuracy sanity check
    YPred  = classify(netTransfer, augVal);
    valAcc = mean(YPred == imdsValidation.Labels);
    fprintf('%s validation accuracy: %.4f\n', name, valAcc);

    % Save the trained net WITH the exact split + preprocessing flag, so
    % evaluate_models.m can score the same held-out images (no leakage).
    valFiles  = imdsValidation.Files;    %#ok<NASGU>
    valLabels = imdsValidation.Labels;   %#ok<NASGU>
    save(outFile, 'netTransfer', 'valFiles', 'valLabels', ...
        'useGray2RGB', 'splitSeed', '-v7.3');
    fprintf('Saved %s -> %s\n', name, outFile);
end

fprintf('\nAll models trained and saved.\n');