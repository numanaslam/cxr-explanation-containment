%% retrain_v2.m
% Retrain AlexNet / VGG16 / VGG19 on the annotated_gray ROI images and save
% each under a reusable "_v2" name together with its exact validation split.
% Based on the supplied training code (sgdm, batch 10, 100 epochs, lr 1e-4).

clc; clear; close all;
rng(42);   % fixed split for reproducibility / leak-free evaluation

dataFolders = { ...
    'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\*.png'; ...
    'C:\paper2_repo\input\annotated_gray\annotated_gray\normal\*.png'};

imds = imageDatastore(dataFolders, 'LabelSource', 'foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
fprintf('Classes: %s | train=%d val=%d\n', strjoin(categories(imdsTrain.Labels),', '), ...
    numel(imdsTrain.Labels), numel(imdsValidation.Labels));

pixelRange = [-30 30];
augmenter = imageDataAugmenter('RandXReflection',true, ...
    'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);

archs = { 'AlexNet', @alexnet, 'alexnet_v2.mat'; ...
    'VGG16',   @vgg16,   'vgg16_v2.mat';   ...
    'VGG19',   @vgg19,   'vgg19_v2.mat' };

for a = 1:size(archs,1)
    name = archs{a,1}; loadNet = archs{a,2}; outFile = archs{a,3};
    fprintf('\n==== Training %s ====\n', name);

    net = loadNet();
    inputSize = net.Layers(1).InputSize;
    layers = [ net.Layers(1:end-3)
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer ];

    augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
    augVal   = augmentedImageDatastore(inputSize(1:2), imdsValidation);

    valFreq = max(1, floor(numel(imdsTrain.Labels)/10));
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',10,'MaxEpochs',100,'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch','ValidationData',augVal, ...
        'ValidationFrequency',valFreq,'ExecutionEnvironment','auto', ...
        'Verbose',true,'Plots','training-progress');

    netTransfer = trainNetwork(augTrain, layers, options);

    YPred = classify(netTransfer, augVal);
    fprintf('%s val accuracy: %.4f\n', name, mean(YPred==imdsValidation.Labels));

    valFiles = imdsValidation.Files; valLabels = imdsValidation.Labels; %#ok<NASGU>
    save(outFile, 'netTransfer','valFiles','valLabels','-v7.3');
    fprintf('Saved -> %s\n', outFile);
end
fprintf('\nDone. Models: alexnet_v2.mat, vgg16_v2.mat, vgg19_v2.mat\n');