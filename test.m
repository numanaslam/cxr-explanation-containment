clc;
clear all;
close all;
image_size = [224 224 3];

imds = imageDatastore({'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\*.png';...
    'C:\paper2_repo\input\annotated_gray\annotated_gray\normal\*.png';
    }, 'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end


net = vgg16();
inputSize = net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain,'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', imageAugmenter);


augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation',imageAugmenter);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',true, ...
    'Plots','training-progress');


netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);


img = imread("C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\CHNCXR_0362_1.png");
img = imresize(img,[224 224]);


inputSize = netTransfer.Layers(1).InputSize(1:2);
classes = netTransfer.Layers(end).Classes;

[YPred,scores] = classify(netTransfer,img);
[~,topIdx] = maxk(scores, 3);
topScores = scores(topIdx);
topClasses = classes(topIdx);

figure;
imshow(img)
titleString = compose("%s (%.2f)",topClasses,topScores');
title(sprintf(join(titleString, "; ")));

map = imageLIME(netTransfer,img,YPred);

figure
imshow(img,'InitialMagnification',150)
hold on
imagesc(map,'AlphaData',0.5)
colormap jet
colorbar

title(sprintf("Image LIME (%s)", ...
    YPred))
hold off



secondClass = topClasses(2);
map = imageLIME(netTransfer,img,secondClass);
figure;
imshow(img,'InitialMagnification',150)
hold on
imagesc(map,'AlphaData',0.5)
colormap jet
colorbar

title(sprintf("Top 2 classes - Image LIME (%s)",secondClass))
hold off

%%
% 

[map,featureMap,featureImportance] = imageLIME(netTransfer,img,YPred);
numTopFeatures = 10;
[~,idx] = maxk(featureImportance,numTopFeatures);
mask = ismember(featureMap,idx);
maskedImg = uint8(mask).*img;

figure
imshow(maskedImg);

title(sprintf("Image LIME (%s - top %i features)", ...
    YPred, numTopFeatures))

