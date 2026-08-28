%% LIME base image + heatmap + IoU vs U-Net mask
clc; clear; close all;
rng(42);   % imageLIME is stochastic; fix for reproducible sweeps

%% --- paths & params ---
modelFile    = 'alexnet_v2.mat';
imgFile      = 'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\CHNCXR_0327_1.png';
unetMaskFile = 'C:\paper2_repo\input\mask\CHNCXR_0327_1_mask.png';
numTopFeatures = 30;      % K
numFeatures    = 49;      % default segmentation granularity; raise (e.g. 64) for finer map
numSamples     = 2048;    % raise (e.g. 3072) for a cleaner, more stable map

%% --- load model ---
S = load(modelFile);
if isfield(S,'netTransfer'), net = S.netTransfer;
else, fn = fieldnames(S); net = S.(fn{1}); end
inputSize = net.Layers(1).InputSize(1:2);

%% --- read & prep image (gray2rgb to match training) ---
img = imread(imgFile);
img = imresize(img, inputSize);
if size(img,3)==1, img = repmat(img,1,1,3); end
img = uint8(img);

%% --- classify + LIME (matches original call) ---
YPred = classify(net, img);
[map, featureMap, featureImportance] = imageLIME(net, img, YPred, ...
    'NumFeatures', numFeatures, 'NumSamples', numSamples);

%% --- top-K superpixel mask + the base image from your original script ---
[~, idx]  = maxk(featureImportance, numTopFeatures);
limeMask  = ismember(featureMap, idx);          % logical
maskedImg = uint8(limeMask) .* img;             % <-- the original "LIME base image"

%% --- U-Net mask (nearest keeps it binary) ---
gt = imread(unetMaskFile);
if size(gt,3)==3, gt = rgb2gray(gt); end
gt = imresize(gt, inputSize, 'nearest');
gtMask = gt > 0;                                % use gt > 127 if it's soft/greyscale

%% --- metrics ---
inter = nnz(limeMask & gtMask);
uni   = nnz(limeMask | gtMask);
IoU       = inter / uni;
Dice      = 2*inter / (nnz(limeMask) + nnz(gtMask));
precision = inter / nnz(limeMask);              % LIME coverage inside lung
recall    = inter / nnz(gtMask);
fprintf('K=%d (of %d feats)  IoU=%.4f  Dice=%.4f  Prec=%.4f  Rec=%.4f\n', ...
    numTopFeatures, numel(unique(featureMap)), IoU, Dice, precision, recall);

%% --- figure 1: smooth LIME heatmap (MathWorks style) ---
figure; imshow(img,'InitialMagnification',150); hold on;
imagesc(map,'AlphaData',0.5); colormap jet; colorbar;
title(sprintf('Image LIME (%s)', YPred)); hold off;

%% --- figure 2: LIME base image (top-K features), your original view ---
figure; imshow(maskedImg);
title(sprintf('Image LIME (%s - top %d features)', YPred, numTopFeatures));

%% --- figure 3: IoU overlay (LIME=red, U-Net=green, overlap=yellow) ---
figure; imshow(img); hold on;
red = cat(3, ones(inputSize), zeros(inputSize), zeros(inputSize));
grn = cat(3, zeros(inputSize), ones(inputSize), zeros(inputSize));
h1 = imshow(red); set(h1,'AlphaData', 0.35*limeMask);
h2 = imshow(grn); set(h2,'AlphaData', 0.35*gtMask);
title(sprintf('K=%d  IoU=%.3f  Prec=%.3f  Rec=%.3f', ...
    numTopFeatures, IoU, precision, recall));
hold off;