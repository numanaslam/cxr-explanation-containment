%% Generate methodology-figure thumbnails: limebase.png + gradcam.png
%  limebase = top-K LIME superpixel mask applied to the image (the "LIME base")
%  gradcam  = jet Grad-CAM heat-map overlaid on the (grayscale) lung image
%  Use a representative ROI image + a model that shows good containment (VGG16).
clc; clear; close all;
rng(42);   % imageLIME is stochastic; fix for a reproducible thumbnail

%% --- config ---
modelFile = 'vgg16_v2.mat';
imgFile   = 'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\CHNCXR_0327_1.png'; % ROI lung image
outDir    = 'C:\paper2_repo\results\assets';   % copy these into the SVG's assets\ folder
numFeatures = 50;    % LIME superpixels
numTopK     = 30;    % top-K for the LIME base
numSamples  = 1000;
gradAlpha   = 0.45;  % heat-map opacity for the Grad-CAM overlay
contrastNorm = true; % imadjust for a cleaner-looking thumbnail

if ~exist(outDir,'dir'), mkdir(outDir); end

%% --- load model ---
S = load(modelFile);
if isfield(S,'netTransfer'), net = S.netTransfer;
else, fn = fieldnames(S); net = S.(fn{1}); end
inputSize = net.Layers(1).InputSize(1:2);

%% --- read & prep image (gray -> rgb to match training) ---
img = imread(imgFile);
if size(img,3)==3, g = rgb2gray(img); else, g = img; end
if contrastNorm, g = imadjust(g); end
g   = imresize(g, inputSize);
img = uint8(repmat(g, 1, 1, 3));          % HxWx3 grayscale-as-rgb
YPred = classify(net, img);
fprintf('Predicted class: %s\n', string(YPred));

%% --- LIME base image (top-K superpixels applied) ---
[~, fMap, fImp] = imageLIME(net, img, YPred, ...
    'Segmentation','superpixels', 'NumFeatures',numFeatures, 'NumSamples',numSamples);
[~, idx]  = maxk(fImp, min(numTopK, numel(fImp)));
limeMask  = ismember(fMap, idx);          % logical top-K mask
limeBase  = uint8(limeMask) .* img;       % keep top-K regions, rest black
imwrite(limeBase, fullfile(outDir,'limebase.png'));

%% --- Grad-CAM heat-map overlay (jet on grayscale) ---
scoreMap = gradCAM(net, img, YPred);
scoreMap = mat2gray(imresize(double(scoreMap), inputSize));   % normalise 0..1
heat     = ind2rgb(uint8(255*scoreMap), jet(256));            % HxWx3 jet
base     = im2double(img);                                    % grayscale rgb
overlay  = (1-gradAlpha)*base + gradAlpha*heat;               % blend
imwrite(overlay, fullfile(outDir,'gradcam.png'));

fprintf('Saved limebase.png and gradcam.png to %s\n', outDir);
fprintf('Copy both into the folder next to methodology.svg (assets\\).\n');

%% --- optional: preview ---
figure;
subplot(1,2,1); imshow(limeBase); title(sprintf('LIME base (top %d)', numTopK));
subplot(1,2,2); imshow(overlay);  title('Grad-CAM overlay');
