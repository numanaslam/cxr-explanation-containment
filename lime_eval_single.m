%% lime_eval_single.m
% Single-image LIME-vs-lung IoU evaluation with visualization.
% LIME settings match the recovered original code: superpixels, 64 features,
% 2048 samples, tree model, plain maxk top-K (no positive filter).
%
% Flat folders:
%   input\cxr\*.png   input\mask\*.png   input\roi\*.png

clc; clear; close all;
rng('shuffle');

%% ---- Config -------------------------------------------------------------
cxrFolder  = 'input\cxr';
maskFolder = 'input\mask';
roiFolder  = 'input\roi';
netFile    = 'vgg16_net.mat';

Segmentation = 'superpixels';   % or 'grid'
NumFeatures  = 30;              % recovered default (grid: use 49)
NumSamples   = 2048;            % recovered default
LimeModel    = 'tree';          % recovered default
TopK         = 30;              % top-K segments
ImageType    = 'complete';       % 'partial' (roi) or 'complete' (cxr)

pickName = '';   % e.g. 'CHNCXR_0020_0'; leave '' to choose at random

%% ---- Choose an image ----------------------------------------------------
roi = dir(fullfile(roiFolder, '*.png'));
if isempty(pickName)
    base = erase(roi(randi(numel(roi))).name, '.png');
else
    base = pickName;
end
fprintf('Image: %s   (%s, %s, K=%d)\n', base, ImageType, Segmentation, TopK);

cxrFile  = fullfile(cxrFolder, [base '.png']);
roiFile  = fullfile(roiFolder, [base '.png']);
hit      = dir(fullfile(maskFolder, [base '*'])); hit = hit(~[hit.isdir]);
maskFile = fullfile(hit(1).folder, hit(1).name);

%% ---- Load net + build input + mask -------------------------------------
S   = load(netFile);
net = S.netTransfer;
inSize = net.Layers(1).InputSize;
H = inSize(1); W = inSize(2);

if strcmpi(ImageType, 'partial'), srcFile = roiFile; else, srcFile = cxrFile; end
X = imread(srcFile);
if size(X,3) == 1, X = repmat(X, [1 1 3]); end
X = uint8(imresize(X, [H W]));

M = imread(maskFile);
if size(M,3) == 3, M = rgb2gray(M); end
B = imresize(M > 0, [H W], 'nearest');

%% ---- LIME + top-K mask A' ----------------------------------------------
label = classify(net, X);
[scoreMap, fmap, fimp] = imageLIME(net, X, label, ...
    'Segmentation', Segmentation, 'NumFeatures', NumFeatures, ...
    'NumSamples', NumSamples, 'Model', LimeModel);

k = min(TopK, numel(fimp));
[~, idx] = maxk(fimp, k);
A = ismember(fmap, idx);

inter = nnz(A & B); uni = nnz(A | B); iou = inter / max(uni,1);

%% ---- Report -------------------------------------------------------------
fprintf('predicted class       : %s\n', string(label));
fprintf('segments returned     : %d\n', numel(fimp));
fprintf('segments selected (A) : %d\n', k);
fprintf('A area frac           : %.3f\n', nnz(A)/(H*W));
fprintf('B area frac           : %.3f\n', nnz(B)/(H*W));
fprintf('intersection / union  : %d / %d\n', inter, uni);
fprintf('IoU                   : %.4f\n', iou);

%% ---- Visualize ----------------------------------------------------------
figure('Name', sprintf('%s  IoU=%.3f', base, iou), 'Position', [60 60 1400 320]);
subplot(1,5,1); imshow(X);                title('input');
subplot(1,5,2); imshow(label2rgb(fmap));  title(sprintf('%d segments', numel(fimp)));
subplot(1,5,3); imshow(X); hold on;
hh = imagesc(scoreMap); set(hh,'AlphaData',0.5); colormap(gca,'jet'); title('importance');
subplot(1,5,4); imshow(X); hold on;
hm = imshow(cat(3, zeros(H,W), ones(H,W), zeros(H,W)));
set(hm, 'AlphaData', 0.45*A); title(sprintf('top-%d mask A''', k));
ov = zeros(H,W,3);
ov(:,:,1) = double(B & ~A) + double(A & B);   % red = lung, yellow = overlap
ov(:,:,2) = double(A & ~B) + double(A & B);   % green = outside, yellow = overlap
subplot(1,5,5); imshow(ov); title(sprintf('R=lung G=A\\\\lung Y=overlap (IoU=%.2f)', iou));