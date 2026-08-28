%% lime_iou_diagnostic.m
% Single-model LIME-vs-lung IoU diagnostic, using the SETTINGS RECOVERED
% from the original training/LIME script:
%   imageLIME(net, X, label)  with NO options  ->  MATLAB defaults:
%     Segmentation = 'superpixels', NumFeatures = 64,
%     NumSamples   = 2048,          Model       = 'tree'
%   top-K via maxk(featureImportance, K)   (NO positive filter)
% Masks read from masks\{normal,ptb}\ (..._mask.png).

clc; clear; close all;
rng(42);

%% ---- Config (matched to recovered code) --------------------------------
netFile      = 'vgg16_net.mat';
maskFolder   = 'C:\Users\Rehan Saleem\Downloads\mask\mask';
Segmentation = 'superpixels';   % recovered default
NumFeatures  = 64;              % recovered default (was 50)
NumSamples   = 2048;            % recovered default (was 1000)
LimeModel    = 'tree';          % recovered default (was 'linear')
TopK         = 30;
ImageType    = 'partial';      % 'complete' (cxr) or 'partial' (roi)
nShow        = 3;

%% ---- Load --------------------------------------------------------------
S   = load(netFile);
net = S.netTransfer;
roiFiles = S.valFiles;
cxrFiles = strrep(roiFiles, [filesep 'roi' filesep], [filesep 'cxr' filesep]);

inSize = net.Layers(1).InputSize;
H = inSize(1); W = inSize(2);

for i = 1:min(nShow, numel(roiFiles))
    if strcmpi(ImageType,'complete'), inputFile = cxrFiles{i};
    else,                             inputFile = roiFiles{i}; end

    % --- model input ---
    X = imread(inputFile);
    if size(X,3)==1, X = repmat(X,[1 1 3]); end
    X = uint8(imresize(X,[H W]));

    % --- lung mask B from the masks folder ---
    [~,base] = fileparts(roiFiles{i});
    hits = dir(fullfile(maskFolder,'**',[base '*']));
    if isempty(hits), warning('No mask for %s; skipping.', base); continue; end
    M = imread(fullfile(hits(1).folder, hits(1).name));
    if size(M,3)==3, M = rgb2gray(M); end
    B = imresize(M > 0, [H W], 'nearest');

    % --- LIME (recovered settings) ---
    label = classify(net, X);
    [~, fmap, fimp] = imageLIME(net, X, label, ...
        'Segmentation', Segmentation, 'NumFeatures', NumFeatures, ...
        'NumSamples', NumSamples, 'Model', LimeModel);

    % --- top-K via maxk, NO positive filter (matches recovered code) ---
    k = min(TopK, numel(fimp));
    [~, idx] = maxk(fimp, k);
    A = ismember(fmap, idx);

    inter = nnz(A & B); uni = nnz(A | B); iou = inter/max(uni,1);

    fprintf('\n--- image %d (%s) ---\n', i, ImageType);
    fprintf('segments returned     : %d\n', numel(fimp));
    fprintf('nonzero importance     : %d\n', nnz(fimp ~= 0));
    fprintf('segments selected (A) : %d\n', k);
    fprintf('A area frac           : %.3f\n', nnz(A)/(H*W));
    fprintf('B area frac           : %.3f\n', nnz(B)/(H*W));
    fprintf('IoU                   : %.4f\n', iou);
    fprintf('importance min/max     : %.4g / %.4g\n', min(fimp), max(fimp));

    figure('Name', sprintf('img %d  IoU=%.3f', i, iou));
    subplot(1,3,1); imshow(X);               title('input');
    subplot(1,3,2); imshow(label2rgb(fmap)); title('LIME segments');
    ov = zeros(H,W,3);
    ov(:,:,1) = double(B & ~A) + double(A & B);   % red lung + yellow overlap
    ov(:,:,2) = double(A & ~B) + double(A & B);   % green outside + yellow overlap
    subplot(1,3,3); imshow(ov); title('R=lung  G=A\\lung  Y=overlap');
end