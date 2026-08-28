%% lime_grid_visualize.m
% Visualize grid-based LIME for one model: segmentation grid, importance
% heatmap, top-K explanation mask A', and its overlay on the lung mask B.
% Masks are read from the masks\{normal,ptb}\ folder (..._mask.png).

clc; clear; close all;
rng(42);

%% ---- Config -------------------------------------------------------------
netFile      = 'vgg16_net.mat';
maskFolder   = 'C:\numan\input\masks';   % contains normal\ and ptb\ subfolders
NumFeatures  = 49;          % grid cells (49 = 7x7); try 64/100 to refine
TopK         = 30;          % top-K positive segments to keep
NumSamples   = 1000;
PositiveOnly = true;
ImageType    = 'complete';  % 'complete' (cxr) or 'partial' (roi)
nShow        = 3;

%% ---- Load ---------------------------------------------------------------
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

    % --- lung mask B from the masks folder (matched by base name) ---
    [~,base] = fileparts(roiFiles{i});
    hits = dir(fullfile(maskFolder,'**',[base '*']));
    if isempty(hits)
        warning('No mask for %s; skipping.', base); continue;
    end
    M = imread(fullfile(hits(1).folder, hits(1).name));
    if size(M,3)==3, M = rgb2gray(M); end
    B = imresize(M > 0, [H W], 'nearest');

    % --- grid LIME ---
    label = classify(net, X);
    [scoreMap, fmap, fimp] = imageLIME(net, X, label, ...
        'Segmentation','grid', 'NumFeatures',NumFeatures, ...
        'NumSamples',NumSamples, 'Model','linear');

    % --- top-K -> A' ---
    if PositiveOnly, cand = find(fimp>0); else, cand = (1:numel(fimp))'; end
    [~,ord] = sort(fimp(cand),'descend');
    k = min(TopK, numel(cand));
    idx = cand(ord(1:k));
    A = ismember(fmap, idx);

    inter = nnz(A & B); uni = nnz(A | B); iou = inter/max(uni,1);

    fprintf('\n--- image %d (%s, grid, NumFeatures=%d, K=%d) ---\n', i, ImageType, NumFeatures, TopK);
    fprintf('segments=%d  positive=%d  selected=%d\n', numel(fimp), nnz(fimp>0), k);
    fprintf('A frac=%.3f  B frac=%.3f  IoU=%.4f\n', nnz(A)/(H*W), nnz(B)/(H*W), iou);

    % --- figure: input | grid | heatmap | top-K mask | overlay ---
    figure('Name', sprintf('grid LIME  img %d  IoU=%.3f', i, iou), 'Position',[80 80 1300 300]);

    subplot(1,5,1); imshow(X); title('input');

    subplot(1,5,2); imshow(X); hold on;
    bnd = boundarymask(fmap);
    hb = imshow(cat(3, ones(H,W), zeros(H,W), zeros(H,W)));  % red grid lines
    set(hb,'AlphaData', 0.9*bnd); title(sprintf('grid (%d cells)', numel(fimp)));

    subplot(1,5,3); imshow(X); hold on;
    hh = imagesc(scoreMap); set(hh,'AlphaData',0.5); colormap(gca,'jet');
    title('importance');

    subplot(1,5,4); imshow(X); hold on;
    hm = imshow(cat(3, zeros(H,W), ones(H,W), zeros(H,W)));   % green top-K cells
    set(hm,'AlphaData', 0.45*A); title(sprintf('top-%d mask A''', k));

    ov = zeros(H,W,3);
    ov(:,:,1) = double(B & ~A) + double(A & B);   % red = lung missed, +overlap
    ov(:,:,2) = double(A & ~B) + double(A & B);   % green = outside lung, +overlap
    subplot(1,5,5); imshow(ov);
    title(sprintf('R=lung  G=A\\\\lung  Y=overlap (IoU=%.2f)', iou));
end