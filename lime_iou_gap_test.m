%% lime_iou_gap_test.m
% Tests ONE hypothesis: is the depressed partial-image IoU caused by the
% lung mask B being a FILLED silhouette (two lobes + the central
% mediastinal gap) while the LIME explanation A' only covers lung tissue?
%
% For each image it reports IoU under three mask definitions, so we can see
% how much of the loss is the central gap vs. genuine spill outside the lung:
%   B_full  : mask exactly as stored (filled silhouette)
%   B_lobes : mask with the central gap removed is NOT done here; instead we
%             measure how much of A' falls inside vs outside B, and how much
%             of B is the central column that A' can never cover.
%
% Diagnostic only - no tuning. Settings = recovered defaults.

clc; clear; close all;
rng('shuffle');

%% ---- Config -------------------------------------------------------------
cxrFolder  = 'input\cxr';
maskFolder = 'input\mask';
roiFolder  = 'input\roi';
netFile    = 'vgg16_net.mat';

Segmentation = 'superpixels';
NumFeatures  = 64;          % recovered default
NumSamples   = 2048;
LimeModel    = 'tree';
TopK         = 30;
ImageType    = 'partial';   % gap effect is cleanest on partial (ROI) input
nShow        = 4;

%% ---- Load --------------------------------------------------------------
S = load(netFile); net = S.netTransfer;
inSize = net.Layers(1).InputSize; H = inSize(1); W = inSize(2);
roi = dir(fullfile(roiFolder, '*.png'));

for n = 1:min(nShow, numel(roi))
    base = erase(roi(n).name, '.png');
    if strcmpi(ImageType,'partial'), srcFile = fullfile(roiFolder,[base '.png']);
    else,                            srcFile = fullfile(cxrFolder,[base '.png']); end

    X = imread(srcFile);
    if size(X,3)==1, X = repmat(X,[1 1 3]); end
    X = uint8(imresize(X,[H W]));

    hit = dir(fullfile(maskFolder,[base '*'])); hit = hit(~[hit.isdir]);
    M = imread(fullfile(hit(1).folder, hit(1).name));
    if size(M,3)==3, M = rgb2gray(M); end
    B = imresize(M > 0, [H W], 'nearest');           % filled silhouette

    % --- LIME -> A' ---
    label = classify(net, X);
    [~, fmap, fimp] = imageLIME(net, X, label, ...
        'Segmentation',Segmentation,'NumFeatures',NumFeatures, ...
        'NumSamples',NumSamples,'Model',LimeModel);
    k = min(TopK, numel(fimp));
    [~, idx] = maxk(fimp, k);
    A = ismember(fmap, idx);

    % --- standard IoU against the filled mask ---
    iou_full = nnz(A & B) / max(nnz(A | B),1);

    % --- decompose where A' lands ---
    A_in   = nnz(A & B);          % explanation inside lung
    A_out  = nnz(A & ~B);         % explanation outside lung (true spill)
    B_uncov= nnz(B & ~A);         % lung not covered by explanation

    % --- "precision-style" overlap: of A', how much is inside the lung? ---
    frac_A_in_lung = A_in / max(nnz(A),1);

    % --- coverage: of the lung, how much did A' cover? ---
    frac_B_covered = A_in / max(nnz(B),1);

    fprintf('\n--- %s (%s) ---\n', base, ImageType);
    fprintf('A area=%d (%.3f)   B area=%d (%.3f)\n', nnz(A),nnz(A)/(H*W), nnz(B),nnz(B)/(H*W));
    fprintf('IoU (filled mask)         : %.4f\n', iou_full);
    fprintf('frac of A INSIDE lung      : %.4f   <- if high, A is not spilling\n', frac_A_in_lung);
    fprintf('frac of lung COVERED by A  : %.4f\n', frac_B_covered);
    fprintf('A outside lung (px)        : %d\n', A_out);
    fprintf('lung uncovered by A (px)   : %d\n', B_uncov);

    % --- visual: where is the IoU being lost? ---
    figure('Name', sprintf('%s gap-test', base), 'Position',[60 60 1100 320]);
    subplot(1,3,1); imshow(X); title('ROI input');
    subplot(1,3,2); imshow(B); title('lung mask B (filled)');
    ov = zeros(H,W,3);
    ov(:,:,1) = double(B & ~A);          % red   = lung A missed (incl. center gap)
    ov(:,:,2) = double(A & ~B);          % green = A outside lung (real spill)
    ov(:,:,3) = double(A & B);           % blue  = correct overlap
    subplot(1,3,3); imshow(ov);
    title('R=missed lung  G=spill  B=overlap');
end