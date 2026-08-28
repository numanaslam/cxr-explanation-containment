%% lime_scores_all.m
% Compute and clearly distinguish the THREE overlap quantities between the
% LIME top-K mask (A') and the lung mask (B). Same A', three denominators:
%
%   Jaccard / IoU  = |A' n B| / |A' u B|   <- the paper's Algorithm 4 / equation
%   Coverage       = |A' n B| / |B|         <- fraction of lung the explanation covers
%   Precision      = |A' n B| / |A'|        <- fraction of explanation inside the lung
%
% These are different metrics. Jaccard penalises spill outside the lung via
% the union; Coverage and Precision do not. Reporting any of them is fine -
% but each must be named as what it is.
%
% Settings = recovered defaults. Runs on N random images, prints a summary.

clc; clear; close all;
rng('shuffle');

%% ---- Config -------------------------------------------------------------
cxrFolder  = 'input\cxr';
maskFolder = 'input\mask';
roiFolder  = 'input\roi';
netFile    = 'vgg16_net.mat';

Segmentation = 'superpixels';
NumFeatures  = 64;          % recovered default
NumSamples   = 2048;        % recovered default
LimeModel    = 'tree';      % recovered default
TopK         = 30;
ImageType    = 'partial';   % 'partial' (roi) or 'complete' (cxr)
N            = 20;          % number of random images to summarise

%% ---- Load --------------------------------------------------------------
S = load(netFile); net = S.netTransfer;
inSize = net.Layers(1).InputSize; H = inSize(1); W = inSize(2);
roi = dir(fullfile(roiFolder,'*.png'));
sel = randperm(numel(roi), min(N, numel(roi)));

J = nan(numel(sel),1); Cov = nan(numel(sel),1); Prec = nan(numel(sel),1);

for n = 1:numel(sel)
    base = erase(roi(sel(n)).name, '.png');
    if strcmpi(ImageType,'partial'), src = fullfile(roiFolder,[base '.png']);
    else,                            src = fullfile(cxrFolder,[base '.png']); end

    X = imread(src); if size(X,3)==1, X = repmat(X,[1 1 3]); end
    X = uint8(imresize(X,[H W]));

    hit = dir(fullfile(maskFolder,[base '*'])); hit = hit(~[hit.isdir]);
    M = imread(fullfile(hit(1).folder, hit(1).name));
    if size(M,3)==3, M = rgb2gray(M); end
    B = imresize(M>0,[H W],'nearest');

    label = classify(net, X);
    [~,fmap,fimp] = imageLIME(net, X, label, ...
        'Segmentation',Segmentation,'NumFeatures',NumFeatures, ...
        'NumSamples',NumSamples,'Model',LimeModel);
    k = min(TopK, numel(fimp));
    [~,idx] = maxk(fimp, k);
    A = ismember(fmap, idx);

    inter = nnz(A & B);
    J(n)    = inter / max(nnz(A | B),1);   % Jaccard (paper)
    Cov(n)  = inter / max(nnz(B),1);       % coverage
    Prec(n) = inter / max(nnz(A),1);       % precision
end

fprintf('\n=== %s, K=%d, %s, NumFeatures=%d, n=%d images ===\n', ...
    'VGG16', TopK, ImageType, NumFeatures, numel(sel));
fprintf('Jaccard/IoU (paper eq.)  : mean %.4f   (std %.4f)\n', mean(J,'omitnan'),   std(J,'omitnan'));
fprintf('Coverage |AnB|/|B|       : mean %.4f   (std %.4f)\n', mean(Cov,'omitnan'), std(Cov,'omitnan'));
fprintf('Precision |AnB|/|A|      : mean %.4f   (std %.4f)\n', mean(Prec,'omitnan'),std(Prec,'omitnan'));
fprintf('\nNote: these are three different metrics. The paper''s Tables 7-10\n');
fprintf('and Algorithm 4 define Jaccard (the first line).\n');