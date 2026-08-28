%% lime_iou_scores.m
% Segmentation-guided IoU (Jaccard) sweep: LIME explanation vs lung mask.
% LIME settings match the recovered original code (tree / 64 / 2048 / maxk).
%
% Data layout (re-downloaded set):
%   input\cxr\{normal,ptb}\*.png   - full chest X-rays
%   input\mask\...  ..._mask.png   - lung masks (matched by base filename)
% "complete" = full cxr image;  "partial" = cxr masked by the lung mask.
%
% Test set is restricted to the saved held-out split (matched by filename),
% so training images do not leak in.

clc; clear; close all;
rng(42);

%% ---- Config -------------------------------------------------------------
cxrFolders = { ...
    'C:\numan\input\cxr\normal\*.png'; ...
    'C:\numan\input\cxr\ptb\*.png'};
maskFolder = 'C:\numan\input\mask';

nets = { ...
    'AlexNet', 'alexnet_net.mat'; ...
    'VGG16',   'vgg16_net.mat';   ...
    'VGG19',   'vgg19_net.mat'};

strategies = { 'superpixels', 64; 'grid', 49 };  % {Segmentation, NumFeatures}
Kvalues    = [30 49];
imageTypes = {'complete', 'partial'};

NumSamples = 2048;        % recovered default
NumRuns    = 5;
LimeModel  = 'tree';      % recovered default
maxImages  = 10;          % proof of concept; set Inf for full sweep
outputCSV  = 'lime_iou_scores.csv';

%% ---- Build image list + matched masks, restricted to held-out split ----
imds = imageDatastore(cxrFolders, 'LabelSource', 'foldernames');
cxrFiles = imds.Files;

% restrict to saved held-out test set (by base filename)
S0 = load(nets{1,2});
if isfield(S0, 'valFiles')
    valBases = baseNames(S0.valFiles);
    keep = ismember(baseNames(cxrFiles), valBases);
    cxrFiles = cxrFiles(keep);
    fprintf('Restricted to held-out split: %d images.\n', numel(cxrFiles));
else
    warning('No saved split found - using ALL images (training images may leak in).');
end

% match a mask to each image
maskFiles = cell(size(cxrFiles));
ok = true(size(cxrFiles));
for i = 1:numel(cxrFiles)
    [~, base] = fileparts(cxrFiles{i});
    hit = dir(fullfile(maskFolder, '**', [base '*']));
    hit = hit(~[hit.isdir]);
    if isempty(hit), ok(i) = false; else, maskFiles{i} = fullfile(hit(1).folder, hit(1).name); end
end
if ~all(ok), warning('%d images had no matching mask; dropped.', nnz(~ok)); end
cxrFiles = cxrFiles(ok);  maskFiles = maskFiles(ok);

if isfinite(maxImages)
    n = min(maxImages, numel(cxrFiles));
    cxrFiles = cxrFiles(1:n);  maskFiles = maskFiles(1:n);
end
fprintf('Images used: %d\n', numel(cxrFiles));

%% ---- Sweep --------------------------------------------------------------
rows = {};
for nIdx = 1:size(nets, 1)
    S   = load(nets{nIdx, 2});
    net = S.netTransfer;

    for sIdx = 1:size(strategies, 1)
        seg = strategies{sIdx, 1};
        nf  = strategies{sIdx, 2};
        for K = Kvalues
            for tIdx = 1:numel(imageTypes)
                opts = struct('Segmentation', seg, 'NumFeatures', nf, ...
                    'TopK', K, 'NumSamples', NumSamples, 'NumRuns', NumRuns, ...
                    'Model', LimeModel, 'Partial', strcmpi(imageTypes{tIdx},'partial'));

                r = computeLimeIoU(net, cxrFiles, maskFiles, opts);

                fprintf('%-8s | %-11s | K=%-2d | %-8s -> meanIoU=%.4f  stab=%.4f (N=%d)\n', ...
                    nets{nIdx,1}, seg, K, imageTypes{tIdx}, r.MeanIoU, r.MeanStability, r.N);

                rows(end+1, :) = {nets{nIdx,1}, seg, K, imageTypes{tIdx}, ...
                    r.MeanIoU, r.MeanStability, r.N}; %#ok<SAGROW>
            end
        end
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'Model','Segmentation','K','ImageType','MeanIoU','MeanStability','N'});
disp(T);
writetable(T, outputCSV);
fprintf('\nSaved IoU scores to %s\n', outputCSV);


%% ========================================================================
function r = computeLimeIoU(net, cxrFiles, maskFiles, opts)
    inSize = net.Layers(1).InputSize;
    H = inSize(1); W = inSize(2);
    n = numel(cxrFiles);

    perImageIoU  = nan(n, 1);
    perImageStab = nan(n, 1);

    for i = 1:n
        X = prepImage(cxrFiles{i}, inSize);
        B = prepMask(maskFiles{i}, [H W]);
        if opts.Partial                          % lung-only input
            X = X .* uint8(repmat(B, [1 1 size(X,3)]));
        end

        label = classify(net, X);

        runMasks = false(H, W, opts.NumRuns);
        for rr = 1:opts.NumRuns
            [~, fmap, fimp] = imageLIME(net, X, label, ...
                'Segmentation', opts.Segmentation, ...
                'NumFeatures',  opts.NumFeatures, ...
                'NumSamples',   opts.NumSamples, ...
                'Model',        opts.Model);
            k = min(opts.TopK, numel(fimp));
            [~, idx] = maxk(fimp, k);            % plain top-K (recovered behaviour)
            runMasks(:, :, rr) = ismember(fmap, idx);
        end

        ious = zeros(opts.NumRuns, 1);
        for rr = 1:opts.NumRuns
            A = runMasks(:, :, rr);
            ious(rr) = nnz(A & B) / max(nnz(A | B), 1);
        end
        perImageIoU(i) = mean(ious);

        pair = [];
        for a = 1:opts.NumRuns
            for b = a+1:opts.NumRuns
                Aa = runMasks(:,:,a); Bb = runMasks(:,:,b);
                pair(end+1) = nnz(Aa & Bb) / max(nnz(Aa | Bb), 1); %#ok<AGROW>
            end
        end
        if ~isempty(pair), perImageStab(i) = mean(pair); end
    end

    r = struct('MeanIoU', mean(perImageIoU,'omitnan'), ...
               'MeanStability', mean(perImageStab,'omitnan'), ...
               'PerImageIoU', perImageIoU, 'N', n);
end

function X = prepImage(file, inSize)
    X = imread(file);
    if size(X,3) == 1, X = repmat(X, [1 1 3]); end
    X = uint8(imresize(X, inSize(1:2)));
end

function B = prepMask(file, sz)
    M = imread(file);
    if size(M,3) == 3, M = rgb2gray(M); end
    B = imresize(M > 0, sz, 'nearest');
end

function bn = baseNames(files)
    bn = cell(size(files));
    for i = 1:numel(files), [~, bn{i}] = fileparts(files{i}); end
end