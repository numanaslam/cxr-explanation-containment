%% LIME vs U-Net lung mask -- paper-aligned overlap (Jaccard / IoU)
%  Phase 4:  J(A',B) = |A' n B| / |A' u B|
%    A' = LIME explanation mask (top-K influential segments)
%    B  = lung ROI mask (U-Net / ground truth)
%  Reports MEAN IoU over 5 stochastic LIME runs, sweeping strategy x K.
clc; clear; close all;

%% --- paths & fixed params ---
modelFile    = 'alexnet_v2.mat';
imgFile      = 'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\CHNCXR_0327_1.png';
unetMaskFile = 'C:\paper2_repo\input\mask\CHNCXR_0327_1_mask.png';

numSamples = 1000;   % paper: N = 1000 perturbations per explanation
nRuns      = 5;      % paper: 5 LIME runs per image, report the MEAN IoU

strategies = { ...
    struct('name','superpixel','seg','superpixels','numFeatures',50), ...
    struct('name','grid',      'seg','grid',       'numFeatures',49) };
Kvalues = [30 49];

vizStrategy = 'superpixel';   % which config the 3 figures are drawn for
vizK        = 30;

%% --- load model ---
S = load(modelFile);
if isfield(S,'netTransfer'), net = S.netTransfer;
else, fn = fieldnames(S); net = S.(fn{1}); end
inputSize = net.Layers(1).InputSize(1:2);

%% --- read & prep image (gray -> rgb to match training) ---
img = imread(imgFile);
img = imresize(img, inputSize);
if size(img,3)==1, img = repmat(img,1,1,3); end
img = uint8(img);
YPred = classify(net, img);

%% --- U-Net / ground-truth lung mask (B) ---
gt = imread(unetMaskFile);
if size(gt,3)==3, gt = rgb2gray(gt); end
gt = imresize(gt, inputSize, 'nearest');
gtMask = gt > 0;              % use gt > 127 if the mask is soft/greyscale

%% --- sweep: strategy x K, each = mean over nRuns ---
rows = {};
for s = 1:numel(strategies)
    st = strategies{s};
    for kk = 1:numel(Kvalues)
        K = Kvalues(kk);
        if K > st.numFeatures
            warning('K=%d > numFeatures=%d for %s; skipping.', K, st.numFeatures, st.name);
            continue;
        end

        iou = zeros(nRuns,1); dice = zeros(nRuns,1);
        prec = zeros(nRuns,1); rec = zeros(nRuns,1);
        masks = cell(nRuns,1);

        for r = 1:nRuns
            [~, featureMap, featureImportance] = imageLIME(net, img, YPred, ...
                'Segmentation', st.seg, ...
                'NumFeatures',  st.numFeatures, ...
                'NumSamples',   numSamples);

            [~, idx] = maxk(featureImportance, K);
            A = ismember(featureMap, idx);        % LIME mask A' (top-K)
            masks{r} = A;

            inter   = nnz(A & gtMask);
            uni     = nnz(A | gtMask);
            iou(r)  = inter / uni;                       % Jaccard / IoU
            dice(r) = 2*inter / (nnz(A) + nnz(gtMask));
            prec(r) = inter / max(nnz(A),1);             % LIME coverage inside lung
            rec(r)  = inter / max(nnz(gtMask),1);        % lung covered by LIME
        end

        % stability: mean pairwise IoU between the nRuns LIME masks
        pw = [];
        for a = 1:nRuns-1
            for b = a+1:nRuns
                ii = nnz(masks{a} & masks{b});
                uu = nnz(masks{a} | masks{b});
                pw(end+1) = ii / uu; %#ok<AGROW>
            end
        end
        stability = mean(pw);

        rows(end+1,:) = { st.name, K, mean(iou), std(iou), ...
            mean(dice), mean(prec), mean(rec), stability }; %#ok<AGROW>

        fprintf(['[%-10s | K=%2d]  meanIoU=%.4f (std %.4f)  Dice=%.4f  ' ...
                 'Prec=%.4f  Rec=%.4f  Stability=%.4f\n'], ...
            st.name, K, mean(iou), std(iou), mean(dice), ...
            mean(prec), mean(rec), stability);
    end
end

%% --- results table (paper-style) ---
T = cell2table(rows, 'VariableNames', ...
    {'Strategy','K','MeanIoU','StdIoU','Dice','Precision','Recall','Stability'});
fprintf('\n==== Mean IoU over %d LIME runs (class: %s) ====\n', nRuns, string(YPred));
disp(T);

%% --- figures for the chosen viz config (one representative run) ---
vs = strategies{find(cellfun(@(c) strcmp(c.name,vizStrategy), strategies),1)};
[map, featureMap, featureImportance] = imageLIME(net, img, YPred, ...
    'Segmentation', vs.seg, 'NumFeatures', vs.numFeatures, 'NumSamples', numSamples);
[~, idx]  = maxk(featureImportance, vizK);
limeMask  = ismember(featureMap, idx);
maskedImg = uint8(limeMask) .* img;

sel      = strcmp(T.Strategy, vizStrategy) & T.K == vizK;
meanIoU  = T.MeanIoU(sel);
inter = nnz(limeMask & gtMask); uni = nnz(limeMask | gtMask);
runIoU = inter / uni;

figure; imshow(img,'InitialMagnification',150); hold on;
imagesc(map,'AlphaData',0.5); colormap jet; colorbar;
title(sprintf('Image LIME (%s) - %s', YPred, vizStrategy)); hold off;

figure; imshow(maskedImg);
title(sprintf('%s, top %d feats | mean IoU=%.3f (this run %.3f)', ...
    YPred, vizK, meanIoU, runIoU));

figure; imshow(img); hold on;
red = cat(3, ones(inputSize), zeros(inputSize), zeros(inputSize));
grn = cat(3, zeros(inputSize), ones(inputSize), zeros(inputSize));
h1 = imshow(red); set(h1,'AlphaData', 0.35*limeMask);
h2 = imshow(grn); set(h2,'AlphaData', 0.35*gtMask);
title(sprintf('%s K=%d | mean IoU=%.3f', vizStrategy, vizK, meanIoU));
hold off;