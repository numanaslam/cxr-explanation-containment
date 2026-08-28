%% Dataset-level LIME vs U-Net lung-mask overlap (paper Phase 4)
%  Reproduces the paper's reported numbers: MEAN over the whole test set of
%  the per-image (5-run) mean IoU, swept over strategy x K.
%
%  Resolves the single-image gap by:
%    (1) looping over ALL test images (paper reports a 132-image mean),
%    (2) trimming uniform borders + contrast-normalising to match the
%        paper's preprocessing (methods, "extraneous markings/borders
%        were cropped out"), which lifts the geometric IoU ceiling that a
%        40%-of-frame lung imposes.
clc; clear; close all;

%% --- paths ---
modelFile = 'alexnet_v2.mat';
imgDir    = 'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb';  % test images
maskDir   = 'C:\paper2_repo\input\mask';                               % lung masks
maskSuffix = '_mask';           % mask file = <base><maskSuffix>.png
imgExt     = '*.png';

%% --- params ---
numSamples = 1000;              % paper: N = 1000
nRuns      = 5;                 % paper: 5 LIME runs per image -> mean IoU
trimBorders   = true;           % remove uniform scanner borders (paper preproc)
contrastNorm  = true;           % imadjust contrast normalisation (paper preproc)

strategies = { ...
    struct('name','superpixel','seg','superpixels','numFeatures',50), ...
    struct('name','grid',      'seg','grid',       'numFeatures',49) };
Kvalues = [30 49];

%% --- load model ---
S = load(modelFile);
if isfield(S,'netTransfer'), net = S.netTransfer;
else, fn = fieldnames(S); net = S.(fn{1}); end
inputSize = net.Layers(1).InputSize(1:2);

%% --- enumerate images that have a matching mask ---
files = dir(fullfile(imgDir, imgExt));
assert(~isempty(files), 'No images found in %s', imgDir);

% accumulators: per-config list of per-image mean IoU (+ dice/prec/rec)
nCfg = numel(strategies)*numel(Kvalues);
acc  = repmat(struct('name','','K',0,'iou',[],'dice',[],'prec',[],'rec',[]), nCfg, 1);
c = 0;
for s = 1:numel(strategies)
    for kk = 1:numel(Kvalues)
        c = c+1;
        acc(c).name = strategies{s}.name; acc(c).K = Kvalues(kk);
    end
end

nUsed = 0;
for f = 1:numel(files)
    base    = erase(files(f).name, '.png');
    maskPth = fullfile(maskDir, [base maskSuffix '.png']);
    if ~isfile(maskPth), continue; end          % skip images without a mask
    nUsed = nUsed + 1;

    %% --- load image + mask, keep them co-registered through preprocessing ---
    img = imread(fullfile(imgDir, files(f).name));
    if size(img,3)==3, imgGray = rgb2gray(img); else, imgGray = img; end
    gt  = imread(maskPth);
    if size(gt,3)==3, gt = rgb2gray(gt); end
    gt  = imresize(gt, size(imgGray), 'nearest');   % align mask to image grid

    if trimBorders
        % bounding box of non-uniform content; crop image AND mask identically
        bw  = imgGray > (min(imgGray(:)) + 5);      % drop the flat black margin
        bb  = regionprops(bw, 'BoundingBox');
        if ~isempty(bb)
            r = round(bb(1).BoundingBox);
            r(3:4) = max(r(3:4), 1);
            imgGray = imcrop(imgGray, r);
            gt      = imcrop(gt, r);
        end
    end
    if contrastNorm
        imgGray = imadjust(imgGray);
    end

    imgR = imresize(imgGray, inputSize);
    imgR = uint8(repmat(imgR, 1, 1, 3));            % gray -> rgb to match training
    gtMask = imresize(gt, inputSize, 'nearest') > 0;
    if nnz(gtMask)==0, continue; end

    YPred = classify(net, imgR);

    %% --- sweep configs, 5-run mean IoU per config ---
    c = 0;
    for s = 1:numel(strategies)
        st = strategies{s};
        for kk = 1:numel(Kvalues)
            c = c+1;
            K = Kvalues(kk);
            if K > st.numFeatures, continue; end

            iou = zeros(nRuns,1); dice = zeros(nRuns,1);
            prec = zeros(nRuns,1); rec = zeros(nRuns,1);
            for r = 1:nRuns
                [~, fMap, fImp] = imageLIME(net, imgR, YPred, ...
                    'Segmentation', st.seg, 'NumFeatures', st.numFeatures, ...
                    'NumSamples', numSamples);
                [~, idx] = maxk(fImp, K);
                A = ismember(fMap, idx);
                inter = nnz(A & gtMask); uni = nnz(A | gtMask);
                iou(r)  = inter / uni;
                dice(r) = 2*inter / (nnz(A) + nnz(gtMask));
                prec(r) = inter / max(nnz(A),1);
                rec(r)  = inter / max(nnz(gtMask),1);
            end
            acc(c).iou(end+1)  = mean(iou);
            acc(c).dice(end+1) = mean(dice);
            acc(c).prec(end+1) = mean(prec);
            acc(c).rec(end+1)  = mean(rec);
        end
    end
    fprintf('  processed %3d/%3d: %s\n', nUsed, numel(files), base);
end

%% --- final table: mean of per-image mean IoU (the paper's reported value) ---
rows = {};
for c = 1:numel(acc)
    if isempty(acc(c).iou), continue; end
    rows(end+1,:) = { acc(c).name, acc(c).K, numel(acc(c).iou), ...
        mean(acc(c).iou), std(acc(c).iou), mean(acc(c).dice), ...
        mean(acc(c).prec), mean(acc(c).rec) }; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', ...
    {'Strategy','K','N','MeanIoU','StdIoU','Dice','Precision','Recall'});
fprintf('\n==== Test-set mean IoU (%d images, %d runs each) ====\n', nUsed, nRuns);
disp(T);
