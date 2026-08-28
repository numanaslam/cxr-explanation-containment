%% OOD explanation-containment evaluation (paper Phase 4, reframed)
%  Setup: classifier trained on ROI (lung-only) images, tested on FULL CXRs.
%  LIME explanation A is extracted on the full CXR; B is the lung mask.
%  We ask: under this distribution shift, does the model's attention stay
%  INSIDE the lung? The direct answer is PRECISION (containment); IoU and
%  recall are reported alongside for comparability / coverage.
%
%  Outputs:
%    (1) MAIN table  : strategy x K  -> Precision (primary), IoU, Recall, Dice
%    (2) SENSITIVITY : superpixel granularity sweep at fixed K -> shows the
%                      headline metric is stable to segment count (rebuts the
%                      "it's just the mask-area / framing" objection).
%  Also reports the mean lung-area fraction so the metric scale is transparent.
clc; clear; close all;

%% --- paths ---
modelFile  = 'alexnet_v2.mat';
imgDir     = 'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb';  % FULL CXRs (OOD)
maskDir    = 'C:\paper2_repo\input\mask';                               % lung masks
maskSuffix = '_mask';
imgExt     = '*.png';

%% --- run mode ---
TEST_MODE = true;           % true = quick 5-image smoke test; false = full run
if TEST_MODE
    maxImages = 5; nRuns = 2;      % fast: verify the code + sane output
else
    maxImages = Inf; nRuns = 5;    % paper: whole test set, 5 LIME runs/image
end

%% --- params ---
numSamples   = 1000;        % paper: N = 1000
trimBorders  = true;        % remove uniform scanner borders (paper preproc)
contrastNorm = true;        % imadjust (paper preproc)

% MAIN sweep: paper's two strategies x two feature counts
strategies = { ...
    struct('name','superpixel','seg','superpixels','numFeatures',50), ...
    struct('name','grid',      'seg','grid',       'numFeatures',49) };
Kvalues = [30 49];

% SENSITIVITY sweep: superpixel granularity at fixed K (scale-robustness)
sensGranularity = [40 50 75 100];   % total superpixels
sensK           = 30;

%% --- load model ---
S = load(modelFile);
if isfield(S,'netTransfer'), net = S.netTransfer;
else, fn = fieldnames(S); net = S.(fn{1}); end
inputSize = net.Layers(1).InputSize(1:2);

%% --- enumerate images that have a matching mask (sorted = deterministic) ---
files = dir(fullfile(imgDir, imgExt));
assert(~isempty(files), 'No images found in %s', imgDir);
[~, order] = sort({files.name});   % stable order so the 5-image test is repeatable
files = files(order);

%% --- accumulators ---
% MAIN: one entry per (strategy,K). 'lift' = precision - lung_fraction (per image),
% i.e. containment ABOVE the chance level set by how much of the frame is lung.
main = struct('name',{},'K',{},'prec',{},'lift',{},'iou',{},'rec',{},'dice',{});
for s = 1:numel(strategies)
    for kk = 1:numel(Kvalues)
        main(end+1) = struct('name',strategies{s}.name,'K',Kvalues(kk), ...
            'prec',[],'lift',[],'iou',[],'rec',[],'dice',[]); %#ok<AGROW>
    end
end
% SENSITIVITY: one entry per granularity
sens = struct('m',{},'prec',{},'iou',{},'rec',{});
for g = 1:numel(sensGranularity)
    sens(end+1) = struct('m',sensGranularity(g),'prec',[],'iou',[],'rec',[]); %#ok<AGROW>
end
lungFrac = [];   % lung-area fraction per image (transparency)

%% --- main loop over test images ---
nUsed = 0;
for f = 1:numel(files)
    base    = erase(files(f).name, '.png');
    maskPth = fullfile(maskDir, [base maskSuffix '.png']);
    if ~isfile(maskPth), continue; end

    img = imread(fullfile(imgDir, files(f).name));
    if size(img,3)==3, imgGray = rgb2gray(img); else, imgGray = img; end
    gt = imread(maskPth);
    if size(gt,3)==3, gt = rgb2gray(gt); end
    gt = imresize(gt, size(imgGray), 'nearest');

    if trimBorders
        % tight bounding box of ALL non-black content (not just one blob),
        % cropping image and mask identically to keep them co-registered.
        bw = imgGray > (min(imgGray(:)) + 5);
        ys = find(any(bw,2)); xs = find(any(bw,1));
        if ~isempty(ys) && ~isempty(xs)
            imgGray = imgGray(ys(1):ys(end), xs(1):xs(end));
            gt      = gt(ys(1):ys(end), xs(1):xs(end));
        end
    end
    if contrastNorm, imgGray = imadjust(imgGray); end

    imgR   = uint8(repmat(imresize(imgGray, inputSize), 1, 1, 3));
    gtMask = imresize(gt, inputSize, 'nearest') > 0;
    if nnz(gtMask)==0, continue; end
    nUsed = nUsed + 1;
    lungFrac(end+1) = nnz(gtMask) / numel(gtMask); %#ok<AGROW>

    YPred = classify(net, imgR);

    % ---- MAIN sweep ----
    idxMain = 0;
    for s = 1:numel(strategies)
        st = strategies{s};
        for kk = 1:numel(Kvalues)
            idxMain = idxMain + 1;
            K = Kvalues(kk); if K > st.numFeatures, continue; end
            [p,io,rc,dc] = runLIME(net, imgR, YPred, gtMask, st.seg, ...
                st.numFeatures, K, numSamples, nRuns);
            lf = nnz(gtMask) / numel(gtMask);           % chance level for THIS image
            main(idxMain).prec(end+1) = p;  main(idxMain).lift(end+1) = p - lf;
            main(idxMain).iou(end+1)  = io;
            main(idxMain).rec(end+1)  = rc; main(idxMain).dice(end+1) = dc;
        end
    end

    % ---- SENSITIVITY sweep (superpixel granularity at fixed K) ----
    for g = 1:numel(sensGranularity)
        m = sensGranularity(g); if sensK > m, continue; end
        [p,io,rc,~] = runLIME(net, imgR, YPred, gtMask, 'superpixels', ...
            m, sensK, numSamples, nRuns);
        sens(g).prec(end+1) = p; sens(g).iou(end+1) = io; sens(g).rec(end+1) = rc;
    end

    fprintf('  %3d: %s\n', nUsed, base);
    if nUsed >= maxImages, break; end   % TEST_MODE stops after maxImages
end

%% --- report: transparency ---
fprintf('\nMean lung-area fraction of frame: %.3f (min %.3f, max %.3f)\n', ...
    mean(lungFrac), min(lungFrac), max(lungFrac));

%% --- report: MAIN table (Precision first = OOD containment) ---
rows = {};
for c = 1:numel(main)
    if isempty(main(c).prec), continue; end
    lift = main(c).lift(:);
    % one-sided test: is containment significantly ABOVE chance (lift > 0)?
    if numel(lift) > 1 && exist('ttest','file') == 2
        [~, pval] = ttest(lift, 0, 'Tail', 'right');   % needs Statistics Toolbox
    else
        pval = NaN;   % toolbox absent or n<2: report lift only
    end
    rows(end+1,:) = { main(c).name, main(c).K, numel(main(c).prec), ...
        mean(main(c).prec), mean(lift), pval, mean(main(c).iou), ...
        mean(main(c).rec), mean(main(c).dice) }; %#ok<AGROW>
end
Tmain = cell2table(rows, 'VariableNames', ...
    {'Strategy','K','N','Precision','LiftOverChance','pValue','IoU','Recall','Dice'});
fprintf('\n==== OOD containment (primary=Precision), %d images x %d runs ====\n', nUsed, nRuns);
fprintf('LiftOverChance = Precision - lung_area_fraction; pValue tests lift>0.\n');
disp(Tmain);

%% --- report: SENSITIVITY table (scale-robustness) ---
rows = {};
for g = 1:numel(sens)
    if isempty(sens(g).prec), continue; end
    rows(end+1,:) = { sens(g).m, sensK, mean(sens(g).prec), ...
        mean(sens(g).iou), mean(sens(g).rec) }; %#ok<AGROW>
end
Tsens = cell2table(rows, 'VariableNames', ...
    {'NumSuperpixels','K','Precision','IoU','Recall'});
fprintf('\n==== Superpixel-granularity sensitivity at K=%d ====\n', sensK);
disp(Tsens);
fprintf(['(If Precision is roughly flat across NumSuperpixels, the containment\n' ...
         ' result is NOT an artefact of segment count / mask area.)\n']);

%% ======================= local function =======================
function [p,io,rc,dc] = runLIME(net, imgR, YPred, gtMask, seg, m, K, numSamples, nRuns)
    prec = zeros(nRuns,1); iou = zeros(nRuns,1);
    rec  = zeros(nRuns,1); dice = zeros(nRuns,1);
    for r = 1:nRuns
        [~, fMap, fImp] = imageLIME(net, imgR, YPred, ...
            'Segmentation', seg, 'NumFeatures', m, 'NumSamples', numSamples);
        [~, idx] = maxk(fImp, min(K, numel(fImp)));   % clamp: LIME may return <K segs
        A = ismember(fMap, idx);
        inter = nnz(A & gtMask); uni = nnz(A | gtMask);
        prec(r) = inter / max(nnz(A),1);       % containment (OOD fidelity)
        iou(r)  = inter / uni;                 % Jaccard (comparability)
        rec(r)  = inter / max(nnz(gtMask),1);  % coverage
        dice(r) = 2*inter / (nnz(A) + nnz(gtMask));
    end
    p = mean(prec); io = mean(iou); rc = mean(rec); dc = mean(dice);
end
