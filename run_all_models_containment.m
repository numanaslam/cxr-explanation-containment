%% OOD explanation-containment: 3 models x {LIME superpixel, LIME grid, Grad-CAM}
%  Setup: classifiers trained on ROI (lung-only) images, tested OOD on FULL CXRs.
%  For each explanation mask A and lung mask B we report, per image, the
%  containment metrics and their MEAN over the set:
%     Precision = |A n B| / |A|        (containment  <-- primary OOD fidelity)
%     LiftOverChance = Precision - lung_area_fraction   (Precision above chance)
%     IoU, Recall, Dice                (comparability / coverage)
%  A one-sided t-test asks whether LiftOverChance > 0.
%
%  Explainers:
%     LIME (superpixel & grid, top-K)  -- stochastic, averaged over nRuns
%     Grad-CAM (normalised heatmap, thresholded) -- deterministic, 1 run
%
%  Runs on a MATLAB parallel pool if the Parallel Computing Toolbox is present.
clc; clear; close all;

%% ================== run mode ==================
TEST_MODE = false;           % true = quick 5-image smoke test; false = FULL run
if TEST_MODE
    maxImages = 5;  nRuns = 2;
else
    maxImages = Inf; nRuns = 5;   % paper: whole set, 5 LIME runs / image
    % RUNTIME: LIME dominates. To keep the full run tractable we (a) run the
    % heavy LIME sweep ONLY in-distribution (ROI), and (b) on OOD run just the
    % cheap deterministic Grad-CAM + always-computed accuracy (see configKeys).
    % If ROI is still too slow on gpu-serial, drop nRuns to 3.
end
resultsDir = 'C:\paper2_repo\results';   % CSVs saved here (crash-safe, incremental)
if ~exist(resultsDir,'dir'), mkdir(resultsDir); end

%% ================== models ==================
%  MODELS_TO_RUN lets you add a model without recomputing (or losing) the others:
%  the CSVs are MERGED on save, so rows for models you skip are carried forward.
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
models    = allModels;    % <-- set to {'resnet50_v2.mat'} to add ResNet50 only
abnormalClass = '';   % exact abnormal class label (e.g. 'ptb'/'abnormal'/'1').
                      % '' = auto-detect from label name (verify the printed classes!).

%% ================== test conditions & paths ==================
%  BOTH are evaluated every run:
%    ROI     = in-distribution control (model trained on these) -> expect containment
%    FullCXR = OOD test (unseen full radiographs)               -> the drift test
%  scaleMatch=true crops each full CXR to the lung bbox + margin, restoring the
%  lung to ~training scale while keeping surrounding anatomy -> isolates SCALE
%  from CONTEXT, so we can tell if the OOD collapse is scale or genuine drift.
%  configKeys = which explainer configs to run for that condition ({} = ALL).
%  Accuracy is ALWAYS computed (just needs classify), regardless of configKeys.
%  Rationale: the paper's spine is ROI containment (heavy LIME sweep there);
%  OOD is now a classification-failure + collapse control, so we only run the
%  cheap deterministic Grad-CAM on OOD (LIME there is expensive and the collapse
%  is already established) — this makes the full run tractable.
ROI_CFGS = {'LIME-sp-K30','LIME-sp-K49','LIME-grid-K30','LIME-grid-K49','LIME-fine-m100-K20','GradCAM@0.5'};
OOD_CFGS = {'GradCAM@0.5','GradCAM@0.7'};
conditions = struct('name',{},'imgDir',{},'recursive',{},'classFrom',{},'scaleMatch',{},'marginFrac',{},'configKeys',{});
conditions(end+1) = struct('name','ROI', ...
    'imgDir','C:\paper2_repo\input\annotated_gray\annotated_gray', ...
    'recursive',true,  'classFrom','folder',  'scaleMatch',false,'marginFrac',0,   'configKeys',{ROI_CFGS}); % in-distribution
conditions(end+1) = struct('name','FullCXR', ...
    'imgDir','C:\paper2_repo\input\cxr_resized', ...
    'recursive',false, 'classFrom','filename','scaleMatch',false,'marginFrac',0,   'configKeys',{OOD_CFGS}); % OOD, native scale
conditions(end+1) = struct('name','FullCXR-scaled', ...
    'imgDir','C:\paper2_repo\input\cxr_resized', ...
    'recursive',false, 'classFrom','filename','scaleMatch',true, 'marginFrac',0.25,'configKeys',{OOD_CFGS}); % OOD, ~training scale
maskDir    = 'C:\paper2_repo\input\mask';
maskSuffix = '_mask';

%% ================== explainer / metric params ==================
numSamples = 1000;           % LIME: N perturbations
workSize   = [512 512];      % images + masks resized to this (align + standardise)

%% ---- preprocessing preset (MATCH your training pipeline) ----
%  Our default eval gave ROI accuracy ~0.74 vs the paper's Table clsperf ~0.85-0.91,
%  which means the eval preprocessing did NOT match how the models were trained.
%  Sweep PREPROC (A/B/C/D) and watch the "[reconcile ...]" line printed after each
%  ROI condition: pick the preset whose ROI accuracy MATCHES Table clsperf. Then the
%  containment and OOD-accuracy from that same run are consistent with the classifier.
PREPROC = 'A';
switch PREPROC
    case 'A', trimBorders = true;  contrastNorm = true;   % paper text: crop borders + imadjust
    case 'B', trimBorders = false; contrastNorm = false;  % minimal: resize + gray->rgb only
    case 'C', trimBorders = false; contrastNorm = true;   % imadjust only
    case 'D', trimBorders = true;  contrastNorm = false;  % border crop only
    otherwise, error('Unknown PREPROC preset: %s', PREPROC);
end

% Target in-distribution ROI accuracies from Table clsperf (for the reconciliation
% check). Field names must match modelName = erase(modelFile,'.mat').
%  ResNet50 is trained fresh by train_resnet50.m -> paste its held-out accuracy here
%  (NaN = no target yet, reconciliation check is skipped for that model).
clsperfAcc = struct('alexnet_v2',0.805, 'vgg16_v2',0.885, 'vgg19_v2',0.867, ...
                    'resnet50_v2',0.814);
reconTol   = 0.03;   % |measured ROI acc - target| within this => MATCH

% Config list (uniform struct array so the per-image loop is generic).
% 'thr' is the Grad-CAM threshold (unused for LIME).
% Coarse configs (K/m ~ 0.6 of frame) work when the lung is large (ROI).
% FINE configs keep K/m ~ 0.13-0.20 so the mask can NEST inside the small
% (~23% of frame) lung on full CXRs -> gives the OOD test statistical power.
cfgs = struct('key',{},'type',{},'seg',{},'m',{},'K',{},'thr',{});
cfgs(end+1) = struct('key','LIME-sp-K30',   'type','lime','seg','superpixels','m',50, 'K',30,'thr',0);
cfgs(end+1) = struct('key','LIME-sp-K49',   'type','lime','seg','superpixels','m',50, 'K',49,'thr',0);
cfgs(end+1) = struct('key','LIME-grid-K30', 'type','lime','seg','grid',       'm',49, 'K',30,'thr',0);
cfgs(end+1) = struct('key','LIME-grid-K49', 'type','lime','seg','grid',       'm',49, 'K',49,'thr',0);
cfgs(end+1) = struct('key','LIME-fine-m100-K20','type','lime','seg','superpixels','m',100,'K',20,'thr',0); % ~20% mask
cfgs(end+1) = struct('key','LIME-fine-m150-K25','type','lime','seg','superpixels','m',150,'K',25,'thr',0); % ~17% mask
cfgs(end+1) = struct('key','LIME-fine-m200-K30','type','lime','seg','superpixels','m',200,'K',30,'thr',0); % ~15% mask
cfgs(end+1) = struct('key','GradCAM@0.5',   'type','gradcam','seg','','m',0,'K',0,'thr',0.5);
cfgs(end+1) = struct('key','GradCAM@0.7',   'type','gradcam','seg','','m',0,'K',0,'thr',0.7); % smaller, tighter mask
nCfg = numel(cfgs);

%% ================== compute mode ==================
%  IMPORTANT: multiple parfor workers sharing ONE GPU causes
%  CUDA_ERROR_ILLEGAL_ADDRESS (it killed VGG16). So:
%    'gpu-serial'  = 1 process on the GPU  (SAFE + fast with a single GPU) <-- default
%    'cpu-parallel'= worker pool on the CPU (parallel but slower; no GPU contention)
%    'serial'      = 1 process, auto device
COMPUTE    = 'gpu-serial';
numWorkers = 4;             % only used for 'cpu-parallel'
switch COMPUTE
    case 'gpu-serial',   USE_PARALLEL = false; execEnv = 'gpu';
    case 'cpu-parallel', USE_PARALLEL = true;  execEnv = 'cpu';
    otherwise,           USE_PARALLEL = false; execEnv = 'auto';
end
if USE_PARALLEL
    if license('test','Distrib_Computing_Toolbox')
        pp = gcp('nocreate'); if isempty(pp), parpool(numWorkers); end
        fprintf('Parallel pool active (%d workers, execEnv=%s).\n', gcp().NumWorkers, execEnv);
    else
        USE_PARALLEL = false; fprintf('PCT not found -> serial (execEnv=%s).\n', execEnv);
    end
else
    fprintf('Serial execution (execEnv=%s).\n', execEnv);
end

%% ================== enumerate images for EACH condition (once) ==================
%  Only image/mask PAIRS are used; unpaired images are skipped. The scanned-vs-
%  paired print below reveals if mask naming is silently dropping images.
for k = 1:numel(conditions)
    [ip, mpz, ab, nsc] = enumerateSet(conditions(k).imgDir, conditions(k).recursive, ...
        conditions(k).classFrom, maskDir, maskSuffix, maxImages);
    conditions(k).imgPaths = ip; conditions(k).maskPaths = mpz; conditions(k).isAbn = ab;
    assert(~isempty(ip), 'No image/mask pairs for condition %s (%s).', ...
        conditions(k).name, conditions(k).imgDir);
    fprintf('Condition=%-8s | scanned %d, PAIRED %d (%d abnormal, %d normal) from %s\n', ...
        conditions(k).name, nsc, numel(ip), nnz(ab), nnz(~ab), conditions(k).imgDir);
    if isfinite(maxImages) == false && nsc > 0 && numel(ip) < 0.5*nsc
        warning(['Only %d of %d images had a matching mask in %s -- ' ...
                 'check mask naming if you expected more.'], numel(ip), nsc, maskDir);
    end
end

%% ================== main: models x conditions ==================
master = {};   % combined comparison rows (config-level containment)
accRows = {};  % classification accuracy rows (condition-level)
for mi = 1:numel(models)
    if ~isfile(models{mi})
        warning('Missing %s -- skipping. (Run train_resnet50.m first?)', models{mi});
        continue;
    end
    S = load(models{mi});
    if isfield(S,'netTransfer'), net = S.netTransfer;
    else, fn = fieldnames(S); net = S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    modelName = erase(models{mi}, '.mat');
    fprintf('\n########## MODEL: %s  (input %dx%d) ##########\n', modelName, inputSize(1), inputSize(2));
    try
        fprintf('  classes: %s  (abnormal auto-map: verify!)\n', strjoin(string(net.Layers(end).Classes), ', '));
    catch, end
    if USE_PARALLEL, netC = parallel.pool.Constant(net); end

    for k = 1:numel(conditions)
        cond = conditions(k);
        ip = cond.imgPaths; mpz = cond.maskPaths; ab = cond.isAbn; nImg = numel(ip);
        if nImg == 0, warning('No pairs for %s / %s.', modelName, cond.name); continue; end
        fprintf('---- condition: %s (%d images) ----\n', cond.name, nImg);

        template = struct('ok',false,'lungFrac',0,'isAbnormal',false, ...
            'prec',zeros(nCfg,1),'iou',zeros(nCfg,1),'rec',zeros(nCfg,1), ...
            'dice',zeros(nCfg,1),'lift',zeros(nCfg,1),'predAbn',false,'trueAbn',false);
        res = repmat(template, nImg, 1);
        sm = cond.scaleMatch; mf = cond.marginFrac;
        % which explainer configs to run for THIS condition ({} = all)
        if isempty(cond.configKeys), runMask = true(1,nCfg);
        else, runMask = ismember({cfgs.key}, cond.configKeys); end

        if USE_PARALLEL
            parfor i = 1:nImg
                res(i) = processImage(netC.Value, ip{i}, mpz{i}, ab(i), ...
                    cfgs, inputSize, numSamples, nRuns, trimBorders, contrastNorm, workSize, ...
                    execEnv, sm, mf, abnormalClass, runMask);
            end
        else
            for i = 1:nImg
                res(i) = processImage(net, ip{i}, mpz{i}, ab(i), ...
                    cfgs, inputSize, numSamples, nRuns, trimBorders, contrastNorm, workSize, ...
                    execEnv, sm, mf, abnormalClass, runMask);
            end
        end

        % ---- aggregate ----
        ok = [res.ok];
        if ~any(ok), warning('No usable images for %s / %s.', modelName, cond.name); continue; end
        P = [res(ok).prec]; IO = [res(ok).iou]; RC = [res(ok).rec];
        DC = [res(ok).dice]; L = [res(ok).lift]; ABN = [res(ok).isAbnormal];
        lf = [res(ok).lungFrac];
        fprintf('Usable %d/%d | mean lung fraction %.3f\n', nnz(ok), nImg, mean(lf));

        % ---- classification accuracy (is the OOD collapse "drift" or "broken"?) ----
        PA = logical([res(ok).predAbn]); TA = logical([res(ok).trueAbn]);
        acc  = mean(PA == TA);
        if nnz(TA)  > 0, sens = mean(PA(TA));   else, sens = NaN; end
        if nnz(~TA) > 0, spec = mean(~PA(~TA)); else, spec = NaN; end
        fprintf('Classification: acc=%.3f  sens=%.3f  spec=%.3f\n', acc, sens, spec);
        accRows(end+1,:) = { modelName, cond.name, nnz(ok), acc, sens, spec }; %#ok<SAGROW>

        % ---- reconciliation check: does ROI accuracy match Table clsperf? ----
        if strcmp(cond.name,'ROI')
            fld = matlab.lang.makeValidName(modelName);
            if isfield(clsperfAcc, fld) && ~isnan(clsperfAcc.(fld))
                tgt = clsperfAcc.(fld); d = acc - tgt;
                if abs(d) <= reconTol, verdict = 'MATCH';
                else,                  verdict = 'MISMATCH'; end
                fprintf('  [reconcile ROI acc vs Table clsperf: measured %.3f, target %.3f, delta %+.3f -> %s (PREPROC=%s)]\n', ...
                    acc, tgt, d, verdict, PREPROC);
                if abs(d) > reconTol
                    fprintf('  -> preprocessing differs from training; try another PREPROC preset before trusting the numbers.\n');
                end
            end
        end

        rows = {};
        for c = 1:nCfg
            if all(isnan(P(c,:))), continue; end   % config not run for this condition
            lift_c = L(c,:);
            if numel(lift_c) > 1 && exist('ttest','file')==2
                [~, pv] = ttest(lift_c, 0, 'Tail','right');
            else, pv = NaN; end
            pAbn = mean(P(c, ABN==1)); pNor = mean(P(c, ABN==0));
            rows(end+1,:) = { cfgs(c).key, mean(P(c,:)), mean(lift_c), pv, ...
                mean(IO(c,:)), mean(RC(c,:)), mean(DC(c,:)), pNor, pAbn }; %#ok<SAGROW>
            master(end+1,:) = { modelName, cond.name, cfgs(c).key, mean(P(c,:)), ...
                mean(lift_c), pv, mean(IO(c,:)) }; %#ok<SAGROW>
        end
        if ~isempty(rows)
            T = cell2table(rows, 'VariableNames', ...
                {'Config','Precision','LiftOverChance','pValue','IoU','Recall','Dice','Prec_normal','Prec_abnormal'});
            disp(T);
        end
    end

    % ---- crash-safe incremental save after each model ----
    %  MERGED with any previous run: rows for models in THIS run are replaced,
    %  rows for models not run (e.g. you only re-ran ResNet50) are preserved.
    try
        if ~isempty(master)
            mergeSave(fullfile(resultsDir,'containment_combined.csv'), ...
                cell2table(master, 'VariableNames', ...
                {'Model','Condition','Config','Precision','LiftOverChance','pValue','IoU'}));
        end
        if ~isempty(accRows)
            mergeSave(fullfile(resultsDir,'classification_accuracy.csv'), ...
                cell2table(accRows, 'VariableNames', ...
                {'Model','Condition','N','Accuracy','Sensitivity','Specificity'}));
        end
        fprintf('[saved cumulative results to %s]\n', resultsDir);
    catch ME
        warning('CSV save failed: %s', ME.message);
    end
end

%% ================== combined comparison (all models x conditions) ==================
Tm = cell2table(master, 'VariableNames', ...
    {'Model','Condition','Config','Precision','LiftOverChance','pValue','IoU'});
fprintf('\n================= COMBINED COMPARISON =================\n');
fprintf('(<=%d images/condition, nRuns=%d) primary = LiftOverChance > 0\n', maxImages, nRuns);
disp(Tm);

%% ================== classification accuracy (the drift-vs-broken control) ==================
Ta = cell2table(accRows, 'VariableNames', ...
    {'Model','Condition','N','Accuracy','Sensitivity','Specificity'});
fprintf('\n============ CLASSIFICATION ACCURACY ============\n');
disp(Ta);
% Interpretation guide:
%  FullCXR: if accuracy stays HIGH but containment ~chance  -> genuine attention
%           drift (right answer, non-lung reasons = shortcut). Interesting.
%           if accuracy also COLLAPSES                       -> model out-of-regime;
%           containment result is trivial, not "drift".
%  FullCXR-scaled vs FullCXR: if containment RECOVERS when the lung is put back at
%           training scale, the collapse was SCALE, not attention.

%% ======================= local functions =======================
function mergeSave(csvPath, Tnew)
    % Replace rows for the models present in Tnew; keep every other model's rows
    % from the previous run. Lets you add ResNet50 without re-running the rest.
    Tnew = textToString(Tnew);
    if isfile(csvPath)
        Told = textToString(readtable(csvPath, 'TextType','string'));
        sameCols = isequal(sort(Told.Properties.VariableNames), sort(Tnew.Properties.VariableNames));
        if ismember('Model', Told.Properties.VariableNames) && sameCols
            Told = Told(:, Tnew.Properties.VariableNames);          % match column order
            Told(ismember(Told.Model, unique(Tnew.Model)), :) = [];
            if height(Told) > 0, Tnew = [Told; Tnew]; end
        else
            warning('%s has an unexpected schema; overwriting instead of merging.', csvPath);
        end
    end
    writetable(Tnew, csvPath);
end

function T = textToString(T)
    % Normalise char/cellstr columns to string so vertcat of an old CSV and a new
    % cell2table result cannot fail on type mismatch.
    for v = T.Properties.VariableNames
        c = T.(v{1});
        if ischar(c) || iscellstr(c) || isstring(c), T.(v{1}) = string(c); end %#ok<ISCLSTR>
    end
end

function ab = predLabelIsAbnormal(lab, abnormalClass)
    s = lower(strtrim(char(string(lab))));
    if ~isempty(abnormalClass), ab = strcmpi(s, lower(strtrim(abnormalClass))); return; end
    if any(strcmp(s, {'0','normal','healthy','neg','negative','control'})), ab = false; return; end
    if contains(s,'normal') || contains(s,'health'), ab = false; return; end
    ab = true;   % anything not clearly "normal" is treated as abnormal
end
function [imgPaths, maskPaths, isAbn, nScanned] = enumerateSet(imgDir, recursive, classFrom, maskDir, maskSuffix, maxImages)
    if recursive, files = dir(fullfile(imgDir, '**', '*.png'));
    else,         files = dir(fullfile(imgDir, '*.png')); end
    files = files(~[files.isdir]);
    [~, ord] = sort({files.name}); files = files(ord);
    nScanned = numel(files);
    imgPaths = {}; maskPaths = {}; isAbn = [];
    for f = 1:numel(files)               % collect ALL pairs first (no cap yet)
        base = erase(files(f).name, '.png');
        mp   = fullfile(maskDir, [base maskSuffix '.png']);
        if ~isfile(mp)
            mp2 = fullfile(maskDir, [base '.png']);
            if isfile(mp2), mp = mp2; else, continue; end
        end
        if strcmp(classFrom, 'folder')
            [~, parent] = fileparts(files(f).folder);
            ab = ~contains(lower(parent), 'normal');
        else
            toks = split(base, '_'); ab = strcmp(toks{end}, '1');
        end
        imgPaths{end+1}  = fullfile(files(f).folder, files(f).name); %#ok<AGROW>
        maskPaths{end+1} = mp;   %#ok<AGROW>
        isAbn(end+1)     = ab;   %#ok<AGROW>
    end
    % class-stratified cap: interleave normal/abnormal so a small maxImages
    % sample contains BOTH classes (else sorted order gives all-normal).
    if isfinite(maxImages) && numel(imgPaths) > maxImages
        idxN = find(~isAbn); idxA = find(isAbn); order = [];
        for t = 1:max(numel(idxN), numel(idxA))
            if t <= numel(idxN), order(end+1) = idxN(t); end %#ok<AGROW>
            if t <= numel(idxA), order(end+1) = idxA(t); end %#ok<AGROW>
        end
        order = order(1:maxImages);
        imgPaths = imgPaths(order); maskPaths = maskPaths(order); isAbn = isAbn(order);
    end
end

function out = processImage(net, imgPath, maskPath, isAbn, cfgs, inputSize, ...
                            numSamples, nRuns, trimBorders, contrastNorm, workSize, ...
                            execEnv, scaleMatch, marginFrac, abnormalClass, runMask)
    nCfg = numel(cfgs);
    out = struct('ok',false,'lungFrac',0,'isAbnormal',logical(isAbn), ...
        'prec',nan(nCfg,1),'iou',nan(nCfg,1),'rec',nan(nCfg,1), ...
        'dice',nan(nCfg,1),'lift',nan(nCfg,1),'predAbn',false,'trueAbn',logical(isAbn));
    if ~isfile(maskPath) || ~isfile(imgPath), return; end

    img = imread(imgPath); if size(img,3)==3, g = rgb2gray(img); else, g = img; end
    gt  = imread(maskPath); if size(gt,3)==3, gt = rgb2gray(gt); end
    g   = imresize(g,  workSize);              % standardise + co-register image & mask
    gt  = imresize(gt, workSize, 'nearest');

    if scaleMatch                              % crop to lung bbox + margin -> ~training scale
        [ys0, xs0] = find(gt > 0);
        if ~isempty(ys0)
            r0=min(ys0); r1=max(ys0); c0=min(xs0); c1=max(xs0);
            mr=round(marginFrac*(r1-r0+1)); mc=round(marginFrac*(c1-c0+1));
            r0=max(1,r0-mr); r1=min(size(g,1),r1+mr);
            c0=max(1,c0-mc); c1=min(size(g,2),c1+mc);
            g  = g(r0:r1, c0:c1);
            gt = gt(r0:r1, c0:c1);
        end
    end

    if trimBorders
        bw = g > (min(g(:)) + 5);
        ys = find(any(bw,2)); xs = find(any(bw,1));
        if ~isempty(ys) && ~isempty(xs)
            g  = g(ys(1):ys(end), xs(1):xs(end));
            gt = gt(ys(1):ys(end), xs(1):xs(end));
        end
    end
    if contrastNorm, g = imadjust(g); end

    imgR   = uint8(repmat(imresize(g, inputSize), 1, 1, 3));
    gtMask = imresize(gt, inputSize, 'nearest') > 0;
    if nnz(gtMask) == 0, return; end
    lf = nnz(gtMask) / numel(gtMask);

    YPred = classify(net, imgR);
    out.predAbn = predLabelIsAbnormal(YPred, abnormalClass);

    types = {cfgs.type};
    % ---- LIME configs, grouped by (seg,m): one imageLIME serves every K that
    %      shares the same segmentation (so K=30 and K=49 don't recompute it) ----
    limeC = find(runMask(:)' & strcmp(types,'lime'));
    handled = false(1,nCfg);
    for c = limeC
        if handled(c), continue; end
        grp = limeC( strcmp({cfgs(limeC).seg}, cfgs(c).seg) & ([cfgs(limeC).m] == cfgs(c).m) );
        acc = zeros(numel(grp), 4);            % [prec iou rec dice] summed over runs
        for r = 1:nRuns
            [~, fMap, fImp] = imageLIME(net, imgR, YPred, ...
                'Segmentation', cfgs(c).seg, 'NumFeatures', cfgs(c).m, ...
                'NumSamples', numSamples, 'ExecutionEnvironment', execEnv);
            for gi = 1:numel(grp)
                Kc = cfgs(grp(gi)).K;
                [~, idx] = maxk(fImp, min(Kc, numel(fImp)));
                A = ismember(fMap, idx);
                [pp, ii, rr, dd] = maskMetrics(A, gtMask);
                acc(gi,:) = acc(gi,:) + [pp ii rr dd];
            end
        end
        acc = acc / nRuns;
        for gi = 1:numel(grp)
            cc = grp(gi);
            out.prec(cc)=acc(gi,1); out.iou(cc)=acc(gi,2); out.rec(cc)=acc(gi,3); out.dice(cc)=acc(gi,4);
            out.lift(cc)=out.prec(cc)-lf; handled(cc)=true;
        end
    end
    % ---- Grad-CAM configs (deterministic, one pass each) ----
    for c = find(runMask(:)' & strcmp(types,'gradcam'))
        A = gradcamMask(net, imgR, YPred, inputSize, cfgs(c).thr, execEnv);
        [pp, ii, rr, dd] = maskMetrics(A, gtMask);
        out.prec(c)=pp; out.iou(c)=ii; out.rec(c)=rr; out.dice(c)=dd; out.lift(c)=pp-lf;
    end
    out.lungFrac = lf; out.ok = true;
end

function [p, io, rc, dc] = maskMetrics(A, B)
    inter = nnz(A & B); uni = nnz(A | B);
    p  = inter / max(nnz(A), 1);         % containment / precision
    io = inter / max(uni, 1);            % IoU
    rc = inter / max(nnz(B), 1);         % recall / coverage
    dc = 2*inter / max(nnz(A)+nnz(B), 1);% Dice
end

function A = gradcamMask(net, imgR, YPred, inputSize, thr, execEnv)
    try
        m = gradCAM(net, imgR, YPred, 'ExecutionEnvironment', execEnv);
    catch
        % Auto-selection is reliable for SeriesNetworks (AlexNet/VGG) but can fail
        % on branched DAGs like ResNet50 -> retry with the last ReLU as the
        % feature layer (resnet50: activation_49_relu) before giving up.
        m = [];
        try
            L = net.Layers; fl = '';
            for i = numel(L):-1:1
                if isa(L(i),'nnet.cnn.layer.ReLULayer'), fl = L(i).Name; break; end
            end
            if ~isempty(fl)
                m = gradCAM(net, imgR, YPred, 'ExecutionEnvironment', execEnv, ...
                            'FeatureLayer', fl);
            end
        catch
        end
        if isempty(m), A = false(inputSize); return; end
    end
    m = imresize(double(m), inputSize);
    mn = min(m(:)); mx = max(m(:));
    if mx > mn, m = (m - mn) / (mx - mn); else, m = zeros(size(m)); end
    A = m >= thr;
end
