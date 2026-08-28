%% Regenerate tab:contain with MINIMAL preprocessing (resize + gray->rgb only).
%  ROI (in-distribution) containment = lift over chance = precision - lung-area fraction.
%  Prints LaTeX-ready rows AND writes results\containment_combined.csv so
%  make_results_charts.m regenerates the containment chart consistently.
%
%  ADDING A MODEL WITHOUT LOSING THE OLD NUMBERS:
%  set MODELS_TO_RUN = {'resnet50_v2.mat'} to compute ONLY ResNet50. The CSV is
%  MERGED, not overwritten -- rows for models you did not run are carried over
%  from the previous run, so the existing AlexNet/VGG16/VGG19 numbers survive and
%  the LaTeX table below still prints all four models.
%
%  COMPARABILITY: the 132-image subset is enumerated deterministically (sorted +
%  class-interleaved), so every model is scored on the SAME images regardless of
%  which subset you run. Some of those images are in ResNet50's training split;
%  per ADR-005 that does not bias containment, which is a localization measure.
clc; clear; close all;

%% --- config ---
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
MODELS_TO_RUN = allModels;      % <-- set to {'resnet50_v2.mat'} to add ResNet50 only
roiDir  = 'C:\paper2_repo\input\annotated_gray\annotated_gray';   % recursive {ptb,normal}
maskDir = 'C:\paper2_repo\input\mask';  maskSuffix = '_mask';
resDir  = 'C:\paper2_repo\results';     if ~exist(resDir,'dir'), mkdir(resDir); end
csvPath = fullfile(resDir,'containment_combined.csv');
maxImages = 132;      % representative ROI subset (containment saturates well below this)
nRuns     = 3;        % LIME repeats; lower to 2 to halve runtime
numSamples = 1000;
execEnv   = 'gpu';    % gpu-serial (never parfor on one GPU)

% ROI accuracy sanity targets: held-out accuracy of the _v2 models from
% evaluate_models.m (shared 113-image split). NOTE this is only a COARSE guide --
% the sanity figure printed below is measured on the 132-image containment subset,
% which includes training images, so it reads several points HIGHER by design
% (e.g. ResNet50: 0.947 here vs 0.814 held-out). It catches gross preprocessing
% faults (the 0.74 that exposed ADR-004), not small deviations.
clsperfAcc = struct('alexnet_v2',0.805, 'vgg16_v2',0.885, 'vgg19_v2',0.867, ...
                    'resnet50_v2',0.814);

% The four configs reported in tab:contain (keys match make_results_charts.m)
cfgs(1) = struct('key','LIME-sp-K30',        'type','lime','seg','superpixels','m',50, 'K',30,'thr',0);
cfgs(2) = struct('key','LIME-grid-K30',      'type','lime','seg','grid',       'm',49, 'K',30,'thr',0);
cfgs(3) = struct('key','LIME-fine-m100-K20', 'type','lime','seg','superpixels','m',100,'K',20,'thr',0);
cfgs(4) = struct('key','GradCAM@0.5',        'type','gradcam','seg','','m',0,'K',0,'thr',0.5);
nCfg = numel(cfgs);
dispKey = {'LIME sp K30','LIME grid K30','LIME fine K20','Grad-CAM'};

%% --- enumerate ROI images with masks (stratified cap) ---
files = dir(fullfile(roiDir,'**','*.png')); files = files(~[files.isdir]);
[~,o] = sort({files.name}); files = files(o);
imgP = {}; mskP = {}; ab = [];
for f = 1:numel(files)
    base = erase(files(f).name,'.png');
    mp = fullfile(maskDir,[base maskSuffix '.png']);
    if ~isfile(mp), mp2 = fullfile(maskDir,[base '.png']); if isfile(mp2), mp=mp2; else, continue; end; end
    [~,par] = fileparts(files(f).folder); a = ~contains(lower(par),'normal');
    imgP{end+1}=fullfile(files(f).folder,files(f).name); mskP{end+1}=mp; ab(end+1)=a; %#ok<SAGROW>
end
if numel(imgP) > maxImages           % interleave normal/abnormal
    iN=find(~ab); iA=find(ab); ord=[];
    for t=1:max(numel(iN),numel(iA))
        if t<=numel(iN), ord(end+1)=iN(t); end %#ok<SAGROW>
        if t<=numel(iA), ord(end+1)=iA(t); end %#ok<SAGROW>
    end
    ord=ord(1:maxImages); imgP=imgP(ord); mskP=mskP(ord); ab=ab(ord);
end
n = numel(imgP);
fprintf('ROI containment on %d images, minimal preprocessing, nRuns=%d\n', n, nRuns);
fprintf('Running models: %s\n', strjoin(erase(MODELS_TO_RUN,'.mat'), ', '));

%% --- run ---
rows = {};      % aggregate rows
perImg = {};    % per-image rows (class-stratified analysis, CIs, recall)
statsRows = {}; % per-config statistics (recall, CI, direction-aware p, by class)
for mi = 1:numel(MODELS_TO_RUN)
    if ~isfile(MODELS_TO_RUN{mi})
        warning('Missing %s -- skipping. (Run train_resnet50.m first?)', MODELS_TO_RUN{mi});
        continue;
    end
    S = load(MODELS_TO_RUN{mi});
    if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    mkey = erase(MODELS_TO_RUN{mi},'.mat');
    gcLayer = gradcamFeatureLayer(net);   % '' = let MATLAB auto-select
    prec = nan(nCfg,n); iou = nan(nCfg,n); lift = nan(nCfg,n); rec = nan(nCfg,n);
    lfAll = nan(1,n); accHit = 0; used = 0;

    for i = 1:n
        g = imread(imgP{i}); if size(g,3)==3, g=rgb2gray(g); end
        gt = imread(mskP{i}); if size(gt,3)==3, gt=rgb2gray(gt); end
        gt = imresize(gt, size(g),'nearest');
        imgR   = uint8(repmat(imresize(g, inputSize),1,1,3));   % MINIMAL: resize + gray->rgb
        gtMask = imresize(gt, inputSize,'nearest') > 0;
        if nnz(gtMask)==0, continue; end
        used = used + 1; lf = nnz(gtMask)/numel(gtMask); lfAll(i) = lf;
        YP = classify(net, imgR);
        accHit = accHit + ((~contains(lower(char(string(YP))),'normal')) == ab(i));
        for c = 1:nCfg
            if strcmp(cfgs(c).type,'lime')
                pp = zeros(nRuns,1); ii = pp; rr = pp;
                for r = 1:nRuns
                    [~,fMap,fImp] = imageLIME(net,imgR,YP,'Segmentation',cfgs(c).seg, ...
                        'NumFeatures',cfgs(c).m,'NumSamples',numSamples,'ExecutionEnvironment',execEnv);
                    [~,idx] = maxk(fImp, min(cfgs(c).K, numel(fImp)));
                    A = ismember(fMap,idx);
                    pp(r) = nnz(A&gtMask)/max(nnz(A),1);
                    ii(r) = nnz(A&gtMask)/max(nnz(A|gtMask),1);
                    rr(r) = nnz(A&gtMask)/max(nnz(gtMask),1);   % recall: lung coverage
                end
                prec(c,i)=mean(pp); iou(c,i)=mean(ii); rec(c,i)=mean(rr);
            else
                m = gradcamMap(net, imgR, YP, inputSize, execEnv, gcLayer);
                A = m >= cfgs(c).thr;
                prec(c,i)=nnz(A&gtMask)/max(nnz(A),1); iou(c,i)=nnz(A&gtMask)/max(nnz(A|gtMask),1);
                rec(c,i)=nnz(A&gtMask)/max(nnz(gtMask),1);
            end
            lift(c,i)=prec(c,i)-lf;
        end
        if mod(used,20)==0, fprintf('  %s: %d/%d\n', mkey, used, n); end
    end

    accMeas = accHit/max(used,1);
    fprintf('== %s ==  (ROI acc sanity = %.3f', mkey, accMeas);
    fld = matlab.lang.makeValidName(mkey);
    if isfield(clsperfAcc, fld) && ~isnan(clsperfAcc.(fld))
        d = accMeas - clsperfAcc.(fld);
        % expect POSITIVE delta (subset includes training images); only a large
        % NEGATIVE delta indicates a preprocessing fault.
        if d >= -0.05, v='consistent'; else, v='CHECK PREPROCESSING'; end
        fprintf('; held-out %.3f, delta %+.3f -> %s)\n', clsperfAcc.(fld), d, v);
    else
        fprintf('; no target set -- expect high => minimal preprocessing correct)\n');
    end
    for c = 1:nCfg
        keep = ~isnan(lift(c,:));
        Lc = lift(c,keep); abK = logical(ab(keep));
        [pv, pDir, dirStr] = liftTest(Lc);
        [lo, hi] = meanCI(Lc);
        st=''; if pv<1e-13, st='***'; elseif pv<1e-3, st='*'; end
        fprintf('   %-14s lift=%+.3f%s  95%% CI [%+.3f, %+.3f]  (p_%s=%.2g)\n', ...
            dispKey{c}, mean(Lc), st, lo, hi, dirStr, pDir);

        % Class-stratified means. Lift subtracts the PER-IMAGE lung fraction, so the
        % class difference in lung area (0.272 vs 0.232) should cancel and leave no
        % systematic gap here. This is the check reviewers ask for.
        mA = NaN; mN = NaN; pCl = NaN;
        if any(abK) && any(~abK)
            mA = mean(Lc(abK)); mN = mean(Lc(~abK));
            if exist('ttest2','file')==2, [~, pCl] = ttest2(Lc(abK), Lc(~abK)); end
            fprintf('        by class: abnormal %+.3f (n=%d) | normal %+.3f (n=%d)  diff p=%.2g\n', ...
                mA, nnz(abK), mN, nnz(~abK), pCl);
        end
        fprintf('        recall (lung coverage) %.3f\n', mean(rec(c,keep)));

        % aggregate row -- SCHEMA UNCHANGED so make_results_charts.m and the
        % FullCXR merge in ood_containment_lime.m keep working
        rows(end+1,:) = { mkey, cfgs(c).key, mean(prec(c,keep),2), mean(Lc), pv, mean(iou(c,keep),2) }; %#ok<SAGROW>

        % richer statistics go to their own file
        statsRows(end+1,:) = { string(mkey), "ROI", string(cfgs(c).key), nnz(keep), ...
            mean(prec(c,keep),2), mean(rec(c,keep),2), mean(iou(c,keep),2), mean(Lc), ...
            lo, hi, pv, pDir, string(dirStr), mA, mN, pCl }; %#ok<SAGROW>

        for q = find(keep)
            perImg(end+1,:) = { string(mkey), "ROI", string(cfgs(c).key), string(baseOf(imgP{q})), ...
                string(classOf(ab(q))), lfAll(q), prec(c,q), rec(c,q), iou(c,q), lift(c,q) }; %#ok<SAGROW>
        end
    end
end

%% --- write CSV, MERGED with previous runs (keeps models you didn't re-run) ---
if isempty(rows)
    warning('No models produced results; leaving %s untouched.', csvPath);
    Tall = readIfExists(csvPath);
else
    Tnew = cell2table(rows, 'VariableNames', {'Model','Config','Precision','LiftOverChance','pValue','IoU'});
    Tnew.Condition = repmat("ROI", height(Tnew),1);
    Tnew = movevars(Tnew,'Condition','After','Model');

    Told = readIfExists(csvPath);
    if ~isempty(Told)
        % drop the ROI rows for the models we just recomputed, keep everything else
        justRan = string(erase(MODELS_TO_RUN,'.mat'));
        stale = ismember(string(Told.Model), justRan) & strcmp(string(Told.Condition),"ROI");
        Told(stale,:) = [];
        Told.Condition = string(Told.Condition); Told.Model = string(Told.Model);
        Told.Config    = string(Told.Config);
        Tnew.Model = string(Tnew.Model); Tnew.Config = string(Tnew.Config);
        Told = Told(:, Tnew.Properties.VariableNames);   % match column order
        Tall = [Told; Tnew];
    else
        Tall = Tnew;
    end
    writetable(Tall, csvPath);
    fprintf('\nWrote %s (%d rows; merged with previous runs)\n', csvPath, height(Tall));
end

%% --- statistics + per-image CSVs (reviewer M2b, M3c, Minor 2, Minor 6) ---
%  Kept in SEPARATE files: containment_combined.csv must retain its 7-column schema,
%  which make_results_charts.m and the FullCXR merge both depend on.
if ~isempty(statsRows)
    Ts = cell2table(statsRows, 'VariableNames', {'Model','Condition','Config','N', ...
        'Precision','Recall','IoU','LiftOverChance','CI95_lo','CI95_hi', ...
        'pValue_gt0','pValue_directional','Direction','Lift_abnormal','Lift_normal','pValue_classDiff'});
    writetable(mergeKeyed(fullfile(resDir,'containment_stats.csv'), Ts, {'Model','Condition','Config'}), ...
        fullfile(resDir,'containment_stats.csv'));
    fprintf('Wrote %s\n', fullfile(resDir,'containment_stats.csv'));
end
if ~isempty(perImg)
    Tp = cell2table(perImg, 'VariableNames', {'Model','Condition','Config','Basename', ...
        'Class','LungFraction','Precision','Recall','IoU','Lift'});
    writetable(mergeKeyed(fullfile(resDir,'containment_perimage.csv'), Tp, {'Model','Condition','Config'}), ...
        fullfile(resDir,'containment_perimage.csv'));
    fprintf('Wrote %s (%d rows)\n', fullfile(resDir,'containment_perimage.csv'), height(Tp));
end

%% --- print LaTeX rows for tab:contain (rows = config, cols = models) ---
%  Reads the MERGED table, so all four models print even if you ran only one.
if isempty(Tall) || ~ismember('Model', Tall.Properties.VariableNames)
    error('No containment results available (nothing run, no previous CSV).');
end
fprintf('\n==== tab:contain LaTeX rows (paste to replace the table body) ====\n');
order  = {'LIME-sp-K30','LIME-grid-K30','LIME-fine-m100-K20','GradCAM@0.5'};
labelL = {'LIME superpixel, $K{=}30$','LIME grid, $K{=}30$','LIME superpixel (fine), $K{=}20$','Grad-CAM ($\tau{=}0.5$)'};
mk     = erase(allModels,'.mat');
fprintf('Explainer / setting & AlexNet & VGG16 & VGG19 & ResNet50 \\\\\n');
for c = 1:numel(order)
    cells = {};
    for j = 1:numel(mk)
        r = strcmp(string(Tall.Model),mk{j}) & strcmp(string(Tall.Config),order{c}) ...
          & strcmp(string(Tall.Condition),"ROI");
        if ~any(r), cells{end+1} = '--'; continue; end %#ok<SAGROW>
        L = Tall.LiftOverChance(find(r,1)); pv = Tall.pValue(find(r,1));
        st=''; if pv<1e-13, st='^{***}'; elseif pv<1e-3, st='^{*}'; end
        cells{end+1} = sprintf('$%+.3f%s$', L, st); %#ok<SAGROW>
    end
    fprintf('%s & %s \\\\\n', labelL{c}, strjoin(cells,' & '));
end

%% ======================= local functions =======================
function [pGt0, pDir, dirStr] = liftTest(L)
    % pGt0  : the paper's convention, one-sided H1 lift > 0
    % pDir  : one-sided test in the DIRECTION OF THE OBSERVED EFFECT, so a
    %         genuinely negative lift is reported as significantly negative
    %         rather than merely non-significant (reviewer Minor 2)
    pGt0 = NaN; pDir = NaN; dirStr = 'gt0';
    if numel(L) < 2 || exist('ttest','file') ~= 2, return; end
    [~, pGt0] = ttest(L, 0, 'Tail', 'right');
    if mean(L) >= 0
        pDir = pGt0; dirStr = 'gt0';
    else
        [~, pDir] = ttest(L, 0, 'Tail', 'left'); dirStr = 'lt0';
    end
end

function [lo, hi] = meanCI(L)
    % 95% CI of the mean; falls back to the normal approximation without Stats TB
    lo = NaN; hi = NaN;
    n = numel(L); if n < 2, return; end
    se = std(L)/sqrt(n);
    if exist('tinv','file') == 2, t = tinv(0.975, n-1); else, t = 1.96; end
    lo = mean(L) - t*se; hi = mean(L) + t*se;
end

function Tall = mergeKeyed(path, Tnew, keyVars)
    % Replace rows matching this run's (Model,Condition,Config) keys; keep the rest,
    % so ROI rows written here and FullCXR rows written by ood_containment_lime.m
    % accumulate in one file instead of overwriting each other.
    Tall = Tnew;
    if ~isfile(path), return; end
    Told = readtable(path, 'TextType','string');
    if ~isequal(sort(Told.Properties.VariableNames), sort(Tnew.Properties.VariableNames))
        warning('%s has an unexpected schema; overwriting.', path); return;
    end
    Told = Told(:, Tnew.Properties.VariableNames);
    kOld = join(string(Told{:,keyVars}), '|');
    kNew = unique(join(string(Tnew{:,keyVars}), '|'));
    Told(ismember(kOld, kNew), :) = [];
    Tall = [Told; Tnew];
end

function b = baseOf(p)
    [~, b] = fileparts(p);
end

function c = classOf(isAbn)
    if isAbn, c = "abnormal"; else, c = "normal"; end
end

function T = readIfExists(p)
    if isfile(p), T = readtable(p, 'TextType','string'); else, T = table(); end
end

function name = gradcamFeatureLayer(net)
    % MATLAB's gradCAM auto-selects a feature layer, which works for AlexNet/VGG
    % (SeriesNetwork) but can fail on branched DAGs like ResNet50. Pre-resolve the
    % last ReLU with spatial output as an explicit fallback. '' = use auto-select.
    name = '';
    try
        if isa(net,'SeriesNetwork'), return; end          % auto-select is fine
        L = net.Layers;
        for i = numel(L):-1:1
            if isa(L(i),'nnet.cnn.layer.ReLULayer')
                name = L(i).Name; return;                 % e.g. resnet50 -> activation_49_relu
            end
        end
    catch
        name = '';
    end
end

function m = gradcamMap(net, imgR, YP, inputSize, execEnv, gcLayer)
    % Normalised [0,1] Grad-CAM map; falls back to an explicit feature layer if
    % auto-selection errors (ResNet50), then to an empty map so one bad image
    % cannot abort a multi-hour run.
    try
        raw = gradCAM(net, imgR, YP, 'ExecutionEnvironment', execEnv);
    catch
        try
            if isempty(gcLayer), error('no fallback layer'); end
            raw = gradCAM(net, imgR, YP, 'ExecutionEnvironment', execEnv, ...
                          'FeatureLayer', gcLayer);
        catch
            m = zeros(inputSize); return;
        end
    end
    m = mat2gray(imresize(double(raw), inputSize));
end
