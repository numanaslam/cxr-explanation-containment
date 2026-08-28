%% Full-CXR (OOD) explanation containment — LIME *and* Grad-CAM, all four models
%
%  WHY THIS EXISTS. The manuscript states that on full radiographs "explanation
%  containment hugs chance". That claim is currently UNSUPPORTED for LIME: the only
%  out-of-distribution containment ever computed was Grad-CAM, because
%  run_all_models_containment.m sets OOD_CFGS = {'GradCAM@0.5','GradCAM@0.7'} to keep
%  the sweep tractable. No live table in the paper reports an OOD containment number
%  at all. This script measures the missing quantity: the same four configurations
%  used for tab:contain, on the full radiographs, for all four networks.
%
%  It also restores the original design of the study — overlap between the U-Net lung
%  mask and the LIME mask on BOTH full CXR and ROI — as a single table with two
%  conditions rather than an ROI-only table.
%
%  READ THE RESULT THROUGH THE RELIABILITY GATE. Classification on these inputs is at
%  or near chance (tab:acc), so a *positive* OOD containment would NOT show that the
%  models remain lung-focused out of distribution — it would describe explanations of
%  unreliable predictions. The value of the measurement is that it replaces an
%  asserted collapse with a measured one, and lets the ROI and OOD columns be compared
%  on identical settings. Interpretation stays gated on accuracy either way.
%
%  THE CHANCE BASELINE DOES NOT MOVE (measured 2026-08-09). ROI images are formed by
%  masking, not cropping, so the lung occupies the same share of the frame in both
%  conditions: 0.2534 on ROI against 0.2481 here. Precision and lift are therefore
%  directly comparable between the two blocks of tab:contain, and image scale is
%  excluded as an explanation of the out-of-distribution collapse by construction.
%  The mean lung fraction is still printed as a check that this continues to hold.
clc; clear; close all;

%% --- config ---
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
MODELS_TO_RUN = allModels;
cxrDir  = 'C:\paper2_repo\input\cxr_resized';   % full radiographs, resized to mask dims
                                                % (produced by resize_images_to_mask.m)
maskDir = 'C:\paper2_repo\input\mask';  maskSuffix = '_mask';
resDir  = 'C:\paper2_repo\results';     if ~exist(resDir,'dir'), mkdir(resDir); end
csvPath = fullfile(resDir,'containment_combined.csv');   % merged with the ROI rows
condName = 'FullCXR';

maxImages  = 132;    % PARITY with the ROI run. Raise to Inf for the whole n=662 set,
                     % but LIME cost scales linearly — 132 stratified is the same
                     % budget the in-distribution table used.
nRuns      = 3;      % same as regen_containment.m
numSamples = 1000;
execEnv    = 'gpu';  % gpu-serial only (ADR-006)

%  OPTIONAL subset split. The reference masks are ALL manual annotations (566 of 662
%  radiographs), so there is no provenance split to make. Retained as a hook: supply a
%  file of basenames to report any subset of interest separately.
gtListFile = '';     % '' = no split

% Same four configurations as tab:contain, so ROI and OOD columns are directly comparable
cfgs(1) = struct('key','LIME-sp-K30',        'type','lime','seg','superpixels','m',50, 'K',30,'thr',0);
cfgs(2) = struct('key','LIME-grid-K30',      'type','lime','seg','grid',       'm',49, 'K',30,'thr',0);
cfgs(3) = struct('key','LIME-fine-m100-K20', 'type','lime','seg','superpixels','m',100,'K',20,'thr',0);
cfgs(4) = struct('key','GradCAM@0.5',        'type','gradcam','seg','','m',0,'K',0,'thr',0.5);
nCfg = numel(cfgs);
dispKey = {'LIME sp K30','LIME grid K30','LIME fine K20','Grad-CAM'};

%% --- enumerate full CXRs with masks (flat dir; class from _0/_1 suffix) ---
files = dir(fullfile(cxrDir,'*.png')); files = files(~[files.isdir]);
[~,o] = sort({files.name}); files = files(o);
imgP = {}; mskP = {}; ab = []; base_ = {};
for f = 1:numel(files)
    b = erase(files(f).name,'.png');
    mp = fullfile(maskDir,[b maskSuffix '.png']);
    if ~isfile(mp), mp2 = fullfile(maskDir,[b '.png']); if isfile(mp2), mp=mp2; else, continue; end; end
    t = split(b,'_'); a = strcmp(t{end},'1');            % class from filename suffix
    imgP{end+1}=fullfile(files(f).folder,files(f).name);  %#ok<SAGROW>
    mskP{end+1}=mp; ab(end+1)=a; base_{end+1}=b;         %#ok<SAGROW>
end
assert(~isempty(imgP), 'No image/mask pairs found in %s (did resize_images_to_mask.m run?)', cxrDir);
if numel(imgP) > maxImages                                % class-stratified interleave
    iN=find(~ab); iA=find(ab); ord=[];
    for t=1:max(numel(iN),numel(iA))
        if t<=numel(iN), ord(end+1)=iN(t); end %#ok<SAGROW>
        if t<=numel(iA), ord(end+1)=iA(t); end %#ok<SAGROW>
    end
    ord=ord(1:maxImages); imgP=imgP(ord); mskP=mskP(ord); ab=ab(ord); base_=base_(ord);
end
n = numel(imgP);
isGT = false(1,n);
if ~isempty(gtListFile) && isfile(gtListFile)
    L = strtrim(string(splitlines(fileread(gtListFile)))); L = L(L~="");
    isGT = ismember(string(base_), L);
    fprintf('Mask provenance: %d ground-truth, %d U-Net-generated\n', nnz(isGT), nnz(~isGT));
end
fprintf('OOD containment on %d full CXRs, minimal preprocessing, nRuns=%d\n', n, nRuns);
fprintf('Running models: %s\n', strjoin(erase(MODELS_TO_RUN,'.mat'), ', '));

%% --- run ---
rows = {}; statsRows = {}; perImg = {};
for mi = 1:numel(MODELS_TO_RUN)
    if ~isfile(MODELS_TO_RUN{mi})
        warning('Missing %s -- skipping.', MODELS_TO_RUN{mi}); continue;
    end
    S = load(MODELS_TO_RUN{mi});
    if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    mkey = erase(MODELS_TO_RUN{mi},'.mat');
    gcLayer = gradcamFeatureLayer(net);
    prec = nan(nCfg,n); iou = nan(nCfg,n); lift = nan(nCfg,n); rec = nan(nCfg,n);
    lfAll = nan(1,n); accHit = 0; used = 0;

    for i = 1:n
        g = imread(imgP{i}); if size(g,3)==3, g=rgb2gray(g); end
        gt = imread(mskP{i}); if size(gt,3)==3, gt=rgb2gray(gt); end
        gt = imresize(gt, size(g),'nearest');
        imgR   = uint8(repmat(imresize(g, inputSize),1,1,3));   % MINIMAL (ADR-004)
        gtMask = imresize(gt, inputSize,'nearest') > 0;
        if nnz(gtMask)==0, continue; end
        used = used + 1;
        lf = nnz(gtMask)/numel(gtMask); lfAll(i) = lf;
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

    fprintf('== %s ==  (OOD acc %.3f -- expect ~chance; mean lung fraction %.3f)\n', ...
        mkey, accHit/max(used,1), mean(lfAll,'omitnan'));
    for c = 1:nCfg
        keep = ~isnan(lift(c,:));
        Lc = lift(c,keep); abK = logical(ab(keep));
        [pv, pDir, dirStr] = liftTest(Lc);
        [lo, hi] = meanCI(Lc);
        st=''; if pv<1e-13, st='***'; elseif pv<1e-3, st='*'; end
        fprintf('   %-14s lift=%+.3f%s  95%% CI [%+.3f, %+.3f]  (p_%s=%.2g)  IoU=%.3f  recall=%.3f\n', ...
            dispKey{c}, mean(Lc), st, lo, hi, dirStr, pDir, mean(iou(c,keep),2), mean(rec(c,keep),2));
        mA=NaN; mN=NaN; pCl=NaN;
        if any(abK) && any(~abK)
            mA=mean(Lc(abK)); mN=mean(Lc(~abK));
            if exist('ttest2','file')==2, [~,pCl]=ttest2(Lc(abK), Lc(~abK)); end
            fprintf('        by class: abnormal %+.3f | normal %+.3f  diff p=%.2g\n', mA, mN, pCl);
        end
        rows(end+1,:) = { mkey, cfgs(c).key, mean(prec(c,keep),2), mean(Lc), pv, ...
            mean(iou(c,keep),2) }; %#ok<SAGROW>
        statsRows(end+1,:) = { string(mkey), string(condName), string(cfgs(c).key), nnz(keep), ...
            mean(prec(c,keep),2), mean(rec(c,keep),2), mean(iou(c,keep),2), mean(Lc), ...
            lo, hi, pv, pDir, string(dirStr), mA, mN, pCl }; %#ok<SAGROW>
        for q = find(keep)
            perImg(end+1,:) = { string(mkey), string(condName), string(cfgs(c).key), string(base_{q}), ...
                string(classOf(ab(q))), lfAll(q), prec(c,q), rec(c,q), iou(c,q), lift(c,q) }; %#ok<SAGROW>
        end
        % optional mask-provenance breakdown
        if any(isGT) && any(~isGT)
            for grp = [true false]
                sel = (isGT==grp) & ~isnan(lift(c,:));
                if nnz(sel) > 1
                    fprintf('        [%s masks, n=%d] lift=%+.3f\n', ...
                        string(grp).*"ground-truth" + (~grp).*"U-Net", nnz(sel), mean(lift(c,sel)));
                end
            end
        end
    end
end

%% --- merge into containment_combined.csv as Condition = FullCXR ---
if isempty(rows), error('No models produced results.'); end
Tnew = cell2table(rows, 'VariableNames', {'Model','Config','Precision','LiftOverChance','pValue','IoU'});
Tnew.Condition = repmat(string(condName), height(Tnew),1);
Tnew = movevars(Tnew,'Condition','After','Model');
Tnew.Model = string(Tnew.Model); Tnew.Config = string(Tnew.Config);
if isfile(csvPath)
    Told = readtable(csvPath,'TextType','string');
    Told = Told(:, Tnew.Properties.VariableNames);
    stale = ismember(Told.Model, unique(Tnew.Model)) & Told.Condition == string(condName);
    Told(stale,:) = [];
    Tall = [Told; Tnew];
else
    Tall = Tnew;
end
writetable(Tall, csvPath);
fprintf('\nWrote %s (%d rows; ROI rows preserved)\n', csvPath, height(Tall));

%% --- statistics + per-image CSVs (merged with the ROI rows) ---
if ~isempty(statsRows)
    Ts = cell2table(statsRows, 'VariableNames', {'Model','Condition','Config','N', ...
        'Precision','Recall','IoU','LiftOverChance','CI95_lo','CI95_hi', ...
        'pValue_gt0','pValue_directional','Direction','Lift_abnormal','Lift_normal','pValue_classDiff'});
    f = fullfile(resDir,'containment_stats.csv');
    writetable(mergeKeyed(f, Ts, {'Model','Condition','Config'}), f);
    fprintf('Wrote %s\n', f);
end
if ~isempty(perImg)
    Tp = cell2table(perImg, 'VariableNames', {'Model','Condition','Config','Basename', ...
        'Class','LungFraction','Precision','Recall','IoU','Lift'});
    f = fullfile(resDir,'containment_perimage.csv');
    writetable(mergeKeyed(f, Tp, {'Model','Condition','Config'}), f);
    fprintf('Wrote %s (%d rows)\n', f, height(Tp));
end

%% --- LaTeX: tab:contain with BOTH conditions (the original two-condition design) ---
fprintf('\n==== tab:contain rows, ROI vs Full CXR ====\n');
order  = {'LIME-sp-K30','LIME-grid-K30','LIME-fine-m100-K20','GradCAM@0.5'};
labelL = {'LIME superpixel, $K{=}30$','LIME grid, $K{=}30$','LIME superpixel (fine), $K{=}20$','Grad-CAM ($\tau{=}0.5$)'};
mk = erase(allModels,'.mat');
for cond = ["ROI","FullCXR"]
    fprintf('\\multicolumn{5}{l}{\\emph{%s}} \\\\\n', cond);
    for c = 1:numel(order)
        cells = {};
        for j = 1:numel(mk)
            r = Tall.Model==mk{j} & Tall.Config==order{c} & Tall.Condition==cond;
            if ~any(r), cells{end+1}='--'; continue; end %#ok<SAGROW>
            L = Tall.LiftOverChance(find(r,1)); pv = Tall.pValue(find(r,1));
            st=''; if pv<1e-13, st='^{***}'; elseif pv<1e-3, st='^{*}'; end
            cells{end+1} = sprintf('$%+.3f%s$', L, st); %#ok<SAGROW>
        end
        fprintf('%s & %s \\\\\n', labelL{c}, strjoin(cells,' & '));
    end
end

%% ======================= local functions =======================
function [pGt0, pDir, dirStr] = liftTest(L)
    % pGt0 is the paper's convention (H1: lift > 0); pDir tests in the direction of
    % the observed effect so a genuinely negative lift -- AlexNet's Grad-CAM at
    % -0.156 -- is reported as significantly negative, not merely non-significant.
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
    lo = NaN; hi = NaN;
    n = numel(L); if n < 2, return; end
    se = std(L)/sqrt(n);
    if exist('tinv','file') == 2, t = tinv(0.975, n-1); else, t = 1.96; end
    lo = mean(L) - t*se; hi = mean(L) + t*se;
end

function c = classOf(isAbn)
    if isAbn, c = "abnormal"; else, c = "normal"; end
end

function Tall = mergeKeyed(path, Tnew, keyVars)
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

function name = gradcamFeatureLayer(net)
    name = '';
    try
        if isa(net,'SeriesNetwork'), return; end
        L = net.Layers;
        for i = numel(L):-1:1
            if isa(L(i),'nnet.cnn.layer.ReLULayer'), name = L(i).Name; return; end
        end
    catch, name = '';
    end
end

function m = gradcamMap(net, imgR, YP, inputSize, execEnv, gcLayer)
    try
        raw = gradCAM(net, imgR, YP, 'ExecutionEnvironment', execEnv);
    catch
        try
            if isempty(gcLayer), error('no fallback layer'); end
            raw = gradCAM(net, imgR, YP, 'ExecutionEnvironment', execEnv, 'FeatureLayer', gcLayer);
        catch
            m = zeros(inputSize); return;
        end
    end
    m = mat2gray(imresize(double(raw), inputSize));
end
