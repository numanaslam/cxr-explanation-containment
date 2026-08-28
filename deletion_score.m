%% Deletion-based faithfulness check for LIME attributions (ROI, in-distribution)
%
%  QUESTION THIS ANSWERS. ResNet50 records the LOWEST LIME containment of the four
%  models (+0.022..+0.037) while recording the HIGHEST Grad-CAM containment
%  (+0.357), despite being a competent classifier (AUC 0.901). Two readings:
%     (A) ResNet50 genuinely attends outside the lung  -> a fact about the MODEL
%     (B) LIME's attributions are uninformative for it -> a fact about the EXPLAINER
%  Containment cannot separate these. Deletion can: if occluding LIME's top-K
%  superpixels barely moves ResNet50's predicted-class probability while the same
%  operation collapses VGG16's, then LIME's ranking carries little information for
%  ResNet50 and (B) holds.
%
%  WHY A PLAIN DELETION SCORE IS NOT ENOUGH. The confidence drop scales with how
%  much IMAGE AREA you remove, and top-K superpixels are not the same area across
%  models or images. So we always compare against an AREA-MATCHED RANDOM control
%  drawn from the same segmentation, and report the GAP:
%       gap = drop(top-K) - drop(random, same area)
%  gap ~ 0 means "the ranking is no better than chance at finding what matters" --
%  which is exactly claim (B). A bottom-K control is included as a third reference.
%
%  SECOND, CHEAPER TEST (no occlusion needed): importance concentration. If the
%  importance vector is FLAT, the top-K carries no more mass than any K features:
%       conc = sum(|imp| over top K) / sum(|imp| over all m),  flat baseline = K/m
%  Excess concentration (conc - K/m) near zero is independent evidence for (B).
%
%  PROTOCOL. Same 132-image subset, same enumeration order, same minimal
%  preprocessing and same LIME settings as regen_containment.m.
%
%  !! THE CONTAINMENT LIFT PRINTED HERE DOES NOT EQUAL tab:contain, BY DESIGN.
%  regen_containment.m computes precision separately per LIME run and averages the
%  PRECISIONS (= expected containment of one LIME run, what a practitioner gets).
%  This script averages the IMPORTANCE vectors across runs first and ranks once
%  (= containment of a denoised explanation). Measured 2026-08-09, fine config:
%      AlexNet  +0.158 -> +0.197     VGG16 +0.202 -> +0.290
%      VGG19    +0.212 -> +0.301     ResNet50 +0.024 -> +0.025
%  So LIME sampling noise suppresses single-run containment by up to ~0.09 for
%  three models and by nothing at all for ResNet50. Do not "fix" either script to
%  match the other -- they measure different quantities and the gap between them
%  is itself a result about the stability of this metric.
%
%  RUNTIME. LIME dominates (numSamples forward passes per run). Roughly comparable
%  to one regen_containment.m sweep. Trim by setting nRuns=2 or cutting cfgs to the
%  fine config alone -- that is where the ResNet50/VGG gap is widest.
clc; clear; close all;
rng(0);   % reproducible random controls

%% --- config ---
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
MODELS_TO_RUN = allModels;      % e.g. {'resnet50_v2.mat','vgg16_v2.mat'} for the key pair
roiDir  = 'C:\paper2_repo\input\annotated_gray\annotated_gray';
maskDir = 'C:\paper2_repo\input\mask';  maskSuffix = '_mask';
resDir  = 'C:\paper2_repo\results';     if ~exist(resDir,'dir'), mkdir(resDir); end
csvPath     = fullfile(resDir,'deletion_scores.csv');       % per-model x per-config summary
csvPerImage = fullfile(resDir,'deletion_scores_perimage.csv');

maxImages  = 132;    % SAME subset as regen_containment.m (deterministic enumeration)
nRuns      = 3;      % LIME repeats; importance is AVERAGED over runs before ranking
numSamples = 1000;
nRandDraws = 5;      % random controls averaged per image (reduces control variance)
execEnv    = 'gpu';  % gpu-serial only (ADR-006)

%  FILL VALUE used to occlude. This is not a detail -- it IS the hypothesis.
%  The ROI images are lung-masked, so outside the lung the image is already 0.
%    'mean'  : occlude with the image mean. Removing a BACKGROUND superpixel then
%              creates a large pixel change the net can react to, which is the
%              proposed mechanism by which background superpixels acquire spurious
%              importance. Use this to match how LIME itself perturbs.
%    'zero'  : occlude with 0. Removing a background superpixel is then a NO-OP.
%  Running both is the cleanest evidence: under 'zero', a model whose top-K sits on
%  background should show a drop of ~0 with no area-matched control able to help it.
FILLS = {'mean','zero'};

% LIME configs -- keys and parameters identical to regen_containment.m
cfgs(1) = struct('key','LIME-fine-m100-K20','seg','superpixels','m',100,'K',20);
cfgs(2) = struct('key','LIME-sp-K30',       'seg','superpixels','m',50, 'K',30);
nCfg = numel(cfgs);

%% --- enumerate ROI images with masks (IDENTICAL to regen_containment.m) ---
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
if numel(imgP) > maxImages
    iN=find(~ab); iA=find(ab); ord=[];
    for t=1:max(numel(iN),numel(iA))
        if t<=numel(iN), ord(end+1)=iN(t); end %#ok<SAGROW>
        if t<=numel(iA), ord(end+1)=iA(t); end %#ok<SAGROW>
    end
    ord=ord(1:maxImages); imgP=imgP(ord); mskP=mskP(ord); ab=ab(ord);
end
n = numel(imgP);
fprintf('Deletion faithfulness on %d ROI images | nRuns=%d | fills: %s\n', ...
    n, nRuns, strjoin(FILLS,', '));

%% --- run ---
sumRows = {};   % model x config x fill aggregates
imgRows = {};   % per-image detail
for mi = 1:numel(MODELS_TO_RUN)
    if ~isfile(MODELS_TO_RUN{mi})
        warning('Missing %s -- skipping.', MODELS_TO_RUN{mi}); continue;
    end
    S = load(MODELS_TO_RUN{mi});
    if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    mkey    = erase(MODELS_TO_RUN{mi},'.mat');
    classes = string(net.Layers(end).Classes);

    % accumulators: [cfg x fill]
    dTop=cell(nCfg,numel(FILLS)); dRnd=dTop; dBot=dTop; aTop=dTop; aRnd=dTop;
    conc=cell(nCfg,1); lift=cell(nCfg,1); precOut=cell(nCfg,1);

    for i = 1:n
        g = imread(imgP{i}); if size(g,3)==3, g=rgb2gray(g); end
        gt = imread(mskP{i}); if size(gt,3)==3, gt=rgb2gray(gt); end
        gt = imresize(gt, size(g),'nearest');
        imgR   = uint8(repmat(imresize(g, inputSize),1,1,3));   % MINIMAL preprocessing
        gtMask = imresize(gt, inputSize,'nearest') > 0;
        if nnz(gtMask)==0, continue; end
        lf = nnz(gtMask)/numel(gtMask);

        % baseline prediction + its probability
        [YP, sc0] = classify(net, imgR);
        ci = find(classes == string(YP), 1);
        p0 = sc0(ci);

        for c = 1:nCfg
            % ---- averaged LIME importance (segmentation is deterministic, so the
            %      feature map is stable across runs and only importance varies) ----
            fMap = []; impAcc = [];
            for r = 1:nRuns
                [~, fM, fI] = imageLIME(net, imgR, YP, 'Segmentation', cfgs(c).seg, ...
                    'NumFeatures', cfgs(c).m, 'NumSamples', numSamples, ...
                    'ExecutionEnvironment', execEnv);
                if isempty(impAcc), fMap = fM; impAcc = zeros(size(fI)); end
                if numel(fI) ~= numel(impAcc), continue; end   % defensive
                impAcc = impAcc + fI;
            end
            if isempty(impAcc), continue; end
            imp = impAcc / nRuns;
            m   = numel(imp);
            K   = min(cfgs(c).K, m);

            [~, ordImp] = sort(imp, 'descend');
            topIdx = ordImp(1:K);
            botIdx = ordImp(end-K+1:end);
            Atop = ismember(fMap, topIdx);
            Abot = ismember(fMap, botIdx);

            % containment of the top-K mask (should reproduce tab:contain)
            pTop = nnz(Atop & gtMask)/max(nnz(Atop),1);
            lift{c}(end+1)    = pTop - lf;              %#ok<SAGROW>
            precOut{c}(end+1) = 1 - pTop;               %#ok<SAGROW> fraction outside lung

            % importance concentration vs the flat baseline K/m
            av = abs(imp);
            conc{c}(end+1) = sum(av(topIdx))/max(sum(av),eps) - K/m; %#ok<SAGROW>

            for fi = 1:numel(FILLS)
                fillVal = fillValue(imgR, FILLS{fi});
                dT = confDrop(net, imgR, Atop, p0, ci, fillVal);
                dB = confDrop(net, imgR, Abot, p0, ci, fillVal);

                % AREA-MATCHED random control, averaged over nRandDraws
                tgt = nnz(Atop); dR = zeros(nRandDraws,1); aR = zeros(nRandDraws,1);
                for q = 1:nRandDraws
                    Arnd  = randomMaskMatchedArea(fMap, m, tgt);
                    dR(q) = confDrop(net, imgR, Arnd, p0, ci, fillVal);
                    aR(q) = nnz(Arnd)/numel(Arnd);
                end
                dTop{c,fi}(end+1) = dT;                      %#ok<SAGROW>
                dBot{c,fi}(end+1) = dB;                      %#ok<SAGROW>
                dRnd{c,fi}(end+1) = mean(dR);                %#ok<SAGROW>
                aTop{c,fi}(end+1) = nnz(Atop)/numel(Atop);   %#ok<SAGROW>
                aRnd{c,fi}(end+1) = mean(aR);                %#ok<SAGROW>

                imgRows(end+1,:) = { string(mkey), string(cfgs(c).key), string(FILLS{fi}), ...
                    i, p0, dT, mean(dR), dB, nnz(Atop)/numel(Atop), pTop - lf }; %#ok<SAGROW>
            end
        end
        if mod(i,20)==0, fprintf('  %s: %d/%d\n', mkey, i, n); end
    end

    % ---- aggregate + report ----
    fprintf('\n===== %s =====\n', mkey);
    for c = 1:nCfg
        fprintf('  %s   containment lift %+.3f | mean %.1f%% of top-K mask lies OUTSIDE the lung\n', ...
            cfgs(c).key, mean(lift{c}), 100*mean(precOut{c}));
        fprintf('    importance concentration above flat baseline: %+.4f  (0 = perfectly flat)\n', ...
            mean(conc{c}));
        for fi = 1:numel(FILLS)
            dT = dTop{c,fi}; dR = dRnd{c,fi}; dB = dBot{c,fi};
            gap = dT - dR;
            if numel(gap) > 1 && exist('ttest','file')==2
                [~,pv] = ttest(gap, 0, 'Tail','right');   % H1: top-K beats matched random
            else, pv = NaN; end
            fprintf(['    fill=%-4s  drop top-K %.4f | matched-random %.4f | bottom-K %.4f\n' ...
                     '               GAP (top - random) %+.4f  (p=%.2g)   area top %.3f vs rnd %.3f\n'], ...
                FILLS{fi}, mean(dT), mean(dR), mean(dB), mean(gap), pv, ...
                mean(aTop{c,fi}), mean(aRnd{c,fi}));
            sumRows(end+1,:) = { string(mkey), string(cfgs(c).key), string(FILLS{fi}), ...
                numel(dT), mean(dT), mean(dR), mean(dB), mean(gap), pv, ...
                mean(aTop{c,fi}), mean(aRnd{c,fi}), mean(conc{c}), mean(lift{c}) }; %#ok<SAGROW>
        end
    end
end

%% --- save (merge-safe, same convention as the other scripts) ---
if isempty(sumRows), error('No results produced.'); end
T = cell2table(sumRows, 'VariableNames', {'Model','Config','Fill','N', ...
    'Drop_topK','Drop_randomMatched','Drop_bottomK','Gap','pValue', ...
    'Area_topK','Area_random','ExcessConcentration','ContainmentLift'});
writetable(mergeOn(csvPath, T, {'Model','Config','Fill'}), csvPath);
TI = cell2table(imgRows, 'VariableNames', {'Model','Config','Fill','ImageIdx', ...
    'p0','Drop_topK','Drop_randomMatched','Drop_bottomK','Area_topK','ContainmentLift'});
writetable(mergeOn(csvPerImage, TI, {'Model','Config','Fill'}), csvPerImage);
fprintf('\nWrote %s\n      %s\n', csvPath, csvPerImage);

%% --- interpretation guide + LaTeX row ---
fprintf('\n==== how to read this ====\n');
fprintf(['  GAP >> 0 and significant : LIME''s ranking finds what the model uses ->\n' ...
         '                             low containment means the model really does look\n' ...
         '                             outside the lung (reading A).\n' ...
         '  GAP ~ 0                  : top-K is no better than an area-matched random set ->\n' ...
         '                             the attributions are uninformative for that model,\n' ...
         '                             so its low containment is an EXPLAINER artefact (B).\n' ...
         '  Compare fills            : if the gap is healthy under fill=mean but collapses\n' ...
         '                             under fill=zero, the "importance" was being carried by\n' ...
         '                             masked BACKGROUND superpixels, whose removal is a no-op\n' ...
         '                             at zero fill. That is the proposed mechanism.\n']);
fprintf('\n==== LaTeX rows (fill=%s) ====\n', FILLS{1});
fprintf('Model & Drop top-$K$ & Matched random & Gap & $p$ \\\\\n');
sel = T(T.Fill == string(FILLS{1}) & T.Config == "LIME-fine-m100-K20", :);
for r = 1:height(sel)
    st=''; if sel.pValue(r) < 1e-3, st='^{*}'; end
    fprintf('%s & %.3f & %.3f & $%+.3f%s$ & %.2g \\\\\n', sel.Model(r), ...
        sel.Drop_topK(r), sel.Drop_randomMatched(r), sel.Gap(r), st, sel.pValue(r));
end

%% ======================= local functions =======================
function v = fillValue(imgR, mode)
    switch lower(mode)
        case 'mean', v = uint8(mean(double(imgR(:))));
        case 'zero', v = uint8(0);
        otherwise,   error('Unknown fill mode: %s', mode);
    end
end

function d = confDrop(net, imgR, A, p0, ci, fillVal)
    % Probability the ORIGINAL predicted class loses when region A is occluded.
    % Positive = occluding A hurt that class, i.e. A mattered.
    occ = imgR;
    A3  = repmat(A, 1, 1, size(imgR,3));
    occ(A3) = fillVal;
    [~, sc] = classify(net, occ);
    d = p0 - sc(ci);
end

function A = randomMaskMatchedArea(fMap, m, targetPx)
    % Draw superpixels uniformly at random until the occluded area reaches the
    % top-K area. Area matching matters: the confidence drop scales with how much
    % of the frame is removed, so an unmatched control would confound importance
    % with size.
    perm = randperm(m);
    A = false(size(fMap));
    for t = 1:m
        A = A | (fMap == perm(t));
        if nnz(A) >= targetPx, break; end
    end
end

function Tall = mergeOn(csvPath, Tnew, keyVars)
    % Replace rows matching this run's (Model,Config,Fill) keys; keep the rest.
    for v = Tnew.Properties.VariableNames
        if iscellstr(Tnew.(v{1})) || ischar(Tnew.(v{1})) %#ok<ISCLSTR>
            Tnew.(v{1}) = string(Tnew.(v{1}));
        end
    end
    if ~isfile(csvPath), Tall = Tnew; return; end
    Told = readtable(csvPath, 'TextType','string');
    if ~isequal(sort(Told.Properties.VariableNames), sort(Tnew.Properties.VariableNames))
        warning('%s has an unexpected schema; overwriting instead of merging.', csvPath);
        Tall = Tnew; return;
    end
    Told = Told(:, Tnew.Properties.VariableNames);
    kOld = join(string(Told{:,keyVars}), '|');
    kNew = unique(join(string(Tnew{:,keyVars}), '|'));
    Told(ismember(kOld, kNew), :) = [];
    Tall = [Told; Tnew];
end
