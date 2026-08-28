%% Occlusion-robustness gate on Montgomery (the paper's admissibility precondition)
%
%  THE QUESTION. On Shenzhen, the models the two explainers agreed on were exactly
%  the two that tolerate arbitrary occlusion (random-occlusion loss 0.11-0.14),
%  and the models they disagreed on collapsed under any comparable occlusion
%  (0.40-0.44). Does the same admissibility split appear on Montgomery? If yes,
%  the one-number check generalises across sites -- which is precisely what the
%  paper claims for it.
%
%  CHEAP BY DESIGN. LIME rankings are NOT recomputed: the segment map and
%  run-averaged importance vector saved by montgomery_containment.m are reused.
%  What remains is ~15 forward passes per image/config (intact + top-K + bottom-K
%  + 5 area-matched random draws, under two fill values). Minutes on GPU.
%
%  DEFINITIONS (identical to deletion_score.m):
%    drop     = p0 - p_occluded, for the class predicted on the INTACT image
%    gap      = drop(top-K) - drop(random, area-matched from the same segmentation)
%    conc     = sum|imp| over top K / sum|imp| over all m   (flat baseline = K/m)
%  FILLS: 'mean' (matches LIME's own perturbation; background removal is visible)
%         'zero' (background removal is a no-op on lung-masked images)
clc; clear; close all;
rng(0);

%% --- config ---
allModels  = {'alexnet_v2','vgg16_v2','vgg19_v2','resnet50_v2'};
MODELS     = allModels;
COND       = 'MG-ROI';                       % deletion is an in-condition check
mgRoot     = 'C:\paper2_repo\input\montgomery';
roiDir     = fullfile(mgRoot,'roi');
resDir     = 'C:\paper2_repo\results';
csvPath    = fullfile(resDir,'montgomery_deletion.csv');
CFG_USE    = {'LIME-fine-m100-K20','LIME-sp-K30'};   % as in deletion_score.m
FILLS      = {'mean','zero'};
nRandDraws = 5;
execEnv    = 'gpu';

%% --- run ---
sumRows = {};
for mi = 1:numel(MODELS)
    mkey = MODELS{mi};
    matPath = fullfile(resDir, sprintf('mont_masks_%s_%s.mat', mkey, COND));
    netPath = [mkey '.mat'];
    if ~isfile(matPath) || ~isfile(netPath)
        warning('Missing %s or %s -- run montgomery_containment.m first.', matPath, netPath); continue;
    end
    D = load(matPath);
    S = load(netPath);
    if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    n = size(D.MASKS,2);

    % re-enumerate the ROI images in the SAME deterministic order as the mask store
    f = dir(fullfile(roiDir,'**','*.png')); f = f(~[f.isdir]);
    [~,o] = sort({f.name}); f = f(o);
    imgByBase = containers.Map;
    for k = 1:numel(f)
        imgByBase(erase(f(k).name,'.png')) = fullfile(f(k).folder, f(k).name);
    end

    fprintf('\n== %s (%s): %d images ==\n', mkey, COND, n);
    for cu = 1:numel(CFG_USE)
        c = find(strcmp(D.CFGKEYS, CFG_USE{cu}), 1);
        if isempty(c), warning('config %s not in mask store', CFG_USE{cu}); continue; end
        Kc = sscanf(regexp(CFG_USE{cu},'K(\d+)$','match','once'),'K%d');
        dTop = nan(numel(FILLS),n); dRnd = dTop; dBot = dTop;
        concE = nan(1,n);

        for i = 1:n
            fMap = D.FMAPS{c,i}; imp = D.FIMPS{c,i};
            if isempty(fMap) || ~isKey(imgByBase, char(D.BASES(i))), continue; end
            g = imread(imgByBase(char(D.BASES(i))));
            if size(g,3)==3, g = rgb2gray(g); end
            imgR = uint8(repmat(imresize(g, inputSize),1,1,3));
            fMap = imresize(fMap, inputSize, 'nearest');

            mSeg = double(max(fMap(:)));
            K = min(Kc, numel(imp));
            [~,ordImp] = sort(imp,'descend');
            topIdx = ordImp(1:K); botIdx = ordImp(end-K+1:end);
            concE(i) = sum(abs(imp(topIdx)))/max(sum(abs(imp)),eps) - K/mSeg;

            [YP, sc0] = classify(net, imgR);
            [~,pcol] = max(sc0); p0 = sc0(pcol);          % predicted-class probability
            topA = ismember(fMap, topIdx); areaT = nnz(topA);

            for fi = 1:numel(FILLS)
                switch FILLS{fi}
                    case 'mean', fillv = uint8(mean(g(:)));
                    case 'zero', fillv = uint8(0);
                end
                dTop(fi,i) = p0 - probOcc(net, imgR, topA, fillv, pcol);
                dBot(fi,i) = p0 - probOcc(net, imgR, ismember(fMap,botIdx), fillv, pcol);
                dr = zeros(nRandDraws,1);
                for rd = 1:nRandDraws
                    sel = []; area = 0; perm = randperm(double(mSeg));
                    for s = perm
                        sel(end+1) = s; %#ok<AGROW>
                        area = area + nnz(fMap==s);
                        if area >= areaT, break; end
                    end
                    dr(rd) = p0 - probOcc(net, imgR, ismember(fMap,sel), fillv, pcol);
                end
                dRnd(fi,i) = mean(dr);
            end
        end

        for fi = 1:numel(FILLS)
            keep = ~isnan(dTop(fi,:));
            if ~any(keep), continue; end
            gap = dTop(fi,keep) - dRnd(fi,keep);
            pv = NaN;
            if numel(gap)>1 && exist('ttest','file')==2, [~,pv]=ttest(gap,0,'Tail','right'); end
            % Delta_random IS the admissibility number the paper gates on
            fprintf('   %-18s fill=%-4s  drop(top)=%.3f  DROP(RANDOM)=%.3f  drop(bottom)=%.3f\n', ...
                CFG_USE{cu}, FILLS{fi}, mean(dTop(fi,keep)), mean(dRnd(fi,keep)), mean(dBot(fi,keep)));
            fprintf('                      gap=%+.3f (p=%.2g) | concentration excess %+.3f\n', ...
                mean(gap), pv, mean(concE(keep)));
            fprintf('                      [Shenzhen gate: robust 0.11-0.14, fragile 0.40-0.44]\n');
            sumRows(end+1,:) = { string(mkey), string(COND), string(CFG_USE{cu}), string(FILLS{fi}), ...
                nnz(keep), mean(dTop(fi,keep)), mean(dRnd(fi,keep)), mean(dBot(fi,keep)), ...
                mean(gap), pv, mean(concE(keep)) }; %#ok<SAGROW>
        end
    end
end

%% --- save ---
if isempty(sumRows), error('No results produced.'); end
T = cell2table(sumRows, 'VariableNames', {'Model','Condition','Config','Fill','N', ...
    'DropTopK','DropRandom','DropBottomK','Gap','pValue_gap_gt0','ConcentrationExcess'});
writetable(mergeKeyed(csvPath, T, {'Model','Condition','Config','Fill'}), csvPath);
fprintf('\nWrote %s\n', csvPath);
fprintf(['\n==== how to read this ====\n' ...
    '  DROP(RANDOM) is the admissibility number. Models whose prediction collapses\n' ...
    '  under an arbitrary area-matched occlusion cannot support a perturbation-based\n' ...
    '  ranking, so their containment values are outside the metric''s valid regime.\n' ...
    '  If the same robust/fragile split appears here as on Shenzhen, the one-number\n' ...
    '  check generalises across sites.\n']);

%% ======================= local functions =======================
function p = probOcc(net, imgR, mask3, fillv, pcol)
    occ = imgR;
    m3 = repmat(mask3,1,1,3);
    occ(m3) = fillv;
    [~, sc] = classify(net, occ);
    p = sc(pcol);
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
