%% Phase 1 audit: inventory the manual lung masks used as the reference region
%
%  WHY. The lung masks in input\mask are the collection's MANUAL annotations, and they
%  are the reference region B against which every containment result in this paper is
%  scored -- containment, lift, IoU and the deletion check all inherit their
%  properties. Methods previously gave no count, no geometry and no quality statement
%  for them. For a soundness-reviewed venue that was the weakest point in the
%  manuscript. This script turns those unstated facts into measurements.
%
%  NOTE ON THE U-NET. The segmentation network does NOT produce the masks used here;
%  it is the deployment path (a manual mask would not exist at inference time). No
%  segmentation training or inference code exists in this repository, so the U-Net
%  hyperparameters cannot be recovered from code and must come from the author's
%  records. Everything else Phase 1 needs is measured below.
%
%  COMPONENT BREAKDOWN. A mask with other than two connected components is ambiguous:
%  one component means the left and right lung touch in the annotation (topology only,
%  area unaffected), whereas three or more means stray fragments that inflate the
%  apparent lung area and therefore shift the per-image chance baseline. The report
%  separates the two cases and quantifies the effect on the baseline, so the
%  manuscript can state whether it is cosmetic or material rather than just flagging
%  a count.
%
%  OPTIONAL ROBUSTNESS CHECK. If automatically generated (U-Net) masks were saved for
%  the same radiographs, point the directories below at them: the script then reports
%  Dice/IoU between the manual reference and the automatic masks, which is exactly the
%  substitution question the Limitations section currently leaves open.
%
%  OUTPUTS
%    results\unet_mask_audit.csv   per-mask geometry, component counts, agreement
%    results\gt_basenames.txt      basenames covered (only written if a second mask
%                                  set is supplied for comparison)
clc; clear; close all;

%% --- config ---
maskDir    = 'C:\paper2_repo\input\mask';       % masks actually used for evaluation
maskSuffix = '_mask';
resDir     = 'C:\paper2_repo\results';  if ~exist(resDir,'dir'), mkdir(resDir); end

%  OPTIONAL: a SECOND mask set to compare the manual reference against -- normally the
%  U-Net output, if it was saved. Leave as-is to skip. Montgomery-style annotations are
%  stored as separate left/right images and are unioned automatically. Any path that
%  does not exist is skipped silently.
gtSingleDirs = { 'C:\paper2_repo\input\gt_mask' };                 % one mask per image
gtLeftDir    = 'C:\paper2_repo\input\ManualMask\leftMask';         % '' to disable
gtRightDir   = 'C:\paper2_repo\input\ManualMask\rightMask';

%% --- inventory the evaluation masks ---
f = dir(fullfile(maskDir,'*.png')); f = f(~[f.isdir]);
assert(~isempty(f), 'No masks found in %s', maskDir);
[~,o] = sort({f.name}); f = f(o);
fprintf('Found %d mask files in %s\n', numel(f), maskDir);

rows = {}; gtNames = strings(0,1);
nBinary = 0; nGT = 0;
for i = 1:numel(f)
    M = imread(fullfile(f(i).folder, f(i).name));
    if size(M,3)==3, M = rgb2gray(M); end
    Mb = M > 0;
    base = erase(erase(f(i).name,'.png'), maskSuffix);

    u = unique(M(:));
    isBin = numel(u) <= 2;                 % strictly binary, or grey-valued?
    nBinary = nBinary + isBin;
    lungFrac = nnz(Mb)/numel(Mb);

    %  Component breakdown. A count other than 2 is ambiguous on its own:
    %    1  = left and right lung touching in the annotation (topology only, benign)
    %   >=3 = fragments or stray specks, which inflate the apparent lung area and
    %         therefore shift the per-image chance baseline
    %  So we also record how much mask area sits beyond the two largest components,
    %  and what the lung fraction would be if only those two were kept.
    CC = bwconncomp(Mb);
    areas = sort(cellfun(@numel, CC.PixelIdxList), 'descend');
    ncomp = numel(areas);
    if isempty(areas)
        top2 = 0; extraFrac = 0;
    else
        top2 = sum(areas(1:min(2,numel(areas))));
        extraFrac = (sum(areas) - top2) / max(sum(areas),1);   % share beyond 2 largest
    end
    lungFracTop2 = top2 / numel(Mb);

    % --- second mask set for comparison, if supplied ---
    G = loadGroundTruth(base, gtSingleDirs, gtLeftDir, gtRightDir);
    if isempty(G)
        dice = NaN; iou = NaN; hasGT = false;
    else
        G = imresize(G, size(Mb), 'nearest') > 0;
        inter = nnz(Mb & G);
        dice  = 2*inter / max(nnz(Mb)+nnz(G), 1);
        iou   = inter / max(nnz(Mb | G), 1);
        hasGT = true; nGT = nGT + 1;
        gtNames(end+1,1) = string(base); %#ok<SAGROW>
    end

    rows(end+1,:) = { string(base), size(Mb,1), size(Mb,2), isBin, lungFrac, ...
                      ncomp, extraFrac, lungFracTop2, hasGT, dice, iou }; %#ok<SAGROW>
    if mod(i,100)==0, fprintf('  %d/%d\n', i, numel(f)); end
end

T = cell2table(rows, 'VariableNames', {'Basename','Rows','Cols','IsBinary', ...
    'LungFraction','NumComponents','ExtraAreaFraction','LungFractionTop2', ...
    'HasGroundTruth','Dice','IoU'});
writetable(T, fullfile(resDir,'unet_mask_audit.csv'));

%% --- report ---
fprintf('\n===== Phase 1 mask audit =====\n');
fprintf('  masks total                : %d\n', height(T));
fprintf('  strictly binary            : %d (%.1f%%)\n', nBinary, 100*nBinary/height(T));
fprintf('  lung fraction              : mean %.3f, sd %.3f, range %.3f-%.3f\n', ...
    mean(T.LungFraction), std(T.LungFraction), min(T.LungFraction), max(T.LungFraction));

%  --- component breakdown: is a count of ~=2 cosmetic or material? ---
fprintf('\n  --- connected components ---\n');
edges = [1 2 3 4 5 inf];
for k = 1:numel(edges)-1
    lo = edges(k); hi = edges(k+1);
    if isinf(hi), sel = T.NumComponents >= lo; lbl = sprintf('%d+', lo);
    else,         sel = T.NumComponents == lo; lbl = sprintf('%d ', lo); end
    if nnz(sel) > 0
        fprintf('    %-3s components : %4d masks (%.1f%%)\n', lbl, nnz(sel), 100*nnz(sel)/height(T));
    end
end
merged = T.NumComponents == 1;
frag   = T.NumComponents >= 3;
fprintf('\n    MERGED (1 component, lungs touching -- topology only, area unaffected): %d\n', nnz(merged));
fprintf('    FRAGMENTED (>=3 components, extra pieces inflate lung area):         %d\n', nnz(frag));
if any(frag)
    e = T.ExtraAreaFraction(frag);
    fprintf('      area beyond the 2 largest: mean %.4f, median %.4f, max %.4f of mask\n', ...
        mean(e), median(e), max(e));
    fprintf('      masks where that exceeds 1%% of mask area: %d\n', nnz(e > 0.01));
    d = abs(T.LungFraction - T.LungFractionTop2);
    fprintf('      effect on the chance baseline if extras were dropped:\n');
    fprintf('        mean |dLungFraction| = %.5f, max = %.5f  (baseline itself is ~%.3f)\n', ...
        mean(d), max(d), mean(T.LungFraction));
    %  Judge on the DISTRIBUTION, not the maximum: a single outlier among hundreds of
    %  masks says nothing about aggregate results, which average over many images.
    nMaterial = nnz(e > 0.01);
    meanShiftPerImage = mean(d);
    if mean(d) < 5e-4 && nMaterial <= 0.02*height(T)
        fprintf('      -> VERDICT: cosmetic. Extra components are specks (%d of %d masks exceed\n', nMaterial, height(T));
        fprintf('         1%%%% of mask area). Mean baseline shift %.5f moves a mean lift by ~%.5f\n', ...
            meanShiftPerImage, meanShiftPerImage);
        fprintf('         -- below reported precision. Use the masks as provided; no filtering.\n');
    else
        fprintf('      -> VERDICT: material. %d of %d masks exceed 1%%%% of mask area and the mean\n', nMaterial, height(T));
        fprintf('         baseline shift is %.5f. Consider keeping only the two largest\n', meanShiftPerImage);
        fprintf('         components, and report that as a preprocessing step.\n');
    end
    worst = T(frag,:); [~,ix] = sort(worst.ExtraAreaFraction,'descend');
    worst = worst(ix,:);
    fprintf('      worst offenders: %s\n', strjoin(cellstr(worst.Basename(1:min(5,height(worst))))', ', '));
end
fprintf('  with manual ground truth   : %d (%.1f%%)\n', nGT, 100*nGT/height(T));
if nGT > 0
    d = T.Dice(T.HasGroundTruth); j = T.IoU(T.HasGroundTruth);
    fprintf('\n  AGREEMENT: manual reference vs the supplied automatic masks\n');
    fprintf('    Dice : mean %.4f, sd %.4f, min %.4f   (n=%d)\n', mean(d), std(d), min(d), nGT);
    fprintf('    IoU  : mean %.4f, sd %.4f, min %.4f\n', mean(j), std(j), min(j));
    fprintf('    -> this is the mask-substitution robustness number Limitations asks for.\n');
    writematrix(gtNames, fullfile(resDir,'gt_basenames.txt'));
    fprintf('    wrote %s (use as gtListFile in ood_containment_lime.m)\n', ...
        fullfile(resDir,'gt_basenames.txt'));
else
    fprintf(['\n  NO ground-truth masks located. Set gtSingleDirs / gtLeftDir / gtRightDir\n' ...
             '  to the correct paths. Without them the Dice figure cannot be verified here\n' ...
             '  and Methods must attribute it to the original segmentation run rather than\n' ...
             '  present it as reproduced.\n']);
end
fprintf('\nWrote %s\n', fullfile(resDir,'unet_mask_audit.csv'));

%% ======================= local functions =======================
function G = loadGroundTruth(base, singleDirs, leftDir, rightDir)
    G = [];
    for k = 1:numel(singleDirs)
        d = singleDirs{k};
        if isempty(d) || ~isfolder(d), continue; end
        for cand = {[base '.png'], [base '_mask.png']}
            p = fullfile(d, cand{1});
            if isfile(p)
                G = imread(p); if size(G,3)==3, G = rgb2gray(G); end
                return;
            end
        end
    end
    % Montgomery-style split masks: union of left and right
    if ~isempty(leftDir) && isfolder(leftDir) && ~isempty(rightDir) && isfolder(rightDir)
        pl = fullfile(leftDir,[base '.png']); pr = fullfile(rightDir,[base '.png']);
        if isfile(pl) && isfile(pr)
            L = imread(pl); if size(L,3)==3, L = rgb2gray(L); end
            R = imread(pr); if size(R,3)==3, R = rgb2gray(R); end
            R = imresize(R, size(L), 'nearest');
            G = uint8((L>0 | R>0)) * 255;
        end
    end
end
