%% Montgomery dataset preparation for the cross-dataset containment experiment
%
%  WHY. Every containment result in the paper rests on one dataset (Shenzhen). Both
%  desk rejections questioned the contribution's generality, and the strongest
%  possible answer is the centrality finding replicated on a SECOND dataset with
%  different mask geometry. Montgomery is the natural pair: same era, same task
%  (TB vs normal), same NLM provenance, DIFFERENT site and collimation -- and it
%  ships MANUAL lung masks (left+right), so the containment pipeline runs on it
%  without training anything.
%
%  WHAT THIS SCRIPT DOES (one-time, CPU-only, ~1 minute):
%    1. Combines leftMask|rightMask into single lung masks, resized to 512x512,
%       written as <base>_mask.png  -- the shape every downstream script expects.
%    2. Writes 512x512 grayscale full CXRs (the MG-FullCXR condition).
%    3. Writes ROI images by MASKING, NOT CROPPING (pixels outside lung set to 0),
%       into {ptb,normal} subfolders -- mirroring the Shenzhen annotated_gray
%       layout so the recursive enumerators work unchanged.
%    4. AUDITS the masks: per-class lung-area fraction with a two-sample t-test.
%       This is the replication target for the paper's class-dependent chance
%       level (Shenzhen: 0.272 abnormal vs 0.232 normal, p ~ 1e-17). Whether
%       Montgomery shows the same gap is itself a reportable result.
%
%  RUN ORDER:  montgomery_prep  ->  montgomery_containment  ->  montgomery_null
%              -> montgomery_deletion.
%
%  INPUT LAYOUT (standard NLM/Kaggle download):
%    <RAW>\CXR_png\MCUCXR_####_0.png            (suffix _0 = normal, _1 = TB)
%    <RAW>\ManualMask\leftMask\MCUCXR_####_?.png
%    <RAW>\ManualMask\rightMask\MCUCXR_####_?.png
clc; clear; close all;

%% --- config ---
RAW    = 'C:\paper2_repo\input\MontgomerySet';       % adjust to your download path
OUT    = 'C:\paper2_repo\input\montgomery';          % everything this script creates
SZ     = 512;                                        % matches the Shenzhen pipeline
resDir = 'C:\paper2_repo\results'; if ~exist(resDir,'dir'), mkdir(resDir); end

cxrDir   = fullfile(RAW,'CXR_png');
lDir     = fullfile(RAW,'ManualMask','leftMask');
rDir     = fullfile(RAW,'ManualMask','rightMask');
maskOut  = fullfile(OUT,'mask');
fullOut  = fullfile(OUT,'cxr_resized');
roiOutA  = fullfile(OUT,'roi','ptb');
roiOutN  = fullfile(OUT,'roi','normal');
for d = {maskOut, fullOut, roiOutA, roiOutN}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

%% --- build ---
files = dir(fullfile(cxrDir,'*.png')); files = files(~[files.isdir]);
[~,o] = sort({files.name}); files = files(o);
fprintf('Montgomery prep: %d CXRs found in %s\n', numel(files), cxrDir);
if numel(files) ~= 138
    warning('Expected 138 Montgomery images, found %d -- check RAW path.', numel(files));
end

nOK = 0; skipped = {};
lf = []; abn = []; ncomp = []; bases = {};
for f = 1:numel(files)
    base = erase(files(f).name,'.png');
    lp = fullfile(lDir,[base '.png']); rp = fullfile(rDir,[base '.png']);
    if ~isfile(lp) || ~isfile(rp)
        skipped{end+1} = base; continue; %#ok<SAGROW>
    end

    g = imread(fullfile(cxrDir,files(f).name));
    if size(g,3)==3, g = rgb2gray(g); end
    L = imread(lp); if size(L,3)==3, L = rgb2gray(L); end
    R = imread(rp); if size(R,3)==3, R = rgb2gray(R); end

    % combine at native resolution, then resize once (nearest keeps it binary)
    M  = (L>0) | (R>0);
    g5 = imresize(g, [SZ SZ]);
    M5 = imresize(uint8(M)*255, [SZ SZ], 'nearest') > 0;

    isAb = endsWith(base,'_1');
    roi  = g5; roi(~M5) = 0;                       % MASKING, not cropping (paper design)

    imwrite(uint8(M5)*255, fullfile(maskOut,[base '_mask.png']));
    imwrite(g5,  fullfile(fullOut,[base '.png']));
    if isAb, imwrite(roi, fullfile(roiOutA,[base '.png']));
    else,    imwrite(roi, fullfile(roiOutN,[base '.png'])); end

    nOK = nOK + 1;
    lf(end+1)   = nnz(M5)/numel(M5); %#ok<SAGROW>
    abn(end+1)  = isAb;              %#ok<SAGROW>
    cc = bwconncomp(M5); ncomp(end+1) = cc.NumObjects; %#ok<SAGROW>
    bases{end+1} = base;             %#ok<SAGROW>
end

%% --- audit ---
abn = logical(abn);
fprintf('\n== Montgomery audit ==\n');
fprintf('prepared %d images (%d abnormal, %d normal); skipped (no mask pair): %d\n', ...
    nOK, nnz(abn), nnz(~abn), numel(skipped));
if ~isempty(skipped), fprintf('  skipped: %s\n', strjoin(skipped,', ')); end
fprintf('mask components: %d images with >2 (expect 2 = two lungs)\n', nnz(ncomp>2));

fprintf('\nlung-area fraction (the per-image chance level):\n');
fprintf('  overall  %.4f   [Shenzhen comparison: 0.245]\n', mean(lf));
pCl = NaN;
if any(abn) && any(~abn)
    if exist('ttest2','file')==2, [~,pCl] = ttest2(lf(abn), lf(~abn)); end
    fprintf('  abnormal %.4f (n=%d) | normal %.4f (n=%d) | diff %+0.4f, p=%.2g\n', ...
        mean(lf(abn)), nnz(abn), mean(lf(~abn)), nnz(~abn), ...
        mean(lf(abn))-mean(lf(~abn)), pCl);
    fprintf('  [Shenzhen replication target: 0.272 vs 0.232, diff +0.040, p ~ 1e-17]\n');
end

T = table(string(bases)', abn', lf', ncomp', ...
    'VariableNames', {'Basename','Abnormal','LungFraction','MaskComponents'});
writetable(T, fullfile(resDir,'montgomery_audit.csv'));
fprintf('\nWrote %s\n', fullfile(resDir,'montgomery_audit.csv'));
fprintf('Next: montgomery_containment.m\n');
