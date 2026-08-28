%% Scale-matched out-of-distribution control
%
%  WHY. The manuscript states twice that "a scale-matched control that restored the
%  lung to approximately training scale did not recover accuracy, indicating a domain
%  shift rather than an image-scale artefact". That control has never been run for the
%  reported models and no numbers exist for it anywhere. A reviewer correctly
%  identified it as pivotal: without it, the failure on full radiographs could be a
%  trivial consequence of the lung appearing at a different size than in training.
%
%  IT ASKS A PRIOR QUESTION FIRST. If the region-of-interest images are formed by
%  MASKING (X (*) M) rather than cropping, the frame is unchanged and the lung already
%  occupies the same share of it in both conditions -- in which case scale was never a
%  candidate explanation and no control is needed. Step 1 measures that directly. Only
%  if the two differ materially does the scale-matched condition carry any weight.
%
%  HOW THE MATCH IS DONE. Rather than the arbitrary fixed margin used in the original
%  sweep, each full radiograph is cropped around the lung bounding box and expanded
%  symmetrically until the lung occupies the SAME frame share as it does in the
%  region-of-interest training images. The achieved fraction is reported, since border
%  clipping can prevent some images from reaching the target exactly.
%
%  RESULT (measured 2026-08-09): ROI 0.2534 vs full CXR 0.2481, difference 0.005.
%  NO SCALE MISMATCH EXISTS. The region-of-interest images are masked, not cropped, so
%  the lung is already presented at training scale on full radiographs. Step 2 is
%  therefore VACUOUS for this dataset -- cropping to the lung bbox would create a
%  mismatch rather than remove one. The manuscript now reports the two measured
%  fractions instead of an asserted control. Keep this script as the record of how
%  that was established, and re-run Step 1 only if the ROI construction changes.
%
%  OUTPUTS
%    results\scale_matched_control.csv   accuracy per model per condition
%    containment rows merged as Condition = FullCXR-scaled
clc; clear; close all;

%% --- config ---
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
MODELS_TO_RUN = allModels;
roiDir  = 'C:\paper2_repo\input\annotated_gray\annotated_gray';   % recursive {ptb,normal}
cxrDir  = 'C:\paper2_repo\input\cxr_resized';                     % flat, class from _0/_1
maskDir = 'C:\paper2_repo\input\mask';  maskSuffix = '_mask';
resDir  = 'C:\paper2_repo\results';     if ~exist(resDir,'dir'), mkdir(resDir); end
csvCont = fullfile(resDir,'containment_combined.csv');
csvAcc  = fullfile(resDir,'scale_matched_control.csv');

maxImages   = 132;    % same stratified subset as the other containment runs
nRuns       = 3;
numSamples  = 1000;
execEnv     = 'gpu';
DO_CONTAINMENT = true;   % false = accuracy only (fast; accuracy is the claim under review)
FRAC_TOL    = 0.02;      % |ROI frac - FullCXR frac| below this => no scale mismatch exists

cfgs(1) = struct('key','LIME-sp-K30',        'type','lime','seg','superpixels','m',50, 'K',30,'thr',0);
cfgs(2) = struct('key','LIME-grid-K30',      'type','lime','seg','grid',       'm',49, 'K',30,'thr',0);
cfgs(3) = struct('key','LIME-fine-m100-K20', 'type','lime','seg','superpixels','m',100,'K',20,'thr',0);
cfgs(4) = struct('key','GradCAM@0.5',        'type','gradcam','seg','','m',0,'K',0,'thr',0.5);
nCfg = numel(cfgs);

%% --- enumerate both conditions on the same basenames where possible ---
[roiP, roiM, roiAb] = enumRecursive(roiDir, maskDir, maskSuffix, maxImages);
[cxrP, cxrM, cxrAb] = enumFlat(cxrDir, maskDir, maskSuffix, maxImages);
fprintf('ROI images: %d | full CXRs: %d\n', numel(roiP), numel(cxrP));

%% ===== STEP 1: does a scale mismatch actually exist? =====
roiFrac = meanLungFraction(roiP, roiM);
cxrFrac = meanLungFraction(cxrP, cxrM);
fprintf('\n===== STEP 1: lung share of the frame =====\n');
fprintf('  ROI (training condition) : %.4f\n', roiFrac);
fprintf('  Full CXR (native)        : %.4f\n', cxrFrac);
fprintf('  difference               : %+.4f\n', cxrFrac - roiFrac);
scaleMismatch = abs(cxrFrac - roiFrac) > FRAC_TOL;
if ~scaleMismatch
    fprintf(['  -> NO MATERIAL SCALE MISMATCH. The region-of-interest images are masked,\n' ...
             '     not cropped, so the lung already occupies the same share of the frame in\n' ...
             '     both conditions. Scale is excluded by construction and the scale-matched\n' ...
             '     control is not needed to support the domain-shift claim; report the two\n' ...
             '     fractions instead of an extra experiment.\n']);
else
    fprintf(['  -> SCALE MISMATCH PRESENT (%.3f vs %.3f). The control below is required:\n' ...
             '     each full CXR is cropped so the lung reaches the ROI frame share.\n'], roiFrac, cxrFrac);
end

%% ===== STEP 2: build the scale-matched condition and evaluate =====
accRows = {}; contRows = {};
for mi = 1:numel(MODELS_TO_RUN)
    if ~isfile(MODELS_TO_RUN{mi})
        warning('Missing %s -- skipping.', MODELS_TO_RUN{mi}); continue;
    end
    S = load(MODELS_TO_RUN{mi});
    if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    mkey = erase(MODELS_TO_RUN{mi},'.mat');
    gcLayer = gradcamFeatureLayer(net);

    prec = nan(nCfg,numel(cxrP)); iou = prec; lift = prec;
    hitNative = 0; hitScaled = 0; used = 0; achieved = [];

    for i = 1:numel(cxrP)
        g = imread(cxrP{i}); if size(g,3)==3, g=rgb2gray(g); end
        gt = imread(cxrM{i}); if size(gt,3)==3, gt=rgb2gray(gt); end
        gt = imresize(gt, size(g),'nearest') > 0;
        if nnz(gt)==0, continue; end
        used = used + 1;

        % native full CXR
        imgN = uint8(repmat(imresize(g, inputSize),1,1,3));
        hitNative = hitNative + (predAbn(classify(net,imgN)) == cxrAb(i));

        % scale-matched crop: expand the lung bbox until the lung hits roiFrac
        [gS, gtS] = scaleMatchCrop(g, gt, roiFrac);
        achieved(end+1) = nnz(gtS)/numel(gtS); %#ok<SAGROW>
        imgS   = uint8(repmat(imresize(gS, inputSize),1,1,3));
        gtMask = imresize(gtS, inputSize,'nearest') > 0;
        YP = classify(net, imgS);
        hitScaled = hitScaled + (predAbn(YP) == cxrAb(i));

        if DO_CONTAINMENT && nnz(gtMask) > 0
            lf = nnz(gtMask)/numel(gtMask);
            for c = 1:nCfg
                if strcmp(cfgs(c).type,'lime')
                    pp = zeros(nRuns,1); ii = pp;
                    for r = 1:nRuns
                        [~,fMap,fImp] = imageLIME(net,imgS,YP,'Segmentation',cfgs(c).seg, ...
                            'NumFeatures',cfgs(c).m,'NumSamples',numSamples,'ExecutionEnvironment',execEnv);
                        [~,idx] = maxk(fImp, min(cfgs(c).K, numel(fImp)));
                        A = ismember(fMap,idx);
                        pp(r) = nnz(A&gtMask)/max(nnz(A),1);
                        ii(r) = nnz(A&gtMask)/max(nnz(A|gtMask),1);
                    end
                    prec(c,i)=mean(pp); iou(c,i)=mean(ii);
                else
                    m = gradcamMap(net, imgS, YP, inputSize, execEnv, gcLayer);
                    A = m >= cfgs(c).thr;
                    prec(c,i)=nnz(A&gtMask)/max(nnz(A),1); iou(c,i)=nnz(A&gtMask)/max(nnz(A|gtMask),1);
                end
                lift(c,i)=prec(c,i)-lf;
            end
        end
        if mod(used,20)==0, fprintf('  %s: %d/%d\n', mkey, used, numel(cxrP)); end
    end

    aN = hitNative/max(used,1); aS = hitScaled/max(used,1);
    fprintf('\n== %s ==\n', mkey);
    fprintf('   full CXR, native scale      : acc %.3f\n', aN);
    fprintf('   full CXR, scale-matched     : acc %.3f   (achieved lung share %.4f, target %.4f)\n', ...
        aS, mean(achieved), roiFrac);
    fprintf('   recovery toward ROI accuracy: %+.3f\n', aS - aN);
    accRows(end+1,:) = { string(mkey), used, aN, aS, aS-aN, mean(achieved), roiFrac }; %#ok<SAGROW>

    if DO_CONTAINMENT
        for c = 1:nCfg
            L = lift(c,~isnan(lift(c,:)));
            if numel(L)>1 && exist('ttest','file')==2, [~,pv]=ttest(L,0,'Tail','right'); else, pv=NaN; end
            fprintf('   %-20s lift=%+.3f (p=%.2g)\n', cfgs(c).key, mean(L), pv);
            contRows(end+1,:) = { mkey, cfgs(c).key, mean(prec(c,~isnan(prec(c,:))),2), mean(L), pv, ...
                mean(iou(c,~isnan(iou(c,:))),2) }; %#ok<SAGROW>
        end
    end
end

%% --- save ---
Ta = cell2table(accRows, 'VariableNames', {'Model','N','Acc_native','Acc_scaleMatched', ...
    'Recovery','AchievedLungFraction','TargetLungFraction'});
writetable(Ta, csvAcc); fprintf('\nWrote %s\n', csvAcc);
disp(Ta);

if DO_CONTAINMENT && ~isempty(contRows)
    Tn = cell2table(contRows, 'VariableNames', {'Model','Config','Precision','LiftOverChance','pValue','IoU'});
    Tn.Condition = repmat("FullCXR-scaled", height(Tn),1);
    Tn = movevars(Tn,'Condition','After','Model');
    Tn.Model=string(Tn.Model); Tn.Config=string(Tn.Config);
    if isfile(csvCont)
        To = readtable(csvCont,'TextType','string');
        To = To(:, Tn.Properties.VariableNames);
        To(ismember(To.Model,unique(Tn.Model)) & To.Condition=="FullCXR-scaled",:) = [];
        Tn = [To; Tn];
    end
    writetable(Tn, csvCont);
    fprintf('Merged FullCXR-scaled containment into %s\n', csvCont);
end

fprintf('\n==== what to write ====\n');
if ~scaleMismatch
    fprintf(['  The lung occupies %.3f of the frame in the ROI condition and %.3f on full\n' ...
             '  radiographs, so scale is excluded by construction. Replace the asserted\n' ...
             '  control with these two measured numbers.\n'], roiFrac, cxrFrac);
else
    fprintf(['  Report the accuracy column above: if scale-matching does not recover accuracy\n' ...
             '  toward the ROI range, the claim stands and is now evidenced.\n']);
end

%% ======================= local functions =======================
function ab = predAbn(lab)
    ab = ~contains(lower(char(string(lab))),'normal');
end

function f = meanLungFraction(imgPaths, maskPaths)
    v = zeros(numel(imgPaths),1);
    for i = 1:numel(imgPaths)
        g  = imread(imgPaths{i}); if size(g,3)==3, g = rgb2gray(g); end
        gt = imread(maskPaths{i}); if size(gt,3)==3, gt = rgb2gray(gt); end
        gt = imresize(gt, size(g), 'nearest') > 0;
        v(i) = nnz(gt)/numel(gt);
    end
    f = mean(v);
end

function [gC, gtC] = scaleMatchCrop(g, gt, targetFrac)
    % Expand the lung bounding box symmetrically until the lung occupies
    % targetFrac of the cropped frame. Never shrinks below the bbox itself.
    [ys,xs] = find(gt);
    if isempty(ys), gC=g; gtC=gt; return; end
    r0=min(ys); r1=max(ys); c0=min(xs); c1=max(xs);
    bh = r1-r0+1; bw = c1-c0+1;
    targetArea = nnz(gt)/max(targetFrac,eps);
    s = max(sqrt(targetArea/(bh*bw)), 1);
    nh = round(bh*s); nw = round(bw*s);
    cr = round((r0+r1)/2); cc = round((c0+c1)/2);
    r0n = max(1, cr-floor(nh/2)); r1n = min(size(g,1), r0n+nh-1);
    c0n = max(1, cc-floor(nw/2)); c1n = min(size(g,2), c0n+nw-1);
    gC  = g(r0n:r1n, c0n:c1n);
    gtC = gt(r0n:r1n, c0n:c1n);
end

function [ip, mp, ab] = enumRecursive(dir0, maskDir, sfx, cap)
    f = dir(fullfile(dir0,'**','*.png')); f=f(~[f.isdir]);
    [~,o]=sort({f.name}); f=f(o); ip={}; mp={}; ab=[];
    for i=1:numel(f)
        b=erase(f(i).name,'.png'); q=fullfile(maskDir,[b sfx '.png']);
        if ~isfile(q), q2=fullfile(maskDir,[b '.png']); if isfile(q2), q=q2; else, continue; end; end
        [~,par]=fileparts(f(i).folder);
        ip{end+1}=fullfile(f(i).folder,f(i).name); mp{end+1}=q; ab(end+1)=~contains(lower(par),'normal'); %#ok<AGROW>
    end
    [ip,mp,ab]=strat(ip,mp,ab,cap);
end

function [ip, mp, ab] = enumFlat(dir0, maskDir, sfx, cap)
    f = dir(fullfile(dir0,'*.png')); f=f(~[f.isdir]);
    [~,o]=sort({f.name}); f=f(o); ip={}; mp={}; ab=[];
    for i=1:numel(f)
        b=erase(f(i).name,'.png'); q=fullfile(maskDir,[b sfx '.png']);
        if ~isfile(q), q2=fullfile(maskDir,[b '.png']); if isfile(q2), q=q2; else, continue; end; end
        t=split(b,'_');
        ip{end+1}=fullfile(f(i).folder,f(i).name); mp{end+1}=q; ab(end+1)=strcmp(t{end},'1'); %#ok<AGROW>
    end
    [ip,mp,ab]=strat(ip,mp,ab,cap);
end

function [ip,mp,ab]=strat(ip,mp,ab,cap)
    if numel(ip) <= cap, return; end
    iN=find(~ab); iA=find(ab); ord=[];
    for t=1:max(numel(iN),numel(iA))
        if t<=numel(iN), ord(end+1)=iN(t); end %#ok<AGROW>
        if t<=numel(iA), ord(end+1)=iA(t); end %#ok<AGROW>
    end
    ord=ord(1:cap); ip=ip(ord); mp=mp(ord); ab=ab(ord);
end

function name = gradcamFeatureLayer(net)
    name='';
    try
        if isa(net,'SeriesNetwork'), return; end
        L=net.Layers;
        for i=numel(L):-1:1
            if isa(L(i),'nnet.cnn.layer.ReLULayer'), name=L(i).Name; return; end
        end
    catch, name='';
    end
end

function m = gradcamMap(net, img, YP, inputSize, execEnv, gcLayer)
    try
        raw = gradCAM(net, img, YP, 'ExecutionEnvironment', execEnv);
    catch
        try
            if isempty(gcLayer), error('no fallback'); end
            raw = gradCAM(net, img, YP, 'ExecutionEnvironment', execEnv, 'FeatureLayer', gcLayer);
        catch
            m = zeros(inputSize); return;
        end
    end
    m = mat2gray(imresize(double(raw), inputSize));
end
