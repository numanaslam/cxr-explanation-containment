%% Grad-CAM threshold sensitivity: is the architecture ordering an artefact of tau=0.5?
%
%  WHY. Grad-CAM heat maps are binarised at tau = 0.5 to make them comparable with the
%  top-K LIME masks. The paper reports a monotonic ordering of Grad-CAM containment
%  across architectures (AlexNet < VGG16 < VGG19 < ResNet50), and a reader is entitled
%  to ask whether that ordering is a property of the explanations or of the threshold.
%
%  CHEAP TO ANSWER. Grad-CAM is deterministic and needs one forward-backward pass per
%  image, with no perturbation sampling, so sweeping several thresholds costs a small
%  fraction of a LIME run. The maps are computed ONCE per image and thresholded at
%  every tau, so the extra thresholds are almost free.
%
%  WHAT IT REPORTS
%    - mean containment lift at each tau, per architecture
%    - the mask area at each tau, since a higher threshold gives a smaller mask and
%      containment is a precision (small masks score better -- see the recall block of
%      tab:contain)
%    - Kendall's tau between the architecture ordering at each threshold and at 0.5,
%      which is the actual question: does the ORDERING move?
%
%  Interpreting the outcome: if the ordering is identical at every threshold, report
%  that the finding is threshold-stable. If it moves, that is a genuine limitation and
%  must be stated -- do not report only the threshold that agrees with tau = 0.5.
clc; clear; close all;

%% --- config ---
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
MODELS_TO_RUN = allModels;
CONDITION = 'ROI';                    % 'ROI' or 'FullCXR'
roiDir  = 'C:\paper2_repo\input\annotated_gray\annotated_gray';
cxrDir  = 'C:\paper2_repo\input\cxr_resized';
maskDir = 'C:\paper2_repo\input\mask';  maskSuffix = '_mask';
resDir  = 'C:\paper2_repo\results';     if ~exist(resDir,'dir'), mkdir(resDir); end
csvOut  = fullfile(resDir,'gradcam_threshold_sweep.csv');

maxImages = 132;                      % same stratified subset as every other run
TAUS      = [0.3 0.4 0.5 0.6 0.7];
execEnv   = 'gpu';

switch CONDITION
    case 'ROI',     [imgP, mskP, ab] = enumRecursive(roiDir, maskDir, maskSuffix, maxImages);
    case 'FullCXR', [imgP, mskP, ab] = enumFlat(cxrDir, maskDir, maskSuffix, maxImages);
    otherwise, error('CONDITION must be ROI or FullCXR');
end
n = numel(imgP);
fprintf('Grad-CAM threshold sweep | %s | %d images | tau = %s\n', ...
    CONDITION, n, mat2str(TAUS));

%% --- run ---
nT = numel(TAUS);
liftAll = nan(numel(MODELS_TO_RUN), nT);
areaAll = nan(numel(MODELS_TO_RUN), nT);
names   = strings(numel(MODELS_TO_RUN),1);
rows = {};
for mi = 1:numel(MODELS_TO_RUN)
    if ~isfile(MODELS_TO_RUN{mi})
        warning('Missing %s -- skipping.', MODELS_TO_RUN{mi}); continue;
    end
    S = load(MODELS_TO_RUN{mi});
    if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    mkey = erase(MODELS_TO_RUN{mi},'.mat'); names(mi) = string(mkey);
    gcLayer = gradcamFeatureLayer(net);

    lift = nan(nT,n); area = nan(nT,n); used = 0;
    for i = 1:n
        g = imread(imgP{i}); if size(g,3)==3, g=rgb2gray(g); end
        gt = imread(mskP{i}); if size(gt,3)==3, gt=rgb2gray(gt); end
        gt = imresize(gt, size(g),'nearest');
        imgR   = uint8(repmat(imresize(g, inputSize),1,1,3));
        gtMask = imresize(gt, inputSize,'nearest') > 0;
        if nnz(gtMask)==0, continue; end
        used = used + 1; lf = nnz(gtMask)/numel(gtMask);
        YP = classify(net, imgR);

        m = gradcamMap(net, imgR, YP, inputSize, execEnv, gcLayer);   % ONE map per image
        for k = 1:nT
            A = m >= TAUS(k);                                          % thresholds are free
            if nnz(A)==0, continue; end
            lift(k,i) = nnz(A&gtMask)/max(nnz(A),1) - lf;
            area(k,i) = nnz(A)/numel(A);
        end
        if mod(used,40)==0, fprintf('  %s: %d/%d\n', mkey, used, n); end
    end

    fprintf('\n== %s ==\n', mkey);
    for k = 1:nT
        L = lift(k,~isnan(lift(k,:)));
        if isempty(L), continue; end
        if numel(L)>1 && exist('ttest','file')==2, [~,pv]=ttest(L,0,'Tail','right'); else, pv=NaN; end
        liftAll(mi,k) = mean(L); areaAll(mi,k) = mean(area(k,~isnan(area(k,:))));
        fprintf('   tau=%.1f  lift=%+.3f  (p=%.2g)  mask area %.3f of frame\n', ...
            TAUS(k), mean(L), pv, areaAll(mi,k));
        rows(end+1,:) = { string(mkey), string(CONDITION), TAUS(k), nnz(~isnan(lift(k,:))), ...
            mean(L), pv, areaAll(mi,k) }; %#ok<SAGROW>
    end
end

%% --- the actual question: does the ORDERING move with tau? ---
ok = ~isnan(liftAll(:,1));
fprintf('\n===== architecture ordering by Grad-CAM containment =====\n');
ref = find(abs(TAUS-0.5) < 1e-9, 1);
for k = 1:nT
    [~,ord] = sort(liftAll(ok,k), 'ascend');
    nm = names(ok);
    fprintf('  tau=%.1f : %s\n', TAUS(k), strjoin(cellstr(nm(ord))', ' < '));
end
if ~isempty(ref)
    fprintf('\n  Kendall tau of each ordering against the tau=0.5 ordering:\n');
    for k = 1:nT
        t = kendall(liftAll(ok,k), liftAll(ok,ref));
        fprintf('    tau=%.1f -> %+0.2f%s\n', TAUS(k), t, ternary(abs(t-1)<1e-9,'  (identical ordering)',''));
    end
    stable = all(arrayfun(@(k) abs(kendall(liftAll(ok,k), liftAll(ok,ref))-1) < 1e-9, 1:nT));
    fprintf('\n  VERDICT: ordering is %s across tau in [%.1f, %.1f].\n', ...
        ternary(stable,'IDENTICAL','NOT stable'), min(TAUS), max(TAUS));
    if stable
        fprintf('  -> report as threshold-stable in Section III-D.\n');
    else
        fprintf('  -> the ordering depends on the threshold; state it as a limitation and do\n');
        fprintf('     NOT report only the threshold that agrees with tau = 0.5.\n');
    end
end

T = cell2table(rows, 'VariableNames', {'Model','Condition','Tau','N','LiftOverChance','pValue','MaskAreaFraction'});
writetable(T, csvOut); fprintf('\nWrote %s\n', csvOut);

%% ======================= local functions =======================
function t = kendall(a, b)
    m = numel(a); c = 0; d = 0;
    for i = 1:m-1
        for j = i+1:m
            s = (a(i)-a(j))*(b(i)-b(j));
            if s > 0, c = c+1; elseif s < 0, d = d+1; end
        end
    end
    if c+d == 0, t = NaN; else, t = (c-d)/(c+d); end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
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
