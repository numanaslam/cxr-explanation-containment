%% Geometry-matched empirical null for explanation containment
%
%  WHY (reviewer M2a, and it also settles M2b). Containment is currently reported as
%  lift over an ANALYTIC null: precision minus the lung's share of the frame. That
%  null assumes an explanation placed uniformly at random over the frame. Real
%  explanations are not uniform -- LIME superpixels and Grad-CAM activations both
%  carry spatial structure, and both tend toward the image centre, where the lungs
%  also are. The analytic null may therefore credit an explanation for being centred
%  rather than for being anatomical, and overstate "above chance".
%
%  It also bears on the unexplained class effect. Containment differs markedly between
%  abnormal and normal radiographs even after per-image chance correction. Two causes
%  are consistent with that: focal pathology genuinely placing evidence in the lung, or
%  a precision that responds to mask GEOMETRY rather than merely to lung AREA, which
%  the analytic correction does not remove. An empirical null that holds geometry fixed
%  distinguishes them: if the class effect survives against a geometry-matched null, it
%  is not a geometric artefact.
%
%  THE NULLS. Each preserves properties of the real explanation mask A and destroys
%  only its relationship to the anatomy. No forward passes are needed -- these are set
%  operations on masks -- so a large number of draws costs almost nothing and the
%  runtime is dominated by regenerating A with LIME.
%
%    ROTATION (primary)   Rotate A about the image centre by a random angle. Preserves
%                         area, shape, and DISTANCE FROM CENTRE. This is the null that
%                         controls centre bias: a mask credited only for being central
%                         scores as well under rotation as it does in place.
%                         Angles are drawn from [30,150] u [210,330] degrees; angles
%                         near 0 leave the mask in place, and angles near 180 would map
%                         the roughly bilaterally symmetric lung field onto itself.
%
%    TRANSLATION          Circularly shift A by a random offset. Preserves area exactly
%                         and shape, but not distance from centre, so it is the weaker
%                         control and is reported for comparison.
%
%  READ IT AS. lift_analytic  = precision - lung fraction        (what the paper reports)
%              lift_null      = precision - mean null precision  (geometry-matched)
%              p_emp          = fraction of null draws reaching the observed precision
%  If lift_null is close to lift_analytic, the analytic baseline was adequate. If it is
%  much smaller, containment was partly an artefact of where explanations sit generally.
clc; clear; close all;
rng(0);

%% --- config ---
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
MODELS_TO_RUN = allModels;
CONDITION  = 'ROI';      % 'ROI' or 'FullCXR'
roiDir  = 'C:\paper2_repo\input\annotated_gray\annotated_gray';
cxrDir  = 'C:\paper2_repo\input\cxr_resized';
maskDir = 'C:\paper2_repo\input\mask';  maskSuffix = '_mask';
resDir  = 'C:\paper2_repo\results';     if ~exist(resDir,'dir'), mkdir(resDir); end
csvOut  = fullfile(resDir,'geometry_matched_null.csv');

maxImages  = 132;    % same stratified subset as every other containment run
nRuns      = 3;      % LIME repeats, as elsewhere
numSamples = 1000;
nDraws     = 100;    % null draws per image per config -- free, no network calls
execEnv    = 'gpu';

cfgs(1) = struct('key','LIME-sp-K30',        'type','lime','seg','superpixels','m',50, 'K',30,'thr',0);
cfgs(2) = struct('key','LIME-grid-K30',      'type','lime','seg','grid',       'm',49, 'K',30,'thr',0);
cfgs(3) = struct('key','LIME-fine-m100-K20', 'type','lime','seg','superpixels','m',100,'K',20,'thr',0);
cfgs(4) = struct('key','GradCAM@0.5',        'type','gradcam','seg','','m',0,'K',0,'thr',0.5);
nCfg = numel(cfgs);
dispKey = {'LIME sp K30','LIME grid K30','LIME fine K20','Grad-CAM'};

%% --- enumerate (identical deterministic subset as the other scripts) ---
switch CONDITION
    case 'ROI',     [imgP, mskP, ab] = enumRecursive(roiDir, maskDir, maskSuffix, maxImages);
    case 'FullCXR', [imgP, mskP, ab] = enumFlat(cxrDir, maskDir, maskSuffix, maxImages);
    otherwise, error('CONDITION must be ROI or FullCXR');
end
n = numel(imgP);
fprintf('Geometry-matched null | condition %s | %d images | %d null draws each\n', ...
    CONDITION, n, nDraws);

%% --- run ---
rows = {};
for mi = 1:numel(MODELS_TO_RUN)
    if ~isfile(MODELS_TO_RUN{mi})
        warning('Missing %s -- skipping.', MODELS_TO_RUN{mi}); continue;
    end
    S = load(MODELS_TO_RUN{mi});
    if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);
    mkey = erase(MODELS_TO_RUN{mi},'.mat');
    gcLayer = gradcamFeatureLayer(net);

    obs  = nan(nCfg,n);   % observed precision
    nulR = nan(nCfg,n);   % mean null precision, rotation
    nulT = nan(nCfg,n);   % mean null precision, translation
    pEmp = nan(nCfg,n);   % empirical p against the rotation null
    lfA  = nan(1,n);
    used = 0;

    for i = 1:n
        g = imread(imgP{i}); if size(g,3)==3, g=rgb2gray(g); end
        gt = imread(mskP{i}); if size(gt,3)==3, gt=rgb2gray(gt); end
        gt = imresize(gt, size(g),'nearest');
        imgR   = uint8(repmat(imresize(g, inputSize),1,1,3));
        gtMask = imresize(gt, inputSize,'nearest') > 0;
        if nnz(gtMask)==0, continue; end
        used = used + 1; lfA(i) = nnz(gtMask)/numel(gtMask);
        YP = classify(net, imgR);

        for c = 1:nCfg
            % ---- the real explanation mask ----
            if strcmp(cfgs(c).type,'lime')
                accP = 0; Alast = [];
                for r = 1:nRuns
                    [~,fMap,fImp] = imageLIME(net,imgR,YP,'Segmentation',cfgs(c).seg, ...
                        'NumFeatures',cfgs(c).m,'NumSamples',numSamples,'ExecutionEnvironment',execEnv);
                    [~,idx] = maxk(fImp, min(cfgs(c).K, numel(fImp)));
                    A = ismember(fMap,idx);
                    accP = accP + nnz(A&gtMask)/max(nnz(A),1);
                    Alast = A;      % nulls are built from the final run's mask
                end
                obs(c,i) = accP/nRuns; A = Alast;
            else
                m = gradcamMap(net, imgR, YP, inputSize, execEnv, gcLayer);
                A = m >= cfgs(c).thr;
                obs(c,i) = nnz(A&gtMask)/max(nnz(A),1);
            end
            if nnz(A)==0, continue; end

            % ---- geometry-matched nulls (no network calls) ----
            pr = zeros(nDraws,1); pt = zeros(nDraws,1);
            for d = 1:nDraws
                Ar = rotateNull(A);
                At = translateNull(A);
                pr(d) = nnz(Ar & gtMask)/max(nnz(Ar),1);
                pt(d) = nnz(At & gtMask)/max(nnz(At),1);
            end
            nulR(c,i) = mean(pr); nulT(c,i) = mean(pt);
            pEmp(c,i) = (1 + nnz(pr >= obs(c,i))) / (nDraws + 1);
        end
        if mod(used,20)==0, fprintf('  %s: %d/%d\n', mkey, used, n); end
    end

    fprintf('\n== %s (%s) ==\n', mkey, CONDITION);
    for c = 1:nCfg
        keep = ~isnan(obs(c,:)) & ~isnan(nulR(c,:));
        if ~any(keep), continue; end
        o = obs(c,keep); nr = nulR(c,keep); nt = nulT(c,keep);
        la = o - lfA(keep);          % the paper's analytic lift
        ln = o - nr;                 % geometry-matched lift (rotation null)
        lt = o - nt;                 % translation null
        abK = logical(ab(keep));
        if numel(ln)>1 && exist('ttest','file')==2, [~,pv]=ttest(ln,0,'Tail','right'); else, pv=NaN; end
        fprintf('   %-14s precision %.3f | analytic lift %+.3f | ROTATION null %.3f -> lift %+.3f (p=%.2g)\n', ...
            dispKey{c}, mean(o), mean(la), mean(nr), mean(ln), pv);
        fprintf('                    translation null %.3f -> lift %+.3f | median empirical p %.3f\n', ...
            mean(nt), mean(lt), median(pEmp(c,keep)));
        %  The percentage divides by the analytic lift, so it is meaningless when that
        %  lift is itself ~0 (it produced 996%% and 212%% on the ROI run). Only report it
        %  when there is a lift worth apportioning; otherwise give the absolute numbers.
        if mean(la) > 0.02
            fprintf('                    geometry accounts for %.0f%%%% of the analytic lift (%.3f of %.3f)\n', ...
                100*(1 - mean(ln)/mean(la)), mean(la)-mean(ln), mean(la));
        else
            fprintf('                    analytic lift ~0 (%.3f); null-lift %+.3f -- ratio not meaningful\n', ...
                mean(la), mean(ln));
        end
        mA=NaN; mN=NaN; pCl=NaN;
        if any(abK) && any(~abK)
            mA=mean(ln(abK)); mN=mean(ln(~abK));
            if exist('ttest2','file')==2, [~,pCl]=ttest2(ln(abK), ln(~abK)); end
            fprintf('                    by class (vs null): abnormal %+.3f | normal %+.3f  diff p=%.2g\n', mA, mN, pCl);
        end
        rows(end+1,:) = { string(mkey), string(CONDITION), string(cfgs(c).key), nnz(keep), ...
            mean(o), mean(lfA(keep)), mean(la), mean(nr), mean(ln), pv, mean(nt), mean(lt), ...
            median(pEmp(c,keep)), mA, mN, pCl }; %#ok<SAGROW>
    end
end

%% --- save (merge-keyed, like the other result files) ---
if isempty(rows), error('No results produced.'); end
T = cell2table(rows, 'VariableNames', {'Model','Condition','Config','N', ...
    'Precision','LungFraction','Lift_analytic','NullPrec_rotation','Lift_vs_rotation', ...
    'pValue_vs_rotation','NullPrec_translation','Lift_vs_translation','MedianEmpiricalP', ...
    'Lift_abnormal','Lift_normal','pValue_classDiff'});
writetable(mergeKeyed(csvOut, T, {'Model','Condition','Config'}), csvOut);
fprintf('\nWrote %s\n', csvOut);

fprintf('\n==== how to read this ====\n');
fprintf(['  Lift_vs_rotation close to Lift_analytic : the published chance baseline was\n' ...
         '    adequate; containment is anatomical, not merely central.\n' ...
         '  Lift_vs_rotation much smaller           : the analytic null overstated the effect,\n' ...
         '    and part of the reported containment reflects where explanations sit in general.\n' ...
         '  Class difference SURVIVES the null      : the abnormal/normal gap is not a geometric\n' ...
         '    artefact -- it is a property of what the models respond to.\n' ...
         '  Class difference DISAPPEARS             : the gap was geometry, and the per-image\n' ...
         '    area correction was insufficient. Either result is reportable.\n']);

%% ======================= local functions =======================
function A2 = rotateNull(A)
    % Rotate about the image centre, preserving area, shape and radial distance.
    % Angles avoid ~0 (mask unmoved) and ~180 (the lung field is roughly bilaterally
    % symmetric, so a half turn would map it substantially onto itself).
    if rand < 0.5, th = 30 + 120*rand; else, th = 210 + 120*rand; end
    A2 = imrotate(A, th, 'nearest', 'crop');
end

function A2 = translateNull(A)
    % Circular shift: preserves area exactly, destroys position. Weaker than rotation
    % because it does not preserve distance from the image centre.
    A2 = circshift(A, [randi(size(A,1))-1, randi(size(A,2))-1]);
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
