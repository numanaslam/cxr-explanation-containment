%% Cross-dataset containment on Montgomery (accuracy gate + LIME/Grad-CAM containment)
%
%  WHAT THIS ESTABLISHES. The paper's claims were measured on Shenzhen only. This run
%  repeats the containment measurement on Montgomery -- a different site, different
%  collimation, different mask geometry -- using the SAME four Shenzhen-trained models
%  and the SAME four explainer configs, in both conditions:
%     MG-ROI      lung-masked Montgomery images (the training-style condition)
%     MG-FullCXR  full Montgomery radiographs
%
%  READ WITH THE RELIABILITY GATE. The models were trained on Shenzhen, so Montgomery
%  is a cross-site transfer. The accuracy printed per condition IS the gate: where a
%  model is at/near chance, its containment is not interpretable as grounding (the
%  paper's own Phase 4 rule). If MG-ROI accuracy lands at chance for all models,
%  fine-tune on a Montgomery 80/20 split first (clone the recipe in train_resnet50.m)
%  before reading containment. THE NULL COMPARISON REMAINS VALID EITHER WAY --
%  montgomery_null.m measures a property of the metric's baseline (analytic vs
%  geometry-matched), which does not require an accurate classifier.
%
%  MASKS ARE SAVED. Unlike the Shenzhen pipeline, this script saves every explanation
%  mask (plus the LIME segment map and run-averaged importance vector) to
%  results\mont_masks_<model>_<condition>.mat. montgomery_null.m and
%  montgomery_deletion.m load these instead of re-running LIME, so those two scripts
%  run in minutes with no GPU.
%
%  RUNTIME. LIME dominates: 2 conditions x 4 models x 138 images x 3 configs x 3 runs
%  x 1000 samples. Roughly 2x one regen_containment.m sweep. To split across nights,
%  set CONDITIONS to one condition, or MODELS_TO_RUN to a subset -- all outputs are
%  merge-keyed, nothing is overwritten.
clc; clear; close all;
rng(0);

%% --- config ---
allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
MODELS_TO_RUN = allModels;
CONDITIONS = {'MG-ROI','MG-FullCXR'};    % run both by default; trim to split runtime

mgRoot  = 'C:\paper2_repo\input\montgomery';
roiDir  = fullfile(mgRoot,'roi');            % {ptb,normal} from montgomery_prep.m
cxrDir  = fullfile(mgRoot,'cxr_resized');
maskDir = fullfile(mgRoot,'mask');  maskSuffix = '_mask';
resDir  = 'C:\paper2_repo\results'; if ~exist(resDir,'dir'), mkdir(resDir); end
csvPath = fullfile(resDir,'montgomery_containment.csv');

maxImages  = 138;     % all of Montgomery (it is smaller than the Shenzhen cap)
nRuns      = 3;       % LIME repeats, averaged as in regen_containment.m
numSamples = 1000;
execEnv    = 'gpu';   % gpu-serial only (ADR-006)

% identical configs and keys to regen_containment.m
cfgs(1) = struct('key','LIME-sp-K30',        'type','lime','seg','superpixels','m',50, 'K',30,'thr',0);
cfgs(2) = struct('key','LIME-grid-K30',      'type','lime','seg','grid',       'm',49, 'K',30,'thr',0);
cfgs(3) = struct('key','LIME-fine-m100-K20', 'type','lime','seg','superpixels','m',100,'K',20,'thr',0);
cfgs(4) = struct('key','GradCAM@0.5',        'type','gradcam','seg','','m',0,'K',0,'thr',0.5);
nCfg = numel(cfgs);
dispKey = {'LIME sp K30','LIME grid K30','LIME fine K20','Grad-CAM'};

%% --- run ---
rows = {}; statsRows = {}; perImg = {}; accRows = {};
for ci = 1:numel(CONDITIONS)
    COND = CONDITIONS{ci};
    switch COND
        case 'MG-ROI',     [imgP, mskP, ab] = enumRecursive(roiDir, maskDir, maskSuffix, maxImages);
        case 'MG-FullCXR', [imgP, mskP, ab] = enumFlat(cxrDir, maskDir, maskSuffix, maxImages);
    end
    n = numel(imgP);
    fprintf('\n########  %s : %d images  ########\n', COND, n);
    if n == 0, warning('No images for %s -- run montgomery_prep.m first.', COND); continue; end

    for mi = 1:numel(MODELS_TO_RUN)
        if ~isfile(MODELS_TO_RUN{mi})
            warning('Missing %s -- skipping.', MODELS_TO_RUN{mi}); continue;
        end
        S = load(MODELS_TO_RUN{mi});
        if isfield(S,'netTransfer'), net=S.netTransfer; else, fn=fieldnames(S); net=S.(fn{1}); end
        inputSize = net.Layers(1).InputSize(1:2);
        mkey    = erase(MODELS_TO_RUN{mi},'.mat');
        classes = string(net.Layers(end).Classes);
        abCol   = find(~contains(lower(classes),'normal'), 1);   % score column for AUC
        gcLayer = gradcamFeatureLayer(net);

        prec=nan(nCfg,n); iou=nan(nCfg,n); lift=nan(nCfg,n); rec=nan(nCfg,n);
        lfAll=nan(1,n); hit=nan(1,n); scoreAb=nan(1,n); used=0;

        % saved-mask store for montgomery_null / montgomery_deletion
        MASKS = cell(nCfg,n); FMAPS = cell(nCfg,n); FIMPS = cell(nCfg,n);

        for i = 1:n
            g = imread(imgP{i}); if size(g,3)==3, g=rgb2gray(g); end
            gt = imread(mskP{i}); if size(gt,3)==3, gt=rgb2gray(gt); end
            gt = imresize(gt, size(g),'nearest');
            imgR   = uint8(repmat(imresize(g, inputSize),1,1,3));   % minimal preprocessing
            gtMask = imresize(gt, inputSize,'nearest') > 0;
            if nnz(gtMask)==0, continue; end
            used = used + 1; lf = nnz(gtMask)/numel(gtMask); lfAll(i) = lf;

            [YP, sc] = classify(net, imgR);
            hit(i) = ((~contains(lower(char(string(YP))),'normal')) == ab(i));
            if ~isempty(abCol), scoreAb(i) = sc(abCol); end

            for c = 1:nCfg
                if strcmp(cfgs(c).type,'lime')
                    pp=zeros(nRuns,1); ii=pp; rr=pp; impSum=[]; fMap1=[]; Alast=[];
                    for r = 1:nRuns
                        [~,fMap,fImp] = imageLIME(net,imgR,YP,'Segmentation',cfgs(c).seg, ...
                            'NumFeatures',cfgs(c).m,'NumSamples',numSamples,'ExecutionEnvironment',execEnv);
                        [~,idx] = maxk(fImp, min(cfgs(c).K, numel(fImp)));
                        A = ismember(fMap,idx);
                        pp(r) = nnz(A&gtMask)/max(nnz(A),1);
                        ii(r) = nnz(A&gtMask)/max(nnz(A|gtMask),1);
                        rr(r) = nnz(A&gtMask)/max(nnz(gtMask),1);
                        if isempty(impSum), impSum = fImp; else, impSum = impSum + fImp; end
                        fMap1 = fMap; Alast = A;
                    end
                    prec(c,i)=mean(pp); iou(c,i)=mean(ii); rec(c,i)=mean(rr);
                    MASKS{c,i}=logical(Alast); FMAPS{c,i}=uint8(fMap1); FIMPS{c,i}=impSum/nRuns;
                else
                    m = gradcamMap(net, imgR, YP, inputSize, execEnv, gcLayer);
                    A = m >= cfgs(c).thr;
                    prec(c,i)=nnz(A&gtMask)/max(nnz(A),1); iou(c,i)=nnz(A&gtMask)/max(nnz(A|gtMask),1);
                    rec(c,i)=nnz(A&gtMask)/max(nnz(gtMask),1);
                    MASKS{c,i}=logical(A);
                end
                lift(c,i)=prec(c,i)-lf;
            end
            if mod(used,20)==0, fprintf('  %s: %d/%d\n', mkey, used, n); end
        end

        % ---- reliability gate: cross-site accuracy (+ AUC when available) ----
        keepH = ~isnan(hit); acc = mean(hit(keepH));
        auc = NaN;
        if exist('perfcurve','file')==2 && any(~isnan(scoreAb))
            k2 = ~isnan(scoreAb);
            try, [~,~,~,auc] = perfcurve(double(ab(k2)), scoreAb(k2), 1); catch, auc=NaN; end
        end
        gate = 'INTERPRETABLE'; if acc < 0.60, gate = 'AT/NEAR CHANCE -- containment not readable as grounding'; end
        fprintf('\n== %s (%s) ==  accuracy %.3f | AUC %.3f  -> %s\n', mkey, COND, acc, auc, gate);
        accRows(end+1,:) = { string(mkey), string(COND), nnz(keepH), acc, auc }; %#ok<SAGROW>

        for c = 1:nCfg
            keep = ~isnan(lift(c,:));
            if ~any(keep), continue; end
            Lc = lift(c,keep); abK = logical(ab(keep));
            [pv, pDir, dirStr] = liftTest(Lc);
            [lo, hi] = meanCI(Lc);
            st=''; if pv<1e-13, st='***'; elseif pv<1e-3, st='*'; end
            fprintf('   %-14s lift=%+.3f%s  95%% CI [%+.3f, %+.3f]  (p_%s=%.2g)  recall %.3f\n', ...
                dispKey{c}, mean(Lc), st, lo, hi, dirStr, pDir, mean(rec(c,keep)));
            mA=NaN; mN=NaN; pCl=NaN;
            if any(abK) && any(~abK)
                mA=mean(Lc(abK)); mN=mean(Lc(~abK));
                if exist('ttest2','file')==2, [~,pCl]=ttest2(Lc(abK), Lc(~abK)); end
                fprintf('        by class: abnormal %+.3f (n=%d) | normal %+.3f (n=%d)  diff p=%.2g\n', ...
                    mA, nnz(abK), mN, nnz(~abK), pCl);
            end
            rows(end+1,:) = { mkey, COND, cfgs(c).key, mean(prec(c,keep),2), mean(Lc), pv, mean(iou(c,keep),2) }; %#ok<SAGROW>
            statsRows(end+1,:) = { string(mkey), string(COND), string(cfgs(c).key), nnz(keep), ...
                mean(prec(c,keep),2), mean(rec(c,keep),2), mean(iou(c,keep),2), mean(Lc), ...
                lo, hi, pv, pDir, string(dirStr), mA, mN, pCl }; %#ok<SAGROW>
            for q = find(keep)
                perImg(end+1,:) = { string(mkey), string(COND), string(cfgs(c).key), string(baseOf(imgP{q})), ...
                    string(classOf(ab(q))), lfAll(q), prec(c,q), rec(c,q), iou(c,q), lift(c,q) }; %#ok<SAGROW>
            end
        end

        % ---- save the mask store for the follow-on scripts ----
        BASES = cellfun(@(p) string(baseOf(p)), imgP);
        CFGKEYS = string({cfgs.key}); AB = logical(ab); INPUTSIZE = inputSize;
        matPath = fullfile(resDir, sprintf('mont_masks_%s_%s.mat', mkey, COND));
        save(matPath, 'MASKS','FMAPS','FIMPS','BASES','CFGKEYS','AB','INPUTSIZE','mskP','-v7.3');
        fprintf('   saved explanation masks -> %s\n', matPath);
    end
end

%% --- write CSVs (merge-keyed; montgomery rows never collide with Shenzhen rows) ---
if isempty(rows), error('No results produced.'); end
Tnew = cell2table(rows, 'VariableNames', {'Model','Condition','Config','Precision','LiftOverChance','pValue','IoU'});
Tnew.Model=string(Tnew.Model); Tnew.Condition=string(Tnew.Condition); Tnew.Config=string(Tnew.Config);
writetable(mergeKeyed(csvPath, Tnew, {'Model','Condition','Config'}), csvPath);
fprintf('\nWrote %s\n', csvPath);

Ts = cell2table(statsRows, 'VariableNames', {'Model','Condition','Config','N', ...
    'Precision','Recall','IoU','LiftOverChance','CI95_lo','CI95_hi', ...
    'pValue_gt0','pValue_directional','Direction','Lift_abnormal','Lift_normal','pValue_classDiff'});
writetable(mergeKeyed(fullfile(resDir,'containment_stats.csv'), Ts, {'Model','Condition','Config'}), ...
    fullfile(resDir,'containment_stats.csv'));
Tp = cell2table(perImg, 'VariableNames', {'Model','Condition','Config','Basename', ...
    'Class','LungFraction','Precision','Recall','IoU','Lift'});
writetable(mergeKeyed(fullfile(resDir,'containment_perimage.csv'), Tp, {'Model','Condition','Config'}), ...
    fullfile(resDir,'containment_perimage.csv'));
Ta = cell2table(accRows, 'VariableNames', {'Model','Condition','N','Accuracy','AUC'});
writetable(mergeKeyed(fullfile(resDir,'montgomery_accuracy.csv'), Ta, {'Model','Condition'}), ...
    fullfile(resDir,'montgomery_accuracy.csv'));
fprintf('Wrote containment_stats.csv, containment_perimage.csv, montgomery_accuracy.csv (merged)\n');
fprintf('Next: montgomery_null.m (fast, loads the saved masks)\n');

%% ======================= local functions =======================
function [pGt0, pDir, dirStr] = liftTest(L)
    pGt0=NaN; pDir=NaN; dirStr='gt0';
    if numel(L)<2 || exist('ttest','file')~=2, return; end
    [~,pGt0]=ttest(L,0,'Tail','right');
    if mean(L)>=0, pDir=pGt0; dirStr='gt0';
    else, [~,pDir]=ttest(L,0,'Tail','left'); dirStr='lt0'; end
end

function [lo,hi] = meanCI(L)
    lo=NaN; hi=NaN; n=numel(L); if n<2, return; end
    se=std(L)/sqrt(n);
    if exist('tinv','file')==2, t=tinv(0.975,n-1); else, t=1.96; end
    lo=mean(L)-t*se; hi=mean(L)+t*se;
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

function b = baseOf(p)
    [~,b] = fileparts(p);
end

function c = classOf(isAbn)
    if isAbn, c="abnormal"; else, c="normal"; end
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
