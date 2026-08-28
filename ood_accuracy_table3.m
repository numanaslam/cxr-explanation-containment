%% Recompute classification accuracy (ROI + full-CXR/OOD) with a Table-3-consistent pipeline.
%  GOAL: first reproduce Table 3 ROI accuracy (AlexNet 0.850 / VGG16 0.885 / VGG19 0.911);
%  once the ROI numbers MATCH, the full-CXR (OOD) accuracy from the SAME pipeline is the
%  consistent value to put in tab:acc.
%
%  NOTE: the surest route is to reuse your ORIGINAL Table-3 evaluation code and point it at
%  the full-CXR test images. Use this script only as a standalone re-evaluation. The ONE thing
%  that matters is that the preprocessing below matches how the models were trained/evaluated.
clc; clear;

allModels = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
allNamesD = {'AlexNet','VGG16','VGG19','ResNet50'};
%  tab:clsperf's published accuracies. RESOLVED 2026-08-09 (ADR-010): these come from the
%  older *_net.mat models on a 132-image split; the *_v2 models evaluated here have their own
%  113-image split and legitimately give different numbers. A "MISMATCH" below is therefore
%  EXPECTED and is NOT a preprocessing fault -- do not chase it by changing doTrim/doImadjust.
%  The fix is to replace tab:clsperf with evaluate_models.m's _v2 output.
%  Now set to evaluate_models.m's _v2 held-out accuracies (the training-consistent
%  pipeline). Small residual deltas here are EXPECTED: this script resizes with
%  imresize after rgb2gray, while training/evaluate_models use augmentedImageDatastore
%  + gray2rgb, which flips 0-3 borderline images out of 113.
allTarget = [0.805 0.885 0.867 0.814];
MODELS_TO_RUN = allModels;             % <-- set to {'resnet50_v2.mat'} to evaluate only ResNet50
tol    = 0.03;

sel    = ismember(allModels, MODELS_TO_RUN);
models = allModels(sel); namesD = allNamesD(sel); target = allTarget(sel);
resDir = 'C:\paper2_repo\results';  if ~exist(resDir,'dir'), mkdir(resDir); end
csvPath = fullfile(resDir,'ood_accuracy.csv');

% Held-out test split (n=132). ROI = lung-cropped (in-distribution); CXR = full radiograph (OOD).
roiDir = 'C:\paper2_repo\input\annotated_gray\annotated_gray';  % recursive over {ptb,normal}
cxrDir = 'C:\paper2_repo\input\cxr';                            % flat; class from _0/_1 suffix
testListFile = '';        % optional: text file of your exact test basenames (one per line). '' = all found.
abnormalClass = '';       % '' = auto ('ptb'/'1' => abnormal); set explicitly if auto is wrong.

%% ---- preprocessing: MATCH your Table-3 / training pipeline ----
%  If the ROI accuracy printed below does NOT match Table 3, adjust these until it does.
%  (Our containment-eval used trim+imadjust and gave ROI ~0.74; minimal resize often gives ~0.85.)
doTrim     = false;       % border crop
doImadjust = false;       % contrast normalisation
%  (resize to inputSize + gray->rgb is always applied)

testList = {};
if ~isempty(testListFile) && isfile(testListFile)
    testList = strtrim(string(splitlines(fileread(testListFile))));
    testList = cellstr(testList(testList~=""));
end

rows = {};
for mi = 1:numel(models)
    if ~isfile(models{mi})
        warning('Missing %s -- skipping. (Run train_resnet50.m first?)', models{mi});
        continue;
    end
    S = load(models{mi});
    if isfield(S,'netTransfer'), net = S.netTransfer; else, fn = fieldnames(S); net = S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);

    % HELD-OUT SPLIT: every _v2 model stores its own validation file list
    % (retrain_v2.m saves valFiles/valLabels inside the .mat). Use it, exactly
    % as evaluate_models.m does -- otherwise the ROI row is scored on images
    % the model trained on and comes out inflated.
    mkey = erase(models{mi},'.mat');
    thisList = testList;
    if isempty(thisList) && isfield(S,'valFiles')
        thisList = cell(numel(S.valFiles),1);
        for q = 1:numel(S.valFiles), [~, thisList{q}] = fileparts(S.valFiles{q}); end
        fprintf('[%s: using the split stored in the .mat (%d held-out images) for the ROI row]\n', ...
            mkey, numel(thisList));
    elseif isempty(thisList)
        warning('%s has no saved valFiles -- ROI row includes training images (inflated).', mkey);
    end

    [aR,pR,seR,spR,nR] = evalSet(net, inputSize, roiDir, true,  'folder',   thisList, abnormalClass, doTrim, doImadjust);
    [aC,pC,seC,spC,nC] = evalSet(net, inputSize, cxrDir, false, 'filename', testList,  abnormalClass, doTrim, doImadjust);

    fprintf('\n== %s ==\n', namesD{mi});
    if isnan(target(mi))
        fprintf('  ROI (n=%d):     acc=%.3f  prec=%.3f  sens=%.3f  spec=%.3f   [no Table 3 target -- new model]\n', ...
            nR, aR, pR, seR, spR);
    else
        d = aR - target(mi);
        verdict = 'MATCH';
        if abs(d) > tol, verdict = 'differs (expected: published row is the _net model)'; end
        fprintf('  ROI (n=%d):     acc=%.3f  prec=%.3f  sens=%.3f  spec=%.3f   [vs Table 3 %.3f, delta %+.3f -> %s]\n', ...
            nR, aR, pR, seR, spR, target(mi), d, verdict);
    end
    fprintf('  FullCXR (n=%d): acc=%.3f  prec=%.3f  sens=%.3f  spec=%.3f\n', nC, aC, pC, seC, spC);
    rows(end+1,:) = { string(namesD{mi}), string(mkey), aR, aC, pC, seC, spC, nR, nC }; %#ok<SAGROW>
end

%% ---- merge with previous runs so un-rerun models keep their numbers ----
if isempty(rows)
    error('No models were evaluated (all missing?). Nothing written to %s.', csvPath);
end
Tnew = cell2table(rows, 'VariableNames', ...
    {'Name','Model','ROI_Accuracy','OOD_Accuracy','OOD_Precision','OOD_Sensitivity','OOD_Specificity','N_ROI','N_OOD'});
Tnew.Name = string(Tnew.Name); Tnew.Model = string(Tnew.Model);
if isfile(csvPath)
    Told = readtable(csvPath, 'TextType','string');
    Told.Name = string(Told.Name); Told.Model = string(Told.Model);
    Told = Told(:, Tnew.Properties.VariableNames);      % match column order
    Told(ismember(Told.Model, Tnew.Model),:) = [];
    Tall = [Told; Tnew];
else
    Tall = Tnew;
end
[~,ord] = ismember(allNamesD, cellstr(Tall.Name)); ord = ord(ord>0); Tall = Tall(ord,:);
writetable(Tall, csvPath);
fprintf('\nWrote %s (merged)\n', csvPath);

fprintf('\n==== tab:acc OOD row (use only if all ROI rows MATCH Table 3) ====\n');
fprintf('Condition & %s \\\\\n', strjoin(cellstr(Tall.Name)', ' & '));
fprintf('Full CXR (out-of-distribution) & %s \\\\\n', ...
    strjoin(arrayfun(@(x) sprintf('%.2f',x), Tall.OOD_Accuracy', 'uni',0), ' & '));

%% ================= helpers =================
function [acc,prec,sens,spec,n] = evalSet(net, inputSize, dir0, recursive, classFrom, testList, abnormalClass, doTrim, doImadjust)
    if recursive, f = dir(fullfile(dir0,'**','*.png')); else, f = dir(fullfile(dir0,'*.png')); end
    f = f(~[f.isdir]);
    yTrue = []; yPred = [];
    for i = 1:numel(f)
        base = erase(f(i).name,'.png');
        if ~isempty(testList) && ~any(strcmp(base, testList)), continue; end
        if strcmp(classFrom,'folder')
            [~,par] = fileparts(f(i).folder); ab = ~contains(lower(par),'normal');
        else
            t = split(base,'_'); ab = strcmp(t{end},'1');
        end
        g = imread(fullfile(f(i).folder,f(i).name)); if size(g,3)==3, g = rgb2gray(g); end
        if doTrim
            bw = g > (min(g(:))+5); ys=find(any(bw,2)); xs=find(any(bw,1));
            if ~isempty(ys)&&~isempty(xs), g = g(ys(1):ys(end), xs(1):xs(end)); end
        end
        if doImadjust, g = imadjust(g); end
        imgR = uint8(repmat(imresize(g, inputSize),1,1,3));
        pab = predLabelIsAbnormal(classify(net, imgR), abnormalClass);
        yTrue(end+1) = ab; yPred(end+1) = pab; %#ok<AGROW>
    end
    yTrue = logical(yTrue); yPred = logical(yPred); n = numel(yTrue);
    TP = nnz(yPred & yTrue); TN = nnz(~yPred & ~yTrue);
    FP = nnz(yPred & ~yTrue); FN = nnz(~yPred & yTrue);
    acc  = (TP+TN)/max(n,1);
    prec = TP/max(TP+FP,1); sens = TP/max(TP+FN,1); spec = TN/max(TN+FP,1);
end

function ab = predLabelIsAbnormal(lab, abnormalClass)
    s = lower(strtrim(char(string(lab))));
    if ~isempty(abnormalClass), ab = strcmpi(s, lower(strtrim(abnormalClass))); return; end
    if any(strcmp(s,{'0','normal','healthy','neg','negative','control'})), ab = false; return; end
    if contains(s,'normal') || contains(s,'health'), ab = false; return; end
    ab = true;
end
