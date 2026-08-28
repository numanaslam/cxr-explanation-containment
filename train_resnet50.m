%% Fine-tune ResNet50 on ROI (lung-only) images -> resnet50_v2.mat
%  PROTOCOL IS A DELIBERATE CLONE OF retrain_v2.m -- the script that actually
%  produced alexnet_v2/vgg16_v2/vgg19_v2. Same seed, same split fraction, same
%  augmenter, same optimiser settings, same WeightLearnRateFactor, same save
%  format. If ResNet50 were trained under a different recipe, any containment
%  difference would confound ARCHITECTURE with TRAINING PROTOCOL and the
%  depth-vs-containment claim would not survive review.
%
%  DO NOT "correct" these to the manuscript's Table 2 values (lr 1e-3, 300 epochs,
%  batch 16, rotations). Table 2 does not describe the code that made the models --
%  see PROJECT_NOTES.md. Fix the manuscript, not this script.
%
%  SAME SPLIT AS THE OTHER THREE: rng(42) + the same imageDatastore folder order +
%  splitEachLabel(0.8,'randomized') reproduces retrain_v2.m's split exactly, so
%  ResNet50's held-out numbers are directly comparable to Table 3's. Verified at
%  run time against the valFiles stored inside the existing _v2 models.
%
%  Requires: Deep Learning Toolbox + "Deep Learning Toolbox Model for ResNet-50
%  Network" support package (run `resnet50` once; MATLAB prompts to install).
clc; clear; close all;
rng(42);   % fixed split for reproducibility / leak-free evaluation (retrain_v2.m)

%% ================== config (mirrors retrain_v2.m) ==================
outFile = 'resnet50_v2.mat';
dataFolders = { ...
    'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\*.png'; ...
    'C:\paper2_repo\input\annotated_gray\annotated_gray\normal\*.png'};
refModel = 'vgg16_v2.mat';   % used only to verify the split matches; '' = skip
positiveClass = "ptb";

miniBatch = 10;      % retrain_v2.m
maxEpochs = 100;     % retrain_v2.m
initialLR = 1e-4;    % retrain_v2.m
trainFrac = 0.8;     % retrain_v2.m

%% ================== data + split ==================
imds = imageDatastore(dataFolders, 'LabelSource', 'foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds, trainFrac, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
fprintf('Classes: %s | train=%d val=%d\n', strjoin(categories(imdsTrain.Labels),', '), ...
    numel(imdsTrain.Labels), numel(imdsValidation.Labels));

% ---- verify we reproduced retrain_v2.m's split (comparability gate) ----
if ~isempty(refModel) && isfile(refModel)
    R = load(refModel, 'valFiles');
    if isfield(R,'valFiles')
        a = sort(string(imdsValidation.Files)); b = sort(string(R.valFiles));
        if numel(a)==numel(b) && all(a==b)
            fprintf('[split check] MATCHES %s -- ResNet50 shares the held-out set.\n', refModel);
        else
            warning(['[split check] Split does NOT match %s. ResNet50 accuracy will not be ' ...
                     'same-split comparable with Table 3. Check that the image folder ' ...
                     'contents are unchanged since retrain_v2.m was run.'], refModel);
        end
    end
end

%% ================== network surgery (DAG-safe) ==================
%  retrain_v2.m used net.Layers(1:end-3) array surgery. That works for AlexNet/VGG
%  (SeriesNetwork) but NOT for ResNet50, which is a branched DAGNetwork -- indexing
%  its Layers array discards the skip connections. layerGraph + replaceLayer is the
%  equivalent operation, and the new head keeps retrain_v2.m's LR factors of 20.
net0 = loadPretrained();
lgraph    = layerGraph(net0);
inputSize = net0.Layers(1).InputSize;

learnName = findLayerByType(lgraph, {'nnet.cnn.layer.FullyConnectedLayer', ...
                                     'nnet.cnn.layer.Convolution2DLayer'});
clsName   = findLayerByType(lgraph, {'nnet.cnn.layer.ClassificationOutputLayer'});
fprintf('Replacing last learnable "%s" and output "%s" (%d classes).\n', ...
    learnName, clsName, numClasses);

lgraph = replaceLayer(lgraph, learnName, ...
    fullyConnectedLayer(numClasses, 'Name','fc_new', ...
        'WeightLearnRateFactor',20, 'BiasLearnRateFactor',20));
lgraph = replaceLayer(lgraph, clsName, classificationLayer('Name','classoutput'));

%% ================== augmentation + datastores ==================
%  retrain_v2.m's augmenter exactly: horizontal reflection + +-30px translation.
%  (NOT the rotations the manuscript describes -- again, the manuscript is wrong.)
%  'gray2rgb' is a no-op when the PNGs are already 3-channel, so it is safe here
%  and protects against a grayscale-vs-RGB channel mismatch.
pixelRange = [-30 30];
augmenter = imageDataAugmenter('RandXReflection',true, ...
    'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing','gray2rgb');
augVal   = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing','gray2rgb');

%% ================== train ==================
valFreq = max(1, floor(numel(imdsTrain.Labels)/miniBatch));
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatch,'MaxEpochs',maxEpochs,'InitialLearnRate',initialLR, ...
    'Shuffle','every-epoch','ValidationData',augVal, ...
    'ValidationFrequency',valFreq,'ExecutionEnvironment','auto', ...
    'Verbose',true,'Plots','training-progress');

fprintf('\n==== Training ResNet50 (sgdm, batch %d, %d epochs, lr %g) ====\n', ...
    miniBatch, maxEpochs, initialLR);
netTransfer = trainNetwork(augTrain, lgraph, options);

%% ================== held-out metrics (tab:clsperf row) ==================
%  Same definitions as evaluate_models.m so the row is directly comparable.
[YPred, scores] = classify(netTransfer, augVal);
YTrue = imdsValidation.Labels(:); YPred = YPred(:);
classNames = categories(YTrue);
C = confusionmat(YTrue, YPred);
posIdx = find(string(classNames) == positiveClass, 1);
if isempty(posIdx), error('Positive class "%s" not in labels.', positiveClass); end
TP = C(posIdx,posIdx); FN = sum(C(posIdx,:))-TP; FP = sum(C(:,posIdx))-TP;
TN = sum(C(:))-TP-FN-FP;
acc  = (TP+TN)/sum(C(:));  prec = TP/max(TP+FP,1);
sens = TP/max(TP+FN,1);    spec = TN/max(TN+FP,1);
f1   = 2*(prec*sens)/max(prec+sens,eps);
posCol = find(string(netTransfer.Layers(end).Classes) == positiveClass, 1);
[~,~,~,AUC] = perfcurve(YTrue, scores(:,posCol), char(positiveClass));

fprintf('\n===== ResNet50 held-out (n=%d, positive = %s) =====\n', numel(YTrue), positiveClass);
fprintf('  acc %.4f | prec %.4f | sens %.4f | spec %.4f | F1 %.4f | AUC %.4f\n', ...
    acc, prec, sens, spec, f1, AUC);
fprintf('  TP=%d TN=%d FP=%d FN=%d\n', TP, TN, FP, FN);
fprintf('\n  tab:clsperf LaTeX row:\n');
fprintf('  ResNet50 & %.3f & %.3f & %.3f & %.3f \\\\\n', acc, prec, sens, spec);
fprintf('\n  -> paste accuracy %.3f into: regen_containment.m (clsperfAcc),\n', acc);
fprintf('     run_all_models_containment.m (clsperfAcc), ood_accuracy_table3.m (allTarget),\n');
fprintf('     make_results_charts.m (roiAcc).\n');

%% ================== save (retrain_v2.m format) ==================
valFiles = imdsValidation.Files; valLabels = imdsValidation.Labels; %#ok<NASGU>
save(outFile, 'netTransfer','valFiles','valLabels','-v7.3');
fprintf('\nSaved -> %s (with valFiles/valLabels, same as the other _v2 models)\n', outFile);

%% ======================= local functions =======================
function net = loadPretrained()
    try
        net = resnet50();
    catch ME
        error(['Could not load pretrained resnet50: %s\n' ...
               'Install Add-Ons > "Deep Learning Toolbox Model for ResNet-50 ' ...
               'Network", then re-run.'], ME.message);
    end
end

function name = findLayerByType(lgraph, typeList)
    % Search from the OUTPUT end -> last learnable / the output layer.
    for i = numel(lgraph.Layers):-1:1
        L = lgraph.Layers(i);
        for t = 1:numel(typeList)
            if isa(L, typeList{t}), name = L.Name; return; end
        end
    end
    error('No layer of type %s found.', strjoin(typeList, '/'));
end
