%% evaluate_models.m
% Compute classification metrics (precision, sensitivity, specificity,
% F1, accuracy, AUC) for the trained CXR classifiers (AlexNet / VGG16 /
% VGG19), for the CMPB structured abstract.
%
% Positive class = "ptb" (tuberculosis):
%   Sensitivity = fraction of TB cases correctly flagged
%   Specificity = fraction of normal cases correctly cleared
%
% IN-DISTRIBUTION evaluation: models were trained on ROI images, so the
% headline precision/sensitivity/specificity come from the ROI folder.
% To also report the OOD drop, re-run with the cxr\ (full-image) folders.

rng(42);

%% 1. Config ---------------------------------------------------------------
positiveClass = "ptb";
useGray2RGB   = true;     % grayscale PNGs -> 3-channel net input (fixes the size error)

% In-distribution test images (ROI / lung-segmented). NOTE: verify the ptb
% path - your message listed "C:\numan\input\ptb\normal", likely a typo for roi\ptb.
roiFolders = { ...
    'C:\numan\input\roi\normal\*.png'; ...
    'C:\numan\input\roi\ptb\*.png'};

% Trained networks: {display name, .mat file}
%  RE-POINTED TO THE _v2 MODELS (2026-08-09, ADR-010).
%  This script originally loaded the *_net.mat generation, and its output
%  (classification_metrics.csv) is what tab:clsperf reports. But EVERY containment
%  result uses the *_v2 models, which are a different training run on a different
%  image set (roi ~660 images / 132 held-out vs annotated_gray ~565 / 113 held-out).
%  The manuscript presented both as one set of networks. Re-running against _v2
%  makes the classification table and the containment tables describe the same
%  models. The old _net output is preserved in classification_metrics.csv as
%  provenance; this writes to a separate file.
nets = { ...
    'AlexNet',  'alexnet_v2.mat';  ...
    'VGG16',    'vgg16_v2.mat';    ...
    'VGG19',    'vgg19_v2.mat';    ...
    'ResNet50', 'resnet50_v2.mat'};
outCsv = 'classification_metrics_v2.csv';

%% 2. Held-out set ---------------------------------------------------------
% Prefer the exact split saved by train_models.m (leak-free). Fall back to a
% fresh split only if the .mat has no saved split.
S0 = load(nets{1, 2});
if isfield(S0, 'valFiles')
    imdsValidation = imageDatastore(S0.valFiles);
    imdsValidation.Labels = S0.valLabels;
    if isfield(S0, 'useGray2RGB'), useGray2RGB = S0.useGray2RGB; end
    fprintf('Using saved held-out split: %d images.\n', numel(imdsValidation.Files));
else
    imds = imageDatastore(roiFolders, 'LabelSource', 'foldernames');
    [~, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
    warning(['No saved split in %s - re-splitting with rng(42). ' ...
        'Valid only if this matches the training split (else leakage).'], nets{1,2});
end

%% 3. Evaluate each model --------------------------------------------------
rows = cell(size(nets, 1), 7);

for k = 1:size(nets, 1)
    S   = load(nets{k, 2});
    net = S.netTransfer;                       % <-- adjust if saved under another name

    m = evaluateClassifier(net, imdsValidation, positiveClass, useGray2RGB);

    fprintf('\n=== %s  (positive class = %s) ===\n', nets{k,1}, m.PositiveClass);
    fprintf('Accuracy    : %.4f\n', m.Accuracy);
    fprintf('Precision   : %.4f\n', m.Precision);
    fprintf('Sensitivity : %.4f   (recall / TPR)\n', m.Sensitivity);
    fprintf('Specificity : %.4f   (TNR)\n', m.Specificity);
    fprintf('F1 score    : %.4f\n', m.F1);
    fprintf('AUC         : %.4f\n', m.AUC);
    fprintf('TP=%d  TN=%d  FP=%d  FN=%d\n', m.TP, m.TN, m.FP, m.FN);

    figure('Name', nets{k,1});
    confusionchart(m.ConfusionMatrix, m.ClassNames, ...
        'Title', nets{k,1}, ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    rows(k, :) = {nets{k,1}, m.Accuracy, m.Precision, m.Sensitivity, ...
                  m.Specificity, m.F1, m.AUC};
end

%% 4. Collate and save -----------------------------------------------------
resultsTable = cell2table(rows, 'VariableNames', ...
    {'Model','Accuracy','Precision','Sensitivity','Specificity','F1','AUC'});
disp(resultsTable);
writetable(resultsTable, outCsv);
fprintf('\nSaved metrics to %s\n', outCsv);

% ---- LaTeX row for tab:clsperf ----
fprintf('\n==== tab:clsperf rows (n=%d held-out) ====\n', numel(imdsValidation.Files));
for k = 1:size(rows,1)
    fprintf('%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n', rows{k,1}, rows{k,2:7});
end
fprintf('NOTE: update the tab:clsperf caption -- it currently says n=132 (the _net split).\n');


%% ------------------------------------------------------------------------
function metrics = evaluateClassifier(net, imdsTest, positiveClass, useGray2RGB)
% Binary classification metrics for a trainNetwork model.
% Builds the evaluation datastore at the NET's own input size, so AlexNet
% (227x227) and VGG (224x224) are both handled automatically.

    if nargin < 3 || isempty(positiveClass), positiveClass = ""; end
    if nargin < 4, useGray2RGB = false; end

    inSize = net.Layers(1).InputSize(1:2);
    if useGray2RGB
        aug = augmentedImageDatastore(inSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');
    else
        aug = augmentedImageDatastore(inSize, imdsTest);
    end

    % classify preserves datastore order, so YPred lines up with the labels
    [YPred, scores] = classify(net, aug);
    YTrue = imdsTest.Labels(:);
    YPred = YPred(:);

    classNames = categories(YTrue);            % e.g. {'normal','ptb'}
    if positiveClass == ""
        positiveClass = string(classNames{end});
    end
    positiveClass = string(positiveClass);

    % Confusion matrix: rows = true, cols = predicted, ordered as classNames
    C = confusionmat(YTrue, YPred);

    posIdx = find(string(classNames) == positiveClass, 1);
    if isempty(posIdx)
        error('Positive class "%s" not found in labels.', positiveClass);
    end

    TP = C(posIdx, posIdx);
    FN = sum(C(posIdx, :)) - TP;
    FP = sum(C(:, posIdx)) - TP;
    TN = sum(C(:)) - TP - FN - FP;

    accuracy    = (TP + TN) / sum(C(:));
    precision   = TP / (TP + FP);
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    f1          = 2 * (precision * sensitivity) / (precision + sensitivity);

    % AUC: needs the score column belonging to the positive class
    scoreOrder  = net.Layers(end).Classes;
    posScoreCol = find(string(scoreOrder) == positiveClass, 1);
    [~, ~, ~, AUC] = perfcurve(YTrue, scores(:, posScoreCol), char(positiveClass));

    metrics = struct( ...
        'PositiveClass', positiveClass, ...
        'ClassNames', {classNames}, ...
        'Accuracy', accuracy, 'Precision', precision, ...
        'Sensitivity', sensitivity, 'Specificity', specificity, ...
        'F1', f1, 'AUC', AUC, ...
        'TP', TP, 'TN', TN, 'FP', FP, 'FN', FN, ...
        'ConfusionMatrix', C);
end