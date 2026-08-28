%% coverage_scores.m
% Explanation COVERAGE of the lung field, reported honestly as coverage
% (NOT Jaccard). For each image:
%     coverage = |A' n B| / |B|
% where A' = union of the top-K LIME segments, B = lung mask.
% Coverage = fraction of the lung the explanation covers. It does NOT
% penalise spill outside the lung (that is precision's job) - so we also
% report precision and Jaccard alongside, for full transparency.
%
% Settings = recovered defaults (superpixels / 64 / 2048 / tree / maxk).

clc; clear; close all;
rng(42);

%% ---- Config -------------------------------------------------------------
cxrFolder  = 'input\cxr';
maskFolder = 'input\mask';
roiFolder  = 'input\roi';

nets = { 'AlexNet','alexnet_net.mat'; 'VGG16','vgg16_net.mat'; 'VGG19','vgg19_net.mat' };

strategies = { 'superpixels', 64; 'grid', 49 };
Kvalues    = [30 49];
imageTypes = {'complete','partial'};

NumSamples = 2048;
NumRuns    = 5;
LimeModel  = 'tree';
maxImages  = Inf;          % Inf = all 566; set small for a trial
outputCSV  = 'coverage_scores.csv';

%% ---- Held-out test set (by base filename against saved split) ----------
imds = imageDatastore(roiFolder, 'FileExtensions', '.png');
roiFiles = imds.Files;
S0 = load(nets{1,2});
if isfield(S0,'valFiles')
    valBases = baseNames(S0.valFiles);
    roiFiles = roiFiles(ismember(baseNames(roiFiles), valBases));
    fprintf('Held-out test images: %d\n', numel(roiFiles));
else
    warning('No saved split; using ALL roi images (training may leak in).');
end
if isfinite(maxImages), roiFiles = roiFiles(1:min(maxImages,numel(roiFiles))); end

%% ---- Sweep --------------------------------------------------------------
rows = {};
for nIdx = 1:size(nets,1)
    S = load(nets{nIdx,2}); net = S.netTransfer;
    for sIdx = 1:size(strategies,1)
        seg = strategies{sIdx,1}; nf = strategies{sIdx,2};
        for K = Kvalues
            for tIdx = 1:numel(imageTypes)
                opts = struct('Segmentation',seg,'NumFeatures',nf,'TopK',K, ...
                    'NumSamples',NumSamples,'NumRuns',NumRuns,'Model',LimeModel, ...
                    'Partial',strcmpi(imageTypes{tIdx},'partial'));
                r = computeCoverage(net, roiFiles, cxrFolder, maskFolder, opts);
                fprintf('%-8s | %-11s | K=%-2d | %-8s -> cov=%.4f  prec=%.4f  jac=%.4f (N=%d)\n', ...
                    nets{nIdx,1}, seg, K, imageTypes{tIdx}, r.Coverage, r.Precision, r.Jaccard, r.N);
                rows(end+1,:) = {nets{nIdx,1}, seg, K, imageTypes{tIdx}, ...
                    r.Coverage, r.CoverageStd, r.Precision, r.Jaccard, r.N}; %#ok<SAGROW>
            end
        end
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'Model','Segmentation','K','ImageType','Coverage','CoverageStd','Precision','Jaccard','N'});
disp(T);
writetable(T, outputCSV);
fprintf('\nSaved to %s\n', outputCSV);


%% ========================================================================
function r = computeCoverage(net, roiFiles, cxrFolder, maskFolder, opts)
    inSize = net.Layers(1).InputSize; H = inSize(1); W = inSize(2);
    n = numel(roiFiles);
    cov = nan(n,1); prec = nan(n,1); jac = nan(n,1);

    for i = 1:n
        [~,base] = fileparts(roiFiles{i});
        if opts.Partial, src = roiFiles{i};
        else,            src = fullfile(cxrFolder,[base '.png']); end
        if exist(src,'file')~=2, continue; end

        X = imread(src); if size(X,3)==1, X = repmat(X,[1 1 3]); end
        X = uint8(imresize(X,[H W]));

        hit = dir(fullfile(maskFolder,[base '*'])); hit = hit(~[hit.isdir]);
        if isempty(hit), continue; end
        M = imread(fullfile(hit(1).folder,hit(1).name));
        if size(M,3)==3, M = rgb2gray(M); end
        B = imresize(M>0,[H W],'nearest');

        label = classify(net, X);
        interR = zeros(opts.NumRuns,1); Aar = zeros(opts.NumRuns,1);
        for rr = 1:opts.NumRuns
            [~,fmap,fimp] = imageLIME(net, X, label, ...
                'Segmentation',opts.Segmentation,'NumFeatures',opts.NumFeatures, ...
                'NumSamples',opts.NumSamples,'Model',opts.Model);
            k = min(opts.TopK, numel(fimp));
            [~,idx] = maxk(fimp,k);
            A = ismember(fmap, idx);
            interR(rr) = nnz(A & B); Aar(rr) = nnz(A);
        end
        Bar   = nnz(B);
        inter = mean(interR); Amean = mean(Aar);
        cov(i)  = inter / max(Bar,1);
        prec(i) = inter / max(Amean,1);
        jac(i)  = inter / max(Amean + Bar - inter, 1);
    end

    r = struct('Coverage', mean(cov,'omitnan'), 'CoverageStd', std(cov,'omitnan'), ...
               'Precision', mean(prec,'omitnan'), 'Jaccard', mean(jac,'omitnan'), ...
               'N', sum(~isnan(cov)));
end

function bn = baseNames(files)
    bn = cell(size(files));
    for i = 1:numel(files), [~,bn{i}] = fileparts(files{i}); end
end