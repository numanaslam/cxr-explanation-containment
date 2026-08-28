%% k_sweep_v2_full.m
% Full K-sweep: all three _v2 models, complete + partial, over the entire
% held-out split. Reports coverage / precision / jaccard at each K with
% variance, so the K=5 precision result can be checked at scale rather than
% on 10 images. Writes one tidy CSV.
%
% Lung mask B = the ROI image's own nonzero region (the mask that made it).
% "partial" input = the ROI image; "complete" input = matching cxr image.

clc; clear; close all;
rng(42);

%% ---- Config -------------------------------------------------------------
% If your complete (full) CXRs live elsewhere, set cxrRoot; else leave ''
% and only 'partial' will run.
cxrRoot = '';   % e.g. 'C:\paper2_repo\input\cxr'  (files matched by base name)

nets = { 'AlexNet','alexnet_v2.mat'; 'VGG16','vgg16_v2.mat'; 'VGG19','vgg19_v2.mat' };

Segmentation = 'superpixels';       % or 'grid'
NumFeatures  = 64;
NumSamples   = 2048;
LimeModel    = 'tree';
Kvalues      = [5 10 20 30 40 49];
NumRuns      = 5;
imageTypes   = {'partial','complete'};
maxImages    = Inf;                 % Inf = full held-out set
outputCSV    = 'k_sweep_v2_full.csv';

%% ---- Sweep --------------------------------------------------------------
rows = {};
for nIdx = 1:size(nets,1)
    S = load(nets{nIdx,2}); net = S.netTransfer;
    inSize = net.Layers(1).InputSize; H = inSize(1); W = inSize(2);
    if ~isfield(S,'valFiles'), error('%s has no saved split.', nets{nIdx,2}); end
    roiFiles = S.valFiles;
    if isfinite(maxImages), roiFiles = roiFiles(1:min(maxImages,numel(roiFiles))); end

    for tIdx = 1:numel(imageTypes)
        isPartial = strcmpi(imageTypes{tIdx},'partial');
        if ~isPartial && isempty(cxrRoot), continue; end   % skip complete if no cxr root

        for K = Kvalues
            cov=[]; prec=[]; jac=[];
            for i = 1:numel(roiFiles)
                [~,base] = fileparts(roiFiles{i});
                if isPartial
                    src = roiFiles{i};
                else
                    hitc = dir(fullfile(cxrRoot,'**',[base '.png']));
                    if isempty(hitc), continue; end
                    src = fullfile(hitc(1).folder, hitc(1).name);
                end
                if exist(src,'file')~=2, continue; end

                X = imread(src); if size(X,3)==1, X = repmat(X,[1 1 3]); end
                X = uint8(imresize(X,[H W]));
                Mg = imread(roiFiles{i}); if size(Mg,3)==3, Mg = rgb2gray(Mg); end
                B  = imresize(Mg>0,[H W],'nearest');

                label = classify(net, X);
                iR = zeros(NumRuns,1); aR = zeros(NumRuns,1);
                for rr = 1:NumRuns
                    [~,fmap,fimp] = imageLIME(net, X, label, ...
                        'Segmentation',Segmentation,'NumFeatures',NumFeatures, ...
                        'NumSamples',NumSamples,'Model',LimeModel);
                    kk = min(K, numel(fimp));
                    [~,idx] = maxk(fimp,kk);
                    A = ismember(fmap,idx);
                    iR(rr) = nnz(A & B); aR(rr) = nnz(A);
                end
                Bar = nnz(B); inter = mean(iR); Am = mean(aR);
                cov(end+1)  = inter/max(Bar,1);          %#ok<AGROW>
                prec(end+1) = inter/max(Am,1);           %#ok<AGROW>
                jac(end+1)  = inter/max(Am+Bar-inter,1); %#ok<AGROW>
            end

            fprintf('%-8s | %-8s | K=%-2d -> cov=%.3f±%.3f  prec=%.3f±%.3f  jac=%.3f±%.3f (N=%d)\n', ...
                nets{nIdx,1}, imageTypes{tIdx}, K, mean(cov),std(cov), ...
                mean(prec),std(prec), mean(jac),std(jac), numel(cov));
            rows(end+1,:) = {nets{nIdx,1}, imageTypes{tIdx}, K, ...
                mean(cov),std(cov), mean(prec),std(prec), mean(jac),std(jac), numel(cov)}; %#ok<SAGROW>
        end
    end
end

T = cell2table(rows,'VariableNames', ...
    {'Model','ImageType','K','Coverage','CovStd','Precision','PrecStd','Jaccard','JacStd','N'});
disp(T);
writetable(T, outputCSV);
fprintf('\nSaved -> %s\n', outputCSV);