%% k_sweep_v2.m
% Overlap vs K for 10 images, using a retrained _v2 model.
% Reports coverage, precision and Jaccard at each K so the saturation of
% coverage as K approaches the segment count is visible, and the coverage/
% precision tradeoff is explicit.
%
% Lung mask B = the annotated_gray ROI's own nonzero region (that IS the
% dataset lung mask that produced the ROI). Input = ROI (partial) here.

clc; clear; close all;
rng(42);

%% ---- Config -------------------------------------------------------------
roiFolder = 'C:\paper2_repo\input\annotated_gray\annotated_gray';  % ptb\ + normal\
netFile   = 'vgg16_v2.mat';        % change to alexnet_v2 / vgg19_v2 to reuse

Segmentation = 'superpixels';      % or 'grid'
NumFeatures  = 64;                 % segment granularity
NumSamples   = 2048;
LimeModel    = 'tree';
Kvalues      = [5 10 20 30 40 49]; % sweep
NumRuns      = 5;
nImages      = 10;

%% ---- Load model + pick 10 held-out images ------------------------------
S = load(netFile); net = S.netTransfer;
inSize = net.Layers(1).InputSize; H = inSize(1); W = inSize(2);

if isfield(S,'valFiles')
    files = S.valFiles;                          % use held-out split
else
    d = [dir(fullfile(roiFolder,'ptb','*.png')); dir(fullfile(roiFolder,'normal','*.png'))];
    files = fullfile({d.folder}, {d.name})';
    warning('No saved split; sampling from all images.');
end
files = files(randperm(numel(files), min(nImages, numel(files))));
fprintf('Model %s | %d images | %s, NumFeatures=%d\n', netFile, numel(files), Segmentation, NumFeatures);

%% ---- Sweep K -----------------------------------------------------------
rows = zeros(numel(Kvalues), 4);   % [K coverage precision jaccard]
for ki = 1:numel(Kvalues)
    K = Kvalues(ki);
    cov = nan(numel(files),1); prec = nan(numel(files),1); jac = nan(numel(files),1);

    for i = 1:numel(files)
        X = imread(files{i}); if size(X,3)==1, X = repmat(X,[1 1 3]); end
        X = uint8(imresize(X,[H W]));
        B = imresize(rgb2gray_safe(imread(files{i})) > 0, [H W], 'nearest');

        label = classify(net, X);
        interR = zeros(NumRuns,1); Aar = zeros(NumRuns,1);
        for rr = 1:NumRuns
            [~,fmap,fimp] = imageLIME(net, X, label, ...
                'Segmentation',Segmentation,'NumFeatures',NumFeatures, ...
                'NumSamples',NumSamples,'Model',LimeModel);
            kk = min(K, numel(fimp));
            [~,idx] = maxk(fimp, kk);
            A = ismember(fmap, idx);
            interR(rr) = nnz(A & B); Aar(rr) = nnz(A);
        end
        Bar = nnz(B); inter = mean(interR); Am = mean(Aar);
        cov(i)  = inter/max(Bar,1);
        prec(i) = inter/max(Am,1);
        jac(i)  = inter/max(Am+Bar-inter,1);
    end

    rows(ki,:) = [K mean(cov,'omitnan') mean(prec,'omitnan') mean(jac,'omitnan')];
    fprintf('K=%-2d -> coverage=%.4f  precision=%.4f  jaccard=%.4f\n', ...
        K, rows(ki,2), rows(ki,3), rows(ki,4));
end

T = array2table(rows, 'VariableNames', {'K','Coverage','Precision','Jaccard'});
disp(T);
writetable(T, sprintf('k_sweep_%s.csv', erase(netFile,'.mat')));

figure; plot(rows(:,1),rows(:,2),'-o', rows(:,1),rows(:,3),'-s', rows(:,1),rows(:,4),'-^','LineWidth',1.5);
legend('Coverage','Precision','Jaccard','Location','best'); grid on;
xlabel('K (top features)'); ylabel('score'); title(sprintf('%s: overlap vs K', erase(netFile,'.mat')));

function g = rgb2gray_safe(M)
if size(M,3)==3, g = rgb2gray(M); else, g = M; end
end