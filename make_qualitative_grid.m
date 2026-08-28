%% Qualitative grid figure: 4 models x {ROI, LIME, Grad-CAM}, in-distribution.
%  Shows the containment contrast visually: VGG maps sit inside the lung outline,
%  AlexNet spills out. Uses an ROI (in-distribution) image -- the regime where the
%  classifier is reliable and explanations are interpretable. Lung mask B is drawn
%  as an outline on every panel.
%
%  PREPROCESSING FIX: explanations are now computed on the MINIMAL-preprocessed
%  image (resize + gray->rgb, ADR-004) so this figure illustrates the SAME pipeline
%  as tab:contain. imadjust is applied only to the DISPLAY copy for visibility --
%  previously it fed the contrast-normalised image to classify/imageLIME/gradCAM,
%  which is the pipeline ADR-004 ruled out.
clc; clear; close all; rng(42);

%% --- config ---
models  = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
names   = {'AlexNet','VGG16','VGG19','ResNet50'};
imgFile  = 'C:\paper2_repo\input\annotated_gray\annotated_gray\ptb\CHNCXR_0327_1.png';
maskFile = 'C:\paper2_repo\input\mask\CHNCXR_0327_1_mask.png';
outFile  = 'C:\paper2_repo\images\qualitative_grid.png';   % where main_revised.tex expects it
numFeatures = 50; K = 30; numSamples = 1000; gradAlpha = 0.45;
outlineColor = [0 1 1];   % cyan lung outline (visible on grey and on jet)

gt0 = imread(maskFile); if size(gt0,3)==3, gt0 = rgb2gray(gt0); end

have = cellfun(@isfile, models);
if ~all(have)
    warning('Missing: %s -- those rows are skipped. (Run train_resnet50.m first?)', ...
        strjoin(models(~have), ', '));
    models = models(have); names = names(have);
end
nM = numel(models);

f  = figure('Color','w','Position',[100 100 900 240*nM+40]);
tl = tiledlayout(f, nM, 3, 'TileSpacing','compact', 'Padding','compact');

for mi = 1:nM
    S = load(models{mi});
    if isfield(S,'netTransfer'), net = S.netTransfer; else, fn = fieldnames(S); net = S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);

    g    = imread(imgFile); if size(g,3)==3, g = rgb2gray(g); end
    g    = imresize(g, inputSize);
    img  = uint8(repmat(g, 1, 1, 3));            % MINIMAL -> fed to the network
    disp8 = uint8(repmat(imadjust(g), 1, 1, 3)); % contrast-boosted -> display only
    gt   = imresize(gt0, inputSize, 'nearest') > 0;
    YP   = classify(net, img);

    % --- col 1: ROI + lung outline ---
    ax = nexttile; imshow(disp8); hold(ax,'on'); visboundaries(gt, 'Color', outlineColor, 'LineWidth', 0.9);
    ylabel(ax, names{mi}, 'FontWeight','bold', 'FontSize',12, 'Visible','on');
    if mi==1, title(ax, 'ROI + lung (B)'); end

    % --- col 2: LIME (top-K superpixel) overlay ---
    [~, fMap, fImp] = imageLIME(net, img, YP, 'Segmentation','superpixels', ...
        'NumFeatures', numFeatures, 'NumSamples', numSamples);
    [~, idx] = maxk(fImp, K); A = ismember(fMap, idx);
    ax = nexttile; imshow(disp8); hold(ax,'on');
    hL = imshow(cat(3, ones(inputSize), zeros(inputSize), zeros(inputSize)));
    set(hL, 'AlphaData', 0.40*A);
    visboundaries(gt, 'Color', outlineColor, 'LineWidth', 0.9);
    if mi==1, title(ax, sprintf('LIME (top-%d)', K)); end

    % --- col 3: Grad-CAM overlay ---
    m = gradcamMap(net, img, YP, inputSize);
    ax = nexttile; imshow(disp8); hold(ax,'on');
    hm = imagesc(m); set(hm, 'AlphaData', gradAlpha); colormap(ax, jet);
    visboundaries(gt, 'Color', outlineColor, 'LineWidth', 0.9);
    if mi==1, title(ax, 'Grad-CAM'); end
end

title(tl, 'In-distribution explanations vs lung outline (cyan)');
exportgraphics(f, outFile, 'Resolution', 220);
fprintf('Saved qualitative grid -> %s\n', outFile);

%% ======================= local functions =======================
function m = gradcamMap(net, img, YP, inputSize)
    % Auto layer selection works for AlexNet/VGG but can fail on branched DAGs
    % like ResNet50 -> retry with the last ReLU as the feature layer.
    try
        raw = gradCAM(net, img, YP);
    catch
        fl = '';
        L = net.Layers;
        for i = numel(L):-1:1
            if isa(L(i),'nnet.cnn.layer.ReLULayer'), fl = L(i).Name; break; end
        end
        if isempty(fl), m = zeros(inputSize); return; end
        raw = gradCAM(net, img, YP, 'FeatureLayer', fl);
    end
    m = mat2gray(imresize(double(raw), inputSize));
end
