%% Out-of-distribution qualitative grid: 4 models x {full CXR, LIME, Grad-CAM}
%
%  COMPANION TO make_qualitative_grid.m, which does the same for ROI inputs. Together
%  they give the two-condition qualitative comparison the study was designed around:
%  the LIME mask against the U-Net lung mask on BOTH full radiographs and lung crops.
%
%  WHY THIS REPLACES THE OLD OOD PANELS. Tables tab3/tab5 showed AlexNet-only LIME
%  output on full CXRs and never drew the lung mask, so they could not illustrate
%  containment -- the quantity the paper measures -- at all. This figure draws mask B
%  as an outline on every panel, covers all four architectures, and shows both
%  explainers, matching Figure fig:qualgrid exactly except for the input condition.
%
%  READ IT THROUGH THE RELIABILITY GATE. Classification on these inputs is at or near
%  chance, so these are explanations of unreliable predictions. Each row is therefore
%  annotated with the predicted class and whether it is correct: the point of the
%  figure is that apparent "attention" here accompanies predictions that are wrong as
%  often as not, not that the models remain lung-focused out of distribution.
clc; clear; close all; rng(42);

%% --- config ---
models  = {'alexnet_v2.mat','vgg16_v2.mat','vgg19_v2.mat','resnet50_v2.mat'};
names   = {'AlexNet','VGG16','VGG19','ResNet50'};
% Same patient as the in-distribution grid, so the two figures can be read side by
% side. The trailing _1 marks the true class (1 = abnormal/ptb, 0 = normal).
imgFile  = 'C:\paper2_repo\input\cxr_resized\CHNCXR_0327_1.png';
maskFile = 'C:\paper2_repo\input\mask\CHNCXR_0327_1_mask.png';
outFile  = 'C:\paper2_repo\images\qualitative_grid_ood.png';
numFeatures = 50; K = 30; numSamples = 1000; gradAlpha = 0.45;
outlineColor = [0 1 1];   % cyan lung outline

%  If the named example is absent (naming in cxr_resized can differ), fall back to the
%  first abnormal full CXR that has a matching mask rather than aborting the run.
if ~isfile(imgFile) || ~isfile(maskFile)
    warning('Named example not found; auto-selecting a paired full CXR.');
    [imgFile, maskFile] = pickExample(fileparts(imgFile), fileparts(maskFile), '_mask');
end
fprintf('Example: %s\n', imgFile);

% true class from the filename suffix, matching the convention used elsewhere
[~, bname] = fileparts(imgFile);
tok = split(bname,'_'); trueAbn = strcmp(tok{end},'1');

gt0 = imread(maskFile); if size(gt0,3)==3, gt0 = rgb2gray(gt0); end

have = cellfun(@isfile, models);
if ~all(have)
    warning('Missing: %s -- rows skipped.', strjoin(models(~have), ', '));
    models = models(have); names = names(have);
end
nM = numel(models);

f  = figure('Color','w','Position',[100 100 900 240*nM+40]);
tl = tiledlayout(f, nM, 3, 'TileSpacing','compact', 'Padding','compact');

for mi = 1:nM
    S = load(models{mi});
    if isfield(S,'netTransfer'), net = S.netTransfer; else, fn = fieldnames(S); net = S.(fn{1}); end
    inputSize = net.Layers(1).InputSize(1:2);

    g     = imread(imgFile); if size(g,3)==3, g = rgb2gray(g); end
    g     = imresize(g, inputSize);
    img   = uint8(repmat(g, 1, 1, 3));             % MINIMAL preprocessing -> network
    disp8 = uint8(repmat(imadjust(g), 1, 1, 3));   % contrast-boosted -> display only
    gt    = imresize(gt0, inputSize, 'nearest') > 0;
    YP    = classify(net, img);

    predAbn = ~contains(lower(char(string(YP))), 'normal');
    verdict = 'correct'; if predAbn ~= trueAbn, verdict = 'WRONG'; end

    % --- col 1: full CXR + lung outline ---
    ax = nexttile; imshow(disp8); hold(ax,'on');
    visboundaries(gt, 'Color', outlineColor, 'LineWidth', 0.9);
    ylabel(ax, names{mi}, 'FontWeight','bold', 'FontSize',12, 'Visible','on');
    % predicted class + correctness: these explanations describe THESE predictions
    text(ax, 0.5, 0.04, sprintf('pred: %s (%s)', string(YP), verdict), ...
        'Units','normalized', 'HorizontalAlignment','center', 'FontSize',9, ...
        'Color','w', 'BackgroundColor',[0 0 0], 'Margin',1);
    if mi==1, title(ax, 'Full CXR + lung (B)'); end

    % --- col 2: LIME (top-K superpixel) overlay ---
    [~, fMap, fImp] = imageLIME(net, img, YP, 'Segmentation','superpixels', ...
        'NumFeatures', numFeatures, 'NumSamples', numSamples);
    [~, idx] = maxk(fImp, K); A = ismember(fMap, idx);
    ax = nexttile; imshow(disp8); hold(ax,'on');
    hL = imshow(cat(3, ones(inputSize), zeros(inputSize), zeros(inputSize)));
    set(hL, 'AlphaData', 0.40*A);
    visboundaries(gt, 'Color', outlineColor, 'LineWidth', 0.9);
    % containment of this panel, so the figure carries its own number
    prec = nnz(A & gt)/max(nnz(A),1); lf = nnz(gt)/numel(gt);
    text(ax, 0.5, 0.04, sprintf('lift %+.3f', prec - lf), 'Units','normalized', ...
        'HorizontalAlignment','center','FontSize',9,'Color','w', ...
        'BackgroundColor',[0 0 0],'Margin',1);
    if mi==1, title(ax, sprintf('LIME (top-%d)', K)); end

    % --- col 3: Grad-CAM overlay ---
    m = gradcamMap(net, img, YP, inputSize);
    ax = nexttile; imshow(disp8); hold(ax,'on');
    hm = imagesc(m); set(hm, 'AlphaData', gradAlpha); colormap(ax, jet);
    visboundaries(gt, 'Color', outlineColor, 'LineWidth', 0.9);
    if mi==1, title(ax, 'Grad-CAM'); end
end

title(tl, 'Out-of-distribution (full CXR) explanations vs lung outline (cyan)');
exportgraphics(f, outFile, 'Resolution', 220);
fprintf('Saved OOD qualitative grid -> %s\n', outFile);
fprintf(['NOTE: predictions labelled WRONG are the point -- containment shown here\n' ...
         '      describes unreliable predictions (see tab:acc).\n']);

%% ======================= local functions =======================
function [ip, mp] = pickExample(cxrDir, maskDir, maskSuffix)
    % First abnormal (_1) full CXR with a matching mask; falls back to any pair.
    f = dir(fullfile(cxrDir,'*.png')); f = f(~[f.isdir]);
    [~,o] = sort({f.name}); f = f(o);
    ip = ''; mp = '';
    for pass = [1 0]                       % pass 1: prefer abnormal, pass 0: any
        for i = 1:numel(f)
            b = erase(f(i).name,'.png');
            t = split(b,'_');
            if pass==1 && ~strcmp(t{end},'1'), continue; end
            cand = fullfile(maskDir,[b maskSuffix '.png']);
            if ~isfile(cand)
                alt = fullfile(maskDir,[b '.png']);
                if isfile(alt), cand = alt; else, continue; end
            end
            ip = fullfile(f(i).folder, f(i).name); mp = cand; return;
        end
    end
    error('No image/mask pair found in %s + %s', cxrDir, maskDir);
end

function m = gradcamMap(net, img, YP, inputSize)
    % Auto layer selection works for AlexNet/VGG but can fail on branched DAGs
    % like ResNet50 -> retry with the last ReLU as the feature layer.
    try
        raw = gradCAM(net, img, YP);
    catch
        fl = ''; L = net.Layers;
        for i = numel(L):-1:1
            if isa(L(i),'nnet.cnn.layer.ReLULayer'), fl = L(i).Name; break; end
        end
        if isempty(fl), m = zeros(inputSize); return; end
        raw = gradCAM(net, img, YP, 'FeatureLayer', fl);
    end
    m = mat2gray(imresize(double(raw), inputSize));
end
