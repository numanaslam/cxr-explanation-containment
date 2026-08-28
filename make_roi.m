%% make_roi.m
% Phase 1: build ROI images by applying each lung mask to its CXR
% (lung kept, everything outside the mask set to black), then show one
% random cxr / mask / roi triplet to verify alignment.
%
% Flat folders (no class subfolders):
%   input\cxr\*.png    - full chest X-rays
%   input\mask\*.png   - lung masks (matched to cxr by base filename)
% Output:
%   input\roi\*.png    - masked lung-only images

clc; clear; close all;
rng('shuffle');

cxrFolder  = 'input\cxr';
maskFolder = 'input\mask';
roiFolder  = 'input\roi';
if ~exist(roiFolder, 'dir'), mkdir(roiFolder); end

cxr = dir(fullfile(cxrFolder, '*.png'));
fprintf('Found %d CXR images.\n', numel(cxr));

made = 0; skipped = {};
for i = 1:numel(cxr)
    base = erase(cxr(i).name, '.png');

    % find the matching mask by base name (handles e.g. _mask suffix)
    hit = dir(fullfile(maskFolder, [base '*']));
    hit = hit(~[hit.isdir]);
    if isempty(hit), skipped{end+1} = cxr(i).name; continue; end %#ok<SAGROW>

    X = imread(fullfile(cxrFolder, cxr(i).name));
    M = imread(fullfile(hit(1).folder, hit(1).name));
    if size(M,3) == 3, M = rgb2gray(M); end

    % align mask to image size, binarize
    M = imresize(M, [size(X,1) size(X,2)], 'nearest') > 0;

    % apply: keep lung pixels, zero the rest (preserve channels)
    if size(X,3) == 3, Mrep = repmat(M, [1 1 3]); else, Mrep = M; end
    ROI = X .* cast(Mrep, 'like', X);

    imwrite(ROI, fullfile(roiFolder, [base '.png']));
    made = made + 1;
end

fprintf('ROI images written: %d   (skipped, no mask: %d)\n', made, numel(skipped));
if ~isempty(skipped)
    fprintf('  first few skipped: %s\n', strjoin(skipped(1:min(5,end)), ', '));
end

%% --- show one random triplet to verify alignment ------------------------
roi = dir(fullfile(roiFolder, '*.png'));
j = randi(numel(roi));
base = erase(roi(j).name, '.png');
hit  = dir(fullfile(maskFolder, [base '*'])); hit = hit(~[hit.isdir]);

Xc = imread(fullfile(cxrFolder,  [base '.png']));
Mk = imread(fullfile(hit(1).folder, hit(1).name)); if size(Mk,3)==3, Mk=rgb2gray(Mk); end
Rr = imread(fullfile(roiFolder,  roi(j).name));

figure('Name', base);
subplot(1,3,1); imshow(Xc); title('cxr');
subplot(1,3,2); imshow(Mk); title('mask');
subplot(1,3,3); imshow(Rr); title('roi (masked)');
fprintf('Showing random sample: %s\n', base);