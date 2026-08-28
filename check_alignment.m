%% Check A (LIME) vs B (lung mask) alignment BEFORE trusting Algorithm-4 IoU.
%  Algorithm 4 is a pixelwise overlap -> only valid if A and B are the SAME
%  image, same crop, same resolution. This script verifies that, shows an
%  overlay, and reports IoU + precision + recall.
clc; clear; close all;

limeFile = 'C:\paper2_repo\out\CHNCXR_0327_1_lime.png';   % LIME base image (image 1)
maskFile = 'C:\paper2_repo\input\mask\CHNCXR_0327_1_mask.png';  % lung mask (image 2)

% --- sanity 1: same patient? (filenames must share the same base id) ---
[~,ln,~] = fileparts(limeFile); [~,mn,~] = fileparts(maskFile);
lid = erase(ln,{'_lime','_base'}); mid = erase(mn,'_mask');
if ~strcmp(lid, mid)
    warning('PAIRING: "%s" vs "%s" -> DIFFERENT images. IoU is meaningless.', lid, mid);
end

% --- load + binarise ---
A_img = imread(limeFile); if size(A_img,3)==3, A_img = rgb2gray(A_img); end
B_img = imread(maskFile); if size(B_img,3)==3, B_img = rgb2gray(B_img); end
A = A_img > 0;                 % LIME-kept region = non-black pixels
B = B_img > 0;                 % lung mask

% --- sanity 2: same pixel grid? ---
if ~isequal(size(A), size(B))
    warning('SIZE: A=%s vs B=%s -> resizing B to A (nearest). Confirm same crop!', ...
        mat2str(size(A)), mat2str(size(B)));
    B = imresize(B, size(A), 'nearest');
end

% --- sanity 3: are the two regions even in the same place? (centroid gap) ---
sA = regionprops(A,'Centroid'); sB = regionprops(B,'Centroid');
if ~isempty(sA) && ~isempty(sB)
    cA = mean(cat(1,sA.Centroid),1); cB = mean(cat(1,sB.Centroid),1);
    gap = norm(cA - cB) / mean(size(A));      % fraction of image span
    fprintf('Centroid gap between A and B: %.1f%% of image span\n', 100*gap);
    if gap > 0.05
        warning('MISALIGNED: centroids differ by %.1f%% -> A and B are NOT registered.', 100*gap);
    end
end

% --- Algorithm 4 IoU (+ containment/coverage) ---
inter = nnz(A & B); uni = nnz(A | B);
IoU  = inter / uni;
prec = inter / max(nnz(A),1);   % containment
rec  = inter / max(nnz(B),1);   % coverage
fprintf('IoU=%.4f  Precision(containment)=%.4f  Recall(coverage)=%.4f\n', IoU, prec, rec);

% --- overlay so you can SEE the alignment (A=red, B=green, overlap=yellow) ---
figure; sz = size(A);
imshow(zeros([sz 3])); hold on;
r = cat(3, ones(sz), zeros(sz), zeros(sz));
g = cat(3, zeros(sz), ones(sz), zeros(sz));
h1 = imshow(r); set(h1,'AlphaData', 0.5*A);
h2 = imshow(g); set(h2,'AlphaData', 0.5*B);
title(sprintf('A=LIME (red)  B=mask (green)  overlap=yellow | IoU=%.3f', IoU));
hold off;
fprintf(['\nIf red and green do NOT sit on top of each other, A and B are\n' ...
         'not the same frame -> fix the crop/resize/pairing before reporting IoU.\n']);
