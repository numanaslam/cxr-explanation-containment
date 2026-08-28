%% Resize each full CXR to the pixel size of its lung mask -> new folder.
%  Guarantees image and mask share identical dimensions (exact alignment).
%  Originals are NOT modified. Point the main script's FullCXR imgDir at outDir.
clc; clear;

imgDir     = 'C:\paper2_repo\input\cxr';
maskDir    = 'C:\paper2_repo\input\mask';
outDir     = 'C:\paper2_repo\input\cxr_resized';   % NEW folder for resized copies
maskSuffix = '_mask';

if ~exist(outDir, 'dir'), mkdir(outDir); end

files = dir(fullfile(imgDir, '*.png'));
files = files(~[files.isdir]);
nDone = 0; nSkip = 0;

for f = 1:numel(files)
    base = erase(files(f).name, '.png');

    % locate matching mask (try <base>_mask.png, then <base>.png)
    mp = fullfile(maskDir, [base maskSuffix '.png']);
    if ~isfile(mp)
        mp2 = fullfile(maskDir, [base '.png']);
        if isfile(mp2), mp = mp2; else, nSkip = nSkip + 1; continue; end
    end

    img = imread(fullfile(imgDir, files(f).name));
    m   = imread(mp);
    tgt = [size(m,1) size(m,2)];          % target size = mask H x W

    if isequal([size(img,1) size(img,2)], tgt)
        imgR = img;                       % already matches
    else
        imgR = imresize(img, tgt);        % bilinear (continuous-tone image)
    end

    imwrite(imgR, fullfile(outDir, files(f).name));
    nDone = nDone + 1;
end

fprintf('Resized %d images to their mask size -> %s\n', nDone, outDir);
fprintf('Skipped %d images with no matching mask.\n', nSkip);