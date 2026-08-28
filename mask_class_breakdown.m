%% Does mask fragmentation correlate with class?
%
%  WHY. The audit found 104 of 566 manual lung masks carry small extra components, and
%  the five largest offenders were all abnormal (_1) cases. If fragmentation is
%  concentrated in one class, a reviewer will ask whether the reference region is
%  systematically different between classes -- which would matter for any metric
%  defined against it.
%
%  It does NOT bias our metric: lift is precision minus the PER-IMAGE lung-area
%  fraction, so a class-dependent lung fraction cancels image by image. But the
%  question is cheaper to answer than to rebut, and a plausible mechanism exists
%  (tuberculosis makes lung boundaries harder to delineate, so annotators leave
%  fragments), so it is worth measuring rather than asserting.
%
%  Reads results\unet_mask_audit.csv, written by unet_mask_audit.m.
clc; clear;

csvPath = 'C:\paper2_repo\results\unet_mask_audit.csv';
assert(isfile(csvPath), 'Missing %s -- run unet_mask_audit.m first.', csvPath);
T = readtable(csvPath, 'TextType','string');

%% --- class from the filename suffix (_1 abnormal / _0 normal) ---
cls = strings(height(T),1);
for i = 1:height(T)
    t = split(T.Basename(i), '_');
    switch t(end)
        case "1", cls(i) = "abnormal";
        case "0", cls(i) = "normal";
        otherwise, cls(i) = "unknown";
    end
end
T.Class = cls;
if any(T.Class=="unknown")
    warning('%d basenames did not end in _0/_1 and are excluded.', nnz(T.Class=="unknown"));
end
K = T(T.Class~="unknown", :);

frag = K.NumComponents >= 3;

%% --- contingency + association test ---
fprintf('\n===== mask fragmentation by class (n=%d) =====\n', height(K));
fprintf('%-10s %8s %12s %10s\n', 'class', 'n', 'fragmented', 'rate');
for c = ["normal","abnormal"]
    sel = K.Class==c;
    fprintf('%-10s %8d %12d %9.1f%%\n', c, nnz(sel), nnz(sel & frag), 100*nnz(sel & frag)/max(nnz(sel),1));
end
fprintf('%-10s %8d %12d %9.1f%%\n', 'all', height(K), nnz(frag), 100*nnz(frag)/height(K));

if exist('crosstab','file')==2
    [~, chi2, p] = crosstab(K.Class, frag);
    fprintf('\nchi-square test of independence: chi2=%.3f, p=%.4g\n', chi2, p);
    if p < 0.05
        fprintf('  -> fragmentation IS associated with class. State it in Methods and note\n');
        fprintf('     that lift is per-image chance-corrected, so the metric is unaffected.\n');
    else
        fprintf('  -> no detectable association; the earlier impression from five filenames\n');
        fprintf('     was a sampling artefact of sorting by fragment size.\n');
    end
end

%% --- lung fraction by class (the quantity the chance baseline uses) ---
fprintf('\n----- lung fraction by class -----\n');
for c = ["normal","abnormal"]
    v = K.LungFraction(K.Class==c);
    fprintf('  %-9s mean %.4f  sd %.4f  (n=%d)\n', c, mean(v), std(v), numel(v));
end
vn = K.LungFraction(K.Class=="normal"); va = K.LungFraction(K.Class=="abnormal");
if exist('ttest2','file')==2 && ~isempty(vn) && ~isempty(va)
    [~, pl] = ttest2(vn, va);
    fprintf('  difference: %+.4f (two-sample t-test p=%.4g)\n', mean(va)-mean(vn), pl);
    fprintf(['  NOTE: even a large difference here does not bias lift, which subtracts the\n' ...
             '        per-image lung fraction. It would bias raw precision or raw IoU.\n']);
end

%% --- extra-area magnitude by class, among fragmented masks only ---
if any(frag)
    fprintf('\n----- extra area beyond 2 largest components (fragmented masks only) -----\n');
    for c = ["normal","abnormal"]
        v = K.ExtraAreaFraction(frag & K.Class==c);
        if ~isempty(v)
            fprintf('  %-9s median %.5f  mean %.5f  max %.5f  (n=%d)\n', ...
                c, median(v), mean(v), max(v), numel(v));
        end
    end
end
