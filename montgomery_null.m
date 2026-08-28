%% Geometry-matched null on Montgomery -- THE replication experiment
%
%  THE QUESTION. On Shenzhen, most of the measured containment did not survive a null
%  that preserves each explanation's area, shape and distance from the image centre:
%  the analytic (lung-fraction) baseline was crediting explanations for centrality.
%  If the same happens on Montgomery -- different site, different collimation,
%  different mask geometry -- the centrality bias is a property of the METRIC's
%  conventional baseline, not of one dataset. That single sentence is the strongest
%  answer to both desk rejections, whichever way it comes out:
%     replicates      -> "the standard baseline is inflated wherever the anatomy is
%                        central" (a general methods claim)
%     does not        -> the paper gains an honest boundary condition
%
%  NO GPU, NO LIME. This loads the explanation masks saved by
%  montgomery_containment.m and does pure mask arithmetic: 100 rotation draws and
%  100 translation draws per image/config/model. Minutes on CPU.
%
%  Nulls are IDENTICAL to geometry_matched_null.m:
%    ROTATION (primary): angle from [30,150] u [210,330] deg -- preserves area,
%      shape, distance-from-centre; near-0 and near-180 excluded (bilateral symmetry).
%    TRANSLATION (control on the control): circular shift -- preserves area/shape only.
%      On Shenzhen this recovered the lung fraction (0.245-0.249), confirming the
%      analytic baseline is exactly the uniform-placement null. Check it again here.
clc; clear; close all;
rng(0);

%% --- config ---
allModels  = {'alexnet_v2','vgg16_v2','vgg19_v2','resnet50_v2'};
MODELS     = allModels;
CONDITIONS = {'MG-ROI','MG-FullCXR'};
resDir  = 'C:\paper2_repo\results';
csvOut  = fullfile(resDir,'geometry_matched_null.csv');   % same file as Shenzhen; new Condition keys
nDraws  = 100;
dispKey = {'LIME sp K30','LIME grid K30','LIME fine K20','Grad-CAM'};

%% --- run ---
rows = {};
for ci = 1:numel(CONDITIONS)
    COND = CONDITIONS{ci};
    for mi = 1:numel(MODELS)
        mkey = MODELS{mi};
        matPath = fullfile(resDir, sprintf('mont_masks_%s_%s.mat', mkey, COND));
        if ~isfile(matPath)
            warning('Missing %s -- run montgomery_containment.m first.', matPath); continue;
        end
        D = load(matPath);   % MASKS{cfg,i}, CFGKEYS, AB, INPUTSIZE, mskP
        nCfg = numel(D.CFGKEYS); n = size(D.MASKS,2);
        fprintf('\n== %s (%s): %d images ==\n', mkey, COND, n);

        % ground-truth masks reloaded once, at this model's input size
        GT = cell(1,n); lfA = nan(1,n);
        for i = 1:n
            gt = imread(D.mskP{i}); if size(gt,3)==3, gt = rgb2gray(gt); end
            GT{i} = imresize(gt, D.INPUTSIZE, 'nearest') > 0;
            lfA(i) = nnz(GT{i})/numel(GT{i});
        end

        for c = 1:nCfg
            obs=nan(1,n); nulR=nan(1,n); nulT=nan(1,n); pEmp=nan(1,n);
            for i = 1:n
                A = D.MASKS{c,i};
                if isempty(A) || nnz(A)==0 || nnz(GT{i})==0, continue; end
                obs(i) = nnz(A & GT{i})/nnz(A);
                pr=zeros(nDraws,1); pt=zeros(nDraws,1);
                for d = 1:nDraws
                    Ar = rotateNull(A);  pr(d) = nnz(Ar & GT{i})/max(nnz(Ar),1);
                    At = translateNull(A); pt(d) = nnz(At & GT{i})/max(nnz(At),1);
                end
                nulR(i)=mean(pr); nulT(i)=mean(pt);
                pEmp(i) = (1 + nnz(pr >= obs(i))) / (nDraws + 1);
            end

            keep = ~isnan(obs) & ~isnan(nulR);
            if ~any(keep), continue; end
            o=obs(keep); nr=nulR(keep); nt=nulT(keep);
            la = o - lfA(keep);      % analytic lift (the paper's convention)
            ln = o - nr;             % geometry-matched lift
            lt = o - nt;             % translation-null lift
            abK = logical(D.AB(keep));
            if numel(ln)>1 && exist('ttest','file')==2, [~,pv]=ttest(ln,0,'Tail','right'); else, pv=NaN; end
            fprintf('   %-14s precision %.3f | analytic lift %+.3f | ROTATION null %.3f -> lift %+.3f (p=%.2g)\n', ...
                dispKey{min(c,numel(dispKey))}, mean(o), mean(la), mean(nr), mean(ln), pv);
            fprintf('                    translation null %.3f (lung fraction %.3f -- should match) | lift %+.3f\n', ...
                mean(nt), mean(lfA(keep)), mean(lt));
            if mean(la) > 0.02
                fprintf('                    geometry accounts for %.0f%%%% of the analytic lift\n', ...
                    100*(1 - mean(ln)/mean(la)));
            else
                fprintf('                    analytic lift ~0 (%.3f); ratio not meaningful\n', mean(la));
            end
            mA=NaN; mN=NaN; pCl=NaN;
            if any(abK) && any(~abK)
                mA=mean(ln(abK)); mN=mean(ln(~abK));
                if exist('ttest2','file')==2, [~,pCl]=ttest2(ln(abK), ln(~abK)); end
                fprintf('                    by class (vs null): abnormal %+.3f | normal %+.3f  diff p=%.2g\n', mA, mN, pCl);
            end
            rows(end+1,:) = { string(mkey), string(COND), string(D.CFGKEYS(c)), nnz(keep), ...
                mean(o), mean(lfA(keep)), mean(la), mean(nr), mean(ln), pv, mean(nt), mean(lt), ...
                median(pEmp(keep)), mA, mN, pCl }; %#ok<SAGROW>
        end
    end
end

%% --- save (merges into the Shenzhen null file; MG-* conditions are distinct keys) ---
if isempty(rows), error('No results produced.'); end
T = cell2table(rows, 'VariableNames', {'Model','Condition','Config','N', ...
    'Precision','LungFraction','Lift_analytic','NullPrec_rotation','Lift_vs_rotation', ...
    'pValue_vs_rotation','NullPrec_translation','Lift_vs_translation','MedianEmpiricalP', ...
    'Lift_abnormal','Lift_normal','pValue_classDiff'});
writetable(mergeKeyed(csvOut, T, {'Model','Condition','Config'}), csvOut);
fprintf('\nWrote %s\n', csvOut);
fprintf(['\n==== the replication readout ====\n' ...
        '  Compare Lift_vs_rotation / Lift_analytic here against the Shenzhen ROI rows\n' ...
        '  in the same CSV. Similar shrinkage on both datasets = centrality bias is a\n' ...
        '  property of the conventional baseline, not of one dataset.\n']);

%% ======================= local functions =======================
function A2 = rotateNull(A)
    if rand < 0.5, th = 30 + 120*rand; else, th = 210 + 120*rand; end
    A2 = imrotate(A, th, 'nearest', 'crop');
end

function A2 = translateNull(A)
    A2 = circshift(A, [randi(size(A,1))-1, randi(size(A,2))-1]);
end

function Tall = mergeKeyed(path, Tnew, keyVars)
    Tall = Tnew;
    if ~isfile(path), return; end
    Told = readtable(path, 'TextType','string');
    if ~isequal(sort(Told.Properties.VariableNames), sort(Tnew.Properties.VariableNames))
        warning('%s has an unexpected schema; overwriting.', path); return;
    end
    Told = Told(:, Tnew.Properties.VariableNames);
    kOld = join(string(Told{:,keyVars}), '|');
    kNew = unique(join(string(Tnew{:,keyVars}), '|'));
    Told(ismember(kOld, kNew), :) = [];
    Tall = [Told; Tnew];
end
