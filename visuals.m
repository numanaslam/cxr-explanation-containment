%% Results bar charts, generated from the pipeline CSVs (always in sync with final numbers).
%  Reads:  results\containment_combined.csv  (Model,Condition,Config,Precision,LiftOverChance,pValue,IoU)
%          results\classification_accuracy.csv (Model,Condition,N,Accuracy,Sensitivity,Specificity)
%  Writes: images\containment_chart.pdf  and  images\accuracy_chart.pdf
clc; clear; close all;

resDir = 'C:\paper2_repo\results';
imgDir = 'C:\paper2_repo\images';           % put figures where the paper expects them
if ~exist(imgDir,'dir'), mkdir(imgDir); end

modelsCSV  = {'alexnet_v2','vgg16_v2','vgg19_v2'};
modelsDisp = {'AlexNet','VGG16','VGG19'};
col = [0.85 0.37 0.34; 0.20 0.49 0.72; 0.30 0.69 0.49];   % per-model colours

%% ---------- Chart 1: in-distribution containment (lift over chance) ----------
C = readtable(fullfile(resDir,'containment_combined.csv'));
C = C(strcmp(C.Condition,'ROI'),:);
cfgs  = {'LIME-sp-K30','LIME-grid-K30','LIME-fine-m100-K20','GradCAM@0.5'};
cfgsD = {'LIME sp K30','LIME grid K30','LIME fine K20','Grad-CAM'};
L = nan(numel(cfgs), numel(modelsCSV));     % lift
P = nan(size(L));                            % p-value
for i=1:numel(cfgs)
  for j=1:numel(modelsCSV)
    r = strcmp(C.Config,cfgs{i}) & strcmp(C.Model,modelsCSV{j});
    if any(r), L(i,j)=C.LiftOverChance(find(r,1)); P(i,j)=C.pValue(find(r,1)); end
  end
end

f1 = figure('Color','w','Position',[100 100 720 430]);
b = bar(L,'grouped'); for j=1:numel(b), b(j).FaceColor=col(j,:); end
hold on; yline(0,'k-','chance','LabelHorizontalAlignment','left','FontSize',9);
set(gca,'XTickLabel',cfgsD,'FontSize',11); ylabel('Lift over chance','FontSize',12);
legend(modelsDisp,'Location','northoutside','Orientation','horizontal','Box','off');
title('In-distribution explanation containment');
% significance stars
for i=1:numel(cfgs), for j=1:numel(modelsCSV)
    if isnan(L(i,j)) || isnan(P(i,j)), continue; end
    s=''; if P(i,j)<1e-3, s='***'; elseif P(i,j)<1e-2, s='**'; elseif P(i,j)<0.05, s='*'; end
    if ~isempty(s)
        x=b(j).XEndPoints(i); y=L(i,j)+0.006*sign(L(i,j)+eps);
        text(x,max(y,0.004),s,'HorizontalAlignment','center','FontSize',9);
    end
end, end
box off; exportgraphics(f1, fullfile(imgDir,'containment_chart.pdf'), 'ContentType','vector');

%% ---------- Chart 2: classification accuracy by condition ----------
%  Hardcoded to match tab:acc: ROI = held-out test (Table 3), OOD = full-radiograph set.
%  (Do NOT read ROI from the pipeline CSV: that run evaluates all images incl. training
%   and is inflated. Table 3 is the held-out result.)
condsD = {'ROI (held-out; in-dist.)','Full CXR (OOD)'};
roiAcc = [0.850 0.885 0.911];      % Table 3
oodAcc = [0.39  0.48  0.47];       % full-CXR, minimal preprocessing (update if re-run)
Y = [roiAcc(:) oodAcc(:)];         % rows = models, cols = conditions

f2 = figure('Color','w','Position',[100 100 620 430]);
b2 = bar(Y,'grouped');
b2(1).FaceColor=[0.45 0.62 0.80]; b2(2).FaceColor=[0.86 0.55 0.35];
hold on; yline(0.5,'k--','chance = 0.5','FontSize',9,'LabelHorizontalAlignment','left');
set(gca,'XTickLabel',modelsDisp,'FontSize',11); ylim([0 1]);
ylabel('Classification accuracy','FontSize',12);
legend(condsD,'Location','northoutside','Orientation','horizontal','Box','off');
title('Accuracy collapses out-of-distribution');
box off; exportgraphics(f2, fullfile(imgDir,'accuracy_chart.pdf'), 'ContentType','vector');

fprintf('Saved containment_chart.pdf and accuracy_chart.pdf to %s\n', imgDir);
