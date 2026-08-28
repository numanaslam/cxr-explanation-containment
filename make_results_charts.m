%% Results bar charts, generated from the pipeline CSVs (always in sync with final numbers).
%  Reads:  results\containment_combined.csv  (Model,Condition,Config,Precision,LiftOverChance,pValue,IoU)
%          results\classification_accuracy.csv (Model,Condition,N,Accuracy,Sensitivity,Specificity)
%  Writes: images\containment_chart.pdf  and  images\accuracy_chart.pdf
clc; clear; close all;

resDir = 'C:\paper2_repo\results';
imgDir = 'C:\paper2_repo\images';           % put figures where the paper expects them
if ~exist(imgDir,'dir'), mkdir(imgDir); end

modelsCSV  = {'alexnet_v2','vgg16_v2','vgg19_v2','resnet50_v2'};
modelsDisp = {'AlexNet','VGG16','VGG19','ResNet50'};
col = [0.85 0.37 0.34; 0.20 0.49 0.72; 0.30 0.69 0.49; 0.58 0.40 0.74];   % per-model colours

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
    % MUST match regen_containment.m's LaTeX convention, else the figure and
    % tab:contain disagree on the same numbers (chart said *** where the table
    % says *). Convention: *** p<1e-13, * p<1e-3, blank otherwise.
    s=''; if P(i,j)<1e-13, s='***'; elseif P(i,j)<1e-3, s='*'; end
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
%  ResNet50 entries: paste the held-out accuracy printed by train_resnet50.m and the
%  OOD accuracy printed by ood_accuracy_table3.m. NaN leaves a visible gap in the
%  chart rather than silently plotting a wrong bar.
%  ROI: evaluate_models.m on the _v2 models, shared 113-image held-out split
%  (classification_metrics_v2.csv). This is the training-consistent pipeline --
%  do NOT substitute ood_accuracy_table3.m's ROI column, which resizes differently
%  and disagrees by 0-3 images. Superseded values were 0.850/0.885/0.911 (_net models).
roiAcc = [0.805 0.885 0.867 0.814];
oodAcc = [0.394 0.485 0.467 0.542];   % full-CXR n=662 (ood_accuracy.csv)
if any(isnan([roiAcc oodAcc]))
    warning('Accuracy chart has NaN entries -- fill them before using the figure.');
end
Y = [roiAcc(:) oodAcc(:)];         % rows = models, cols = conditions

f2 = figure('Color','w','Position',[100 100 620 430]);
b2 = bar(Y,'grouped');
b2(1).FaceColor=[0.45 0.62 0.80]; b2(2).FaceColor=[0.86 0.55 0.35];
hold on; yline(0.5,'k--','chance = 0.5','FontSize',9,'LabelHorizontalAlignment','left');
set(gca,'XTickLabel',modelsDisp,'FontSize',11); ylim([0 1]);
ylabel('Classification accuracy','FontSize',12);
legend(condsD,'Location','northoutside','Orientation','horizontal','Box','off');
% NOT "collapses below chance": ResNet50 sits at 0.542, visibly ABOVE the chance
% line, so that title would be contradicted by its own figure.
title('Accuracy falls to near chance on full radiographs');
box off; exportgraphics(f2, fullfile(imgDir,'accuracy_chart.pdf'), 'ContentType','vector');

fprintf('Saved containment_chart.pdf and accuracy_chart.pdf to %s\n', imgDir);
