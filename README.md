# cxr-explanation-containment

Evaluation code for the paper:

> **When Can Explanation Containment Be Trusted? Occlusion Robustness, Centrality Bias,
> and Explainer Disagreement in Chest Radiography CNNs**
> Numan Aslam, Ghulam Mustafa, Adnan N. Qureshi, Ki-Il Kim

Post-hoc explanations are routinely validated in chest radiography by measuring how much
of the explanation falls inside a lung mask ("containment"). This repository contains the
MATLAB code for two checks that determine when that measurement can be interpreted:

1. **Geometry-matched null** — a chance baseline that compares each explanation against
   randomised placements preserving its area, shape, and distance from the image centre,
   separating anatomical specificity from the centrality the conventional lung-fraction
   baseline rewards.
2. **Occlusion-robustness diagnostic** — an admissibility condition: the loss in
   predicted-class probability under occlusion of an explainer's top-ranked regions,
   compared against an area-matched random occlusion drawn from the same segmentation.

Neither check is specific to lungs, to containment, or to chest radiography.

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox (`imageLIME`, `gradCAM`, `trainNetwork`)
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox (`ttest`, `ttest2`, `perfcurve`)
- A CUDA GPU is strongly recommended (LIME with 1000 samples dominates runtime)

## Data

- **Shenzhen (primary):** Kaggle "Pulmonary Chest X-Ray Abnormalities"
  (https://www.kaggle.com/datasets/kmader/chest-xray-abnormalities), which includes the
  manual lung annotations used as the reference region.
- **Montgomery (cross-dataset replication):** NLM Montgomery County set, which ships
  manual left/right lung masks (also mirrored on Kaggle).

Edit the path constants at the top of each script (`C:\paper2_repo\...`) to your layout.

## Primary pipeline (paper order)

| Script | Produces |
|---|---|
| `train_models.m`, `retrain_v2.m`, `train_resnet50.m` | The four fine-tuned classifiers (AlexNet, VGG16, VGG19, ResNet50); identical recipe, shared held-out split |
| `evaluate_models.m` | In-distribution classification performance table |
| `regen_containment.m` | ROI containment: lift over per-image chance, recall, CIs, class-stratified stats |
| `ood_containment_lime.m`, `ood_accuracy_table3.m` | Full-radiograph (out-of-distribution) containment and accuracy |
| `geometry_matched_null.m` | **Geometry-matched null** (rotation + translation controls, 100 draws/image) |
| `deletion_score.m` | **Occlusion-robustness diagnostic** (top-K vs area-matched random vs bottom-K; mean and zero fills; importance concentration) |
| `scale_matched_control.m` | Scale-matched control (rules out image scale as the OOD explanation) |
| `gradcam_threshold_sweep.m` | Grad-CAM threshold stability, tau in [0.3, 0.7] |
| `unet_mask_audit.m`, `mask_class_breakdown.m` | Mask geometry audit; class-dependent chance level |
| `make_results_charts.m`, `make_qualitative_grid.m`, `make_qualitative_grid_ood.m` | Paper figures |

## Cross-dataset replication (Montgomery)

Run in order — the containment step saves every explanation mask, so the null and
deletion steps need no LIME recomputation:

1. `montgomery_prep.m` — combines left/right masks, builds ROI (masking, not cropping)
   and resized full-CXR sets, audits per-class lung fraction
2. `montgomery_containment.m` — accuracy gate + containment, both conditions
3. `montgomery_null.m` — geometry-matched null from saved masks (CPU, minutes)
4. `montgomery_deletion.m` — occlusion-robustness gate from saved masks

## Auxiliary and diagnostic scripts

`check_alignment.m`, `coverage_scores.m`, `iou_sweep.m`, `iou_test.m`, `k_sweep_v2.m`,
`k_sweep_v2_full.m`, `lime_*.m`, `make_figure_thumbs.m`, `make_roi.m`, `phase4.m`,
`resize_images_to_mask.m`, `resizecxr.m`, `run_all_model*.m`, `test.m`, `visuals.m` —
exploration and preprocessing utilities kept for completeness; the tables and figures in
the paper are produced by the primary pipeline above.

## Conventions

- One GPU, serial execution (`ExecutionEnvironment='gpu'`; never `parfor` on one GPU).
- Deterministic image subsets: sorted, class-interleaved enumeration, so every model is
  scored on the same images.
- Result CSVs are merge-keyed on (Model, Condition, Config): re-running one model or one
  condition never overwrites the others.

## Citation

A `CITATION.cff` and an archived DOI (Zenodo) will be added once the paper is accepted.
Until then, please cite the repository URL.

## Python revision pipeline (v2)

`python/` contains a complete PyTorch reimplementation addressing the JMI review:
leakage-free splits, multi-seed retraining, lung-restricted LIME segmentation,
dual-fill occlusion gate, and rotation/translation/mask-swap nulls. See
`python/README.md` for the GPU runbook. The MATLAB scripts remain the record of
the originally published numbers.
