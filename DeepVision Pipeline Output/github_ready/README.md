# 231192_231200_Final_Deep_Vision_Pipeline

Final Computer Vision project continuing Assignments 1-3.

## Pipeline

1. Acquisition: KaggleHub dataset `awansaad6797/cancer-dataset` or uploaded NIfTI data.
2. Pre-processing: percentile radiometric normalization plus Gaussian, median, and mean denoising.
3. Segmentation: paired tumor masks when present, with a classical fallback mask for unannotated data.
4. Description: GLCM texture features and geometric features.
5. Recognition: `3d_transformer` deep model.
6. Evaluation: confusion matrix, precision/recall, classification report, and held-out predictions.

## Current Run

- Recognition task: tumor_presence
- Samples: 158
- Classes: no_tumor, tumor
- Test accuracy: 0.7812
- Macro precision: 0.8478
- Macro recall: 0.7812
- Macro F1: 0.7703

## Main Artifacts

- `annotated_dataset/`: image slices, masks, overlays, and manifest.
- `features/deep_pipeline_features.csv`: Assignment 3 descriptors reused in the final pipeline.
- `models/`: trained Keras model.
- `figures/`: confusion matrix, training curves, descriptor plots, and prediction examples.
- `classification_report.csv` and `metrics_summary.json`: final evaluation.
