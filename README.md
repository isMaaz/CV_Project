# CV Project - Deep Vision Pipeline

Student IDs: 231192, 231200

This repository contains the full AI303 Computer Vision semester project, continuing Assignments 1-3 and completing the final Deep Vision Pipeline.

## Project Flow

1. Assignment 1: Radiometric pre-processing and MRI noise mitigation
2. Assignment 2: Segmentation, morphology, chain code, and convex hull analysis
3. Assignment 3: GLCM texture features, geometric descriptors, and traditional classification
4. Final Project: End-to-end deep vision pipeline for brain tumor MRI slice understanding

## Final Deep Vision Pipeline

The final notebook builds this workflow:

Acquisition -> Pre-processing -> Segmentation -> Description -> Recognition

The final model uses a 3D transformer-style architecture on neighboring MRI slices. The pipeline exports annotated MRI slice data, masks, overlays, handcrafted descriptor CSV files, model artifacts, predictions, evaluation plots, and a report draft.

## Final Run Results

The latest Colab output is stored in:

`DeepVision Pipeline Output/`

Summary:

- Recognition task: tumor presence classification
- Model: 3D transformer
- Samples: 158
- Train/validation/test split: 107 / 19 / 32
- Test accuracy: 78.13%
- Macro precision: 84.78%
- Macro recall: 78.13%
- Macro F1-score: 77.03%
- Confusion matrix: 9 true no-tumor, 7 false positives, 16 true tumor, 0 false negatives

## Important Files

- `CV-Project.pdf`: Original assignment and final project instructions
- `CV_Project_Milestone_1.ipynb`: Assignment 1 notebook
- `CV_ProjectA2_231192_231200.ipynb`: Assignment 2 notebook
- `231192_231200_CV_Project_Assignment3.ipynb`: Assignment 3 notebook
- `231192_231200_Final_Deep_Vision_Pipeline_Colab.ipynb`: Final Colab notebook
- `231192_231200_Final_Project_Research_Paper.docx`: Final research paper for submission
- `231192_231200_Final_Project_Research_Paper.md`: Markdown copy of the final paper
- `DeepVision Pipeline Output/`: Final trained model, dataset exports, metrics, figures, report files, and GitHub-ready files

## Final Output Artifacts

Inside `DeepVision Pipeline Output/`:

- `annotated_dataset/`: image slices, masks, overlays, and manifest
- `features/deep_pipeline_features.csv`: GLCM and geometric feature vectors
- `figures/`: confusion matrix, training curves, precision-recall curve, descriptor distributions, preprocessing previews, and prediction examples
- `models/`: trained Keras model
- `report/`: generated report draft and final research paper
- `classification_report.csv`: precision, recall, F1-score, and support
- `metrics_summary.json`: final summary metrics
- `test_predictions.csv`: held-out test predictions

## Notes

The final run is a slice-level prototype on a limited annotated MRI subset. It achieves full recall for tumor test slices, but also produces some false positives on no-tumor slices. This tradeoff is documented in the generated report draft and should be discussed as a project limitation.
