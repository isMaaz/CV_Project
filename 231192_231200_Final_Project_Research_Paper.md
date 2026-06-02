# Deep Vision Pipeline for Brain Tumor MRI Slice Classification

Submitted By:

231192 - Muhammad Maaz Akram

231200 - Muhammad Umar

BSAI-6A

Submitted to Ms Hina Rashid

Computer Vision Semester Project

Course: AI303 Computer Vision

## Abstract
This paper presents our Computer Vision semester project on brain tumor MRI slice classification. The work extends three prior assignments into a single workflow that performs acquisition, radiometric pre-processing, segmentation, statistical description, and deep recognition. The final recognition model uses a 3D transformer-style architecture to learn dependencies across neighboring MRI slices. On the held-out test split, the model achieved 78.12% accuracy, 84.78% macro precision, 78.12% macro recall, and 77.03% macro F1-score. The system correctly detected all tumor test slices, with the main remaining error mode being false positives on no-tumor slices.

## Keywords
Computer Vision, Brain Tumor MRI, Pre-processing, Segmentation, GLCM, 3D Transformer, Precision, Recall

## I. Introduction

Medical image interpretation is a demanding Computer Vision problem because image evidence can be subtle, noisy, and dependent on the imaging modality. In magnetic resonance imaging, brain tumor analysis often requires several processing stages before a reliable decision can be made. Raw slices must be normalized, artifacts must be reduced, regions of interest must be isolated, and both local texture and spatial context must be interpreted.

The objective of this project was to convert the outputs of Assignments 1, 2, and 3 into a high-level deep vision application. Assignment 1 focused on radiometric pre-processing and noise mitigation. Assignment 2 focused on segmentation, morphological cleaning, boundary representation, and convex hull analysis. Assignment 3 focused on GLCM texture descriptors, geometric features, and traditional classification. The final project integrates these components into a single end-to-end system.

The assigned dataset category was brain tumor MRI. The final pipeline treats each useful MRI slice as a labeled sample and uses paired segmentation masks when available. The system exports an annotated dataset, trained model, feature vectors, figures, metrics, and reproducible notebooks. This makes the work suitable for both technical evaluation and GitHub-based submission.

## II. Literature Review

Classical Computer Vision pipelines commonly begin with pre-processing and segmentation before recognition. Radiometric correction, denoising, and contrast normalization reduce nuisance variation so that later stages operate on more stable image representations. In MRI, this is important because intensity distributions vary between subjects, sequences, and scanners.

Segmentation is a central task in medical image analysis. Edge-based approaches such as Canny edge detection and intensity-based methods such as Otsu thresholding provide interpretable classical baselines. Morphological operations such as erosion, dilation, opening, and closing are useful for removing isolated artifacts and improving binary masks. Although deep segmentation networks are now common, classical mask generation remains valuable in small-data settings because it is transparent and easy to debug.

Texture analysis is another important tradition in medical imaging. Haralick texture features derived from the Gray-Level Co-occurrence Matrix describe how frequently gray-level pairs occur at specified distances and angles. Energy, entropy, contrast, homogeneity, and correlation are particularly useful for describing local regularity, randomness, and variation. In tumor imaging, these descriptors can capture tissue heterogeneity that may not be obvious from shape alone.

Deep learning changes the recognition stage by learning task-specific representations directly from pixels. Convolutional neural networks are effective for 2D image classification because they build hierarchical features from local receptive fields. Transformers add self-attention, allowing the model to connect spatially separated tokens. In 3D medical imaging, a transformer block can model voxel-to-voxel or slice-to-slice dependencies, which is useful when the appearance of pathology spans multiple neighboring slices.

This project uses the classical stages for interpretability and continuity with earlier assignments, while using a deep 3D transformer-style classifier for the final semantic recognition step. This hybrid design is appropriate for a semester project because it demonstrates both foundational Computer Vision concepts and a modern deep learning component.

## III. Dataset and Acquisition

The dataset information used in this final paper is the same dataset information written in Assignment 1 and Assignment 2 reports. The original dataset selected for the project was BraTS-PEDs from The Cancer Imaging Archive. The original collection page is https://www.cancerimagingarchive.net/collection/brats-peds/ and the DOI is 10.7937/dx5c-tj86. The complete TCIA collection is large, with about 457 subjects and around 32.7 GB of image data.

For the lab work and Colab runs, we did not download the full TCIA collection every time. A smaller uploaded chunk was used through KaggleHub with the dataset name awansaad6797/cancer-dataset. This is the same reason given in the Assignment 1 and Assignment 2 reports: the full BraTS-PEDs dataset was too large for repeated notebook testing, so a smaller practical subset was used for processing.

The data files are NIfTI MRI volumes. Across the earlier assignments, the project used MRI modalities and labels such as T1CE_to_SRI_defaced, T1_to_SRI_defaced, FL_to_SRI_defaced, T2_to_SRI_defaced, and selected BraTS-PED subject files such as BraTS-PED-00195-000-t1n and BraTS-PED-00195-000-t2w. These names are visible in the output CSV files and saved figures from Assignment 1 and Assignment 2.

The final annotated dataset contains 158 exported slice samples. The labels are balanced: 80 no-tumor slices and 78 tumor slices. The train, validation, and test split sizes are 107, 19, and 32, respectively.

The dataset used for final training and testing is stored inside the repository under DeepVision Pipeline Output/annotated_dataset. It includes image slices, binary masks, overlays, and a manifest CSV. This satisfies the final deliverable requirement that the annotated dataset used for training and testing be included with the project submission.

## IV. Assignment 1 Integration: Pre-processing

The pre-processing stage follows the Assignment 1 theme of radiometric correction and noise mitigation. Assignment 1 used the same KaggleHub chunk of the TCIA BraTS-PEDs data. The notebook loaded MRI volumes with NiBabel, selected useful center slices, normalized the image intensity values, and then saved original, filtered, and comparison images.

The Assignment 1 report recorded 4,800 metric rows in content_A1/output/metrics.csv. The metric table contains 13 unique MRI labels or subject names. It includes the main SRI-defaced modalities T1CE, T1, FL, and T2, plus selected BraTS-PED and sub002 files. The filters tested were Gaussian, mean, and median, with 1,600 metric rows for each filter. The saved outputs also include 56 comparison images.

In the final project, the same idea is used before recognition. Each MRI slice is converted to floating point, normalized using percentile clipping, and rescaled to the range 0 to 1. Denoising combines Gaussian filtering, median filtering, and mean filtering. This keeps the final model connected to Assignment 1 instead of treating the deep model as a separate task.

The final deep model receives pre-processed slices, not raw image values. This matters because deep networks are sensitive to input scale and distribution. Consistent pre-processing makes training more stable and improves reproducibility between Colab runs.

## V. Assignment 2 Integration: Segmentation and Masking

The segmentation stage continues the Assignment 2 work. Assignment 2 used the same TCIA BraTS-PEDs source and the same KaggleHub dataset chunk, awansaad6797/cancer-dataset. In that assignment, the notebook focused on T1CE_to_SRI_defaced, T1_to_SRI_defaced, FL_to_SRI_defaced, and T2_to_SRI_defaced MRI slices. The main slices used for the shape analysis were slices 76, 77, and 78.

The Assignment 2 output CSV, content_A2/output_a2/chain_code_report.csv, contains 15 rows. It records chain code length, first difference, shape number preview, and convex hull vertices. The chain lengths ranged from 446 to 476 steps, and the convex hull vertex counts ranged from 45 to 60. The report also saved masks, morphology images, chain code images, and convex hull overlays.

In the final project, when a paired segmentation volume exists, the pipeline uses the annotated mask directly. This is preferable for the final training/testing dataset because the label source is explicit and reproducible. If masks are not available, the notebook still contains the classical fallback from Assignment 2: thresholding, Canny edges, morphological closing and opening, hole filling, and connected component selection.

For each exported sample, the final pipeline saves the pre-processed image, its mask, and an overlay showing the segmented region. The final output folder contains 158 image slices, 158 masks, and 158 overlays. This directly supports the deliverable requiring the final annotated dataset used for training and testing.

## VI. Assignment 3 Integration: Description

The descriptor stage continues the Assignment 3 feature extraction work. For each masked slice, the pipeline computes GLCM texture features: energy, entropy, contrast, homogeneity, and correlation. It also computes geometric descriptors including area, perimeter, centroid, and circularity.

These descriptors are exported in DeepVision Pipeline Output/features/deep_pipeline_features.csv. They are not the direct input to the final 3D transformer model, but they provide interpretable measurements that connect the final project to the statistical texture and geometry work completed earlier.

The descriptor distribution figure shows strong differences between tumor and no-tumor samples in several features. Tumor masks generally have larger area and perimeter, while no-tumor slices often have empty or near-empty masks. One limitation observed in the exported descriptor table is that circularity can become unstable for tiny masks with perimeter close to zero. This does not affect the deep model because the deep model learns from image tensors, but it should be handled carefully when interpreting handcrafted descriptors.

## VII. Deep Recognition Model

The final recognition stage applies a 3D transformer-style architecture. Instead of classifying one 2D slice alone, the model forms a local context volume using neighboring MRI slices. A small Conv3D feature extractor embeds this context, then the feature map is reshaped into voxel-patch tokens. Transformer encoder blocks with multi-head self-attention then learn dependencies among these tokens.

This design satisfies the selected research challenge: 3D Transformer Integration. The attention block allows the model to relate spatial and slice-wise evidence across the local MRI context, which is more appropriate for volumetric medical imaging than treating every slice as completely independent.

The output layer performs binary classification between no_tumor and tumor. A validation-tuned decision threshold was used instead of a fixed 0.50 cutoff. The selected threshold was 0.5563, which was chosen from the validation set to improve the precision-recall tradeoff.

## VIII. Training Protocol

Training was performed in Google Colab. The model used the exported MRI slice context tensors with labels derived from segmentation masks. The dataset was split into training, validation, and held-out test subsets. The test subset was not used for training or threshold selection.

The final run used 107 training samples, 19 validation samples, and 32 test samples. The test set was balanced with 16 no-tumor and 16 tumor samples.

The final epoch recorded training accuracy of 90.65% and validation accuracy of 47.37%. Training loss decreased to 0.2919, while validation loss was 1.7894. This gap indicates overfitting, which is expected because the final run is based on a limited annotated subset.

## IX. Evaluation Methodology

Evaluation uses a complete confusion matrix and precision/recall analysis, as required by the final project instructions. The confusion matrix records true no-tumor, false positive, false negative, and true tumor predictions. Precision measures how many predicted tumor cases are actually tumor. Recall measures how many true tumor cases were detected.

Macro precision, macro recall, and macro F1-score were reported to give equal importance to both classes. This is important because medical applications should not hide poor minority-class behavior behind overall accuracy. The final test set was balanced, so accuracy is also meaningful.

The pipeline exports classification_report.csv, metrics_summary.json, test_predictions.csv, confusion_matrix.png, and precision_recall_curve.png. These files make the evaluation reproducible and easy to inspect.

## X. Results

The final 3D transformer model achieved 78.12% accuracy on the held-out test set. Macro precision was 84.78%, macro recall was 78.12%, and macro F1-score was 77.03%.

The confusion matrix was: true no-tumor predicted no-tumor = 9, true no-tumor predicted tumor = 7, true tumor predicted no-tumor = 0, and true tumor predicted tumor = 16. This means the model detected every tumor test slice, producing zero false negatives.

The main error mode is false positives: seven no-tumor test slices were predicted as tumor. In a medical screening scenario, this is usually less dangerous than false negatives, but it still reduces specificity and would require follow-up review by a clinician or a stronger model.

## XI. Discussion

The results show that integrating classical Computer Vision stages with a deep model can produce a working high-level application. The tumor class achieved perfect recall on the test set, demonstrating that the model learned features associated with tumor-containing slices. The false positives suggest that bright anatomical structures, boundary artifacts, or slice-context ambiguity can still look tumor-like to the model.

The validation curve shows overfitting. Training accuracy increased strongly, while validation accuracy remained lower. This is likely caused by the small number of subjects and the similarity between neighboring slices. Slice-level datasets can appear large, but adjacent slices from the same subject are correlated. A stronger final system should include more subjects and should split data subject-wise to measure generalization more fairly.

Even with these limitations, the project meets the educational objective: it demonstrates acquisition, pre-processing, segmentation, description, and recognition in one runnable pipeline. The exported artifacts also allow another user to inspect the data, reproduce the figures, and rerun the notebook in Colab.

## XII. Limitations

The most important limitation is dataset size. The final run contains 158 slices from a limited annotated subset. A larger project should include more subjects, more MRI modalities, and stronger subject-level train/test separation.

The second limitation is that the task is slice-level tumor presence classification, not full 3D lesion segmentation or clinical diagnosis. The system should be viewed as a Computer Vision prototype, not a medical decision system.

The third limitation is descriptor instability for tiny masks. Circularity can be mathematically unstable when perimeter is near zero. Future work should clamp circularity values, ignore tiny masks for shape analysis, or use a more robust perimeter estimate.

The fourth limitation is model overfitting. Regularization, augmentation, transfer learning, and larger data would likely improve validation performance and reduce false positives.

## XIII. Reproducibility and GitHub Submission

The full project has been initialized as a Git repository and pushed to GitHub. The repository contains the original project PDF, assignment notebooks, assignment reports, final Colab notebook, final outputs, trained model, annotated data, figures, and this research paper.

The final notebook is 231192_231200_Final_Deep_Vision_Pipeline_Colab.ipynb. Running it in Google Colab downloads or loads the dataset, builds the annotated slice dataset, trains the 3D transformer model, evaluates it, exports all artifacts, and downloads the final output zip.

The GitHub repository link for submission is https://github.com/isMaaz/CV_Project.git.

## XIV. Conclusion

This semester project successfully integrates the findings from Assignments 1 through 3 into a complete Deep Vision Pipeline. The system starts with acquisition, applies MRI pre-processing, uses segmentation masks, extracts texture and geometric descriptors, and performs semantic recognition with a deep 3D transformer-style model.

The final held-out test accuracy is 78.12%, with 84.78% macro precision, 78.12% macro recall, and 77.03% macro F1-score. The model achieved complete tumor recall on the test split, which is the strongest result of the final run.

Future improvements should focus on subject-level splitting, larger annotated datasets, stronger augmentation, and a dedicated segmentation network. However, as a CLO-5 high-level deep vision application, the submitted system completes the required pipeline and demonstrates both classical and modern Computer Vision techniques.

## References
[1] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.
[2] Haralick, R. M., Shanmugam, K., and Dinstein, I. (1973). Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics.
[3] Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics.
[4] LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.
[5] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.
[6] Vaswani, A. et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems.
[7] Dosovitskiy, A. et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale.
[8] Menze, B. H. et al. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Transactions on Medical Imaging.
[9] KaggleHub dataset used in this project: awansaad6797/cancer-dataset.
