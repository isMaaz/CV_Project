# Final Semester Project Report Draft

## Title
Deep Vision Pipeline for Brain Tumor MRI Slice Understanding

## Abstract
This project extends the first three Computer Vision assignments into an end-to-end deep vision workflow. The system performs data acquisition, radiometric pre-processing, segmentation, feature description, and final recognition using a deep model. The current run used `3d_transformer` for the recognition stage and produced a held-out test accuracy of 0.7812, macro precision of 0.8478, macro recall of 0.7812, and macro F1 of 0.7703.

## Literature Review Notes
Medical image analysis pipelines commonly combine radiometric correction, denoising, segmentation, handcrafted descriptors, and deep learning. GLCM features capture local texture statistics, while geometric descriptors capture object shape and compactness. CNNs learn hierarchical image representations directly from pixels. For the bonus research challenge, the 3D transformer option uses neighboring MRI slices so attention can model voxel-to-voxel dependencies across slice context.

## Methodology
The dataset was acquired from KaggleHub using `awansaad6797/cancer-dataset`. Each NIfTI volume was normalized with percentile clipping, denoised with Gaussian, median, and mean filtering, and converted to 2D slice samples with a local 3D context window. If a paired tumor segmentation volume was available, it was used as the annotation source. Otherwise, a classical segmentation fallback based on thresholding, Canny edges, morphology, hole filling, and connected components was used for descriptor extraction.

## Feature Description
For every sample, the pipeline exported GLCM energy, entropy, contrast, homogeneity, and correlation. It also exported area, perimeter, centroid, and circularity for the segmented region.

## Deep Recognition Model
The selected model was `3d_transformer`. The default 3D transformer first embeds neighboring MRI slices with Conv3D layers, flattens the result into voxel-patch tokens, and applies multi-head self-attention. This connects the final project to the 3D transformer research challenge.

## Results
- Samples: 158
- Recognition task: tumor_presence
- Classes: no_tumor, tumor
- Test accuracy: 0.7812
- Macro precision: 0.8478
- Macro recall: 0.7812
- Macro F1: 0.7703

The confusion matrix, descriptor distributions, training curves, and prediction examples are saved in the `figures/` directory.

## Conclusion
The final pipeline integrates the pre-processing, structural segmentation, texture/geometric description, and classification components developed during Assignments 1-3. The exported annotated dataset and metrics can be used for the final 15-page paper and GitHub submission.
