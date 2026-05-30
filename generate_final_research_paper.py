import csv
import html
import json
import math
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "DeepVision Pipeline Output"
REPORT_DIR = OUTPUT_DIR / "report"
FIG_DIR = OUTPUT_DIR / "figures"

ROOT_DOCX = BASE_DIR / "231192_231200_Final_Project_Research_Paper.docx"
ROOT_MD = BASE_DIR / "231192_231200_Final_Project_Research_Paper.md"
REPORT_DOCX = REPORT_DIR / "231192_231200_Final_Project_Research_Paper.docx"
REPORT_MD = REPORT_DIR / "231192_231200_Final_Project_Research_Paper.md"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


metrics = read_json(OUTPUT_DIR / "metrics_summary.json")
manifest = read_csv(OUTPUT_DIR / "annotated_dataset" / "annotated_dataset_manifest.csv")
predictions = read_csv(OUTPUT_DIR / "test_predictions.csv")
history = read_csv(OUTPUT_DIR / "training_history.csv")


def counts(rows, key):
    result = {}
    for row in rows:
        result[row.get(key, "")] = result.get(row.get(key, ""), 0) + 1
    return result


def split_counts():
    result = {}
    for row in manifest:
        split = row["split"]
        label = row["label"]
        result.setdefault(split, {})
        result[split][label] = result[split].get(label, 0) + 1
    return result


def prediction_counts():
    result = {}
    for row in predictions:
        true_label = row["true_label"]
        pred_label = row["predicted_label"]
        result.setdefault(true_label, {})
        result[true_label][pred_label] = result[true_label].get(pred_label, 0) + 1
    return result


label_counts = counts(manifest, "label")
split_by_label = split_counts()
pred_by_true = prediction_counts()
class_names = metrics["class_names"]
cm = metrics["confusion_matrix"]

latest_epoch = history[-1] if history else {}
train_acc = float(latest_epoch.get("accuracy", 0) or 0)
val_acc = float(latest_epoch.get("val_accuracy", 0) or 0)
train_loss = float(latest_epoch.get("loss", 0) or 0)
val_loss = float(latest_epoch.get("val_loss", 0) or 0)

figures = [
    ("Figure 1. Pre-processing and segmentation preview", FIG_DIR / "preprocessing_segmentation_preview.png"),
    ("Figure 2. Texture and geometric descriptor distributions", FIG_DIR / "descriptor_distributions.png"),
    ("Figure 3. Training and validation curves", FIG_DIR / "training_curves.png"),
    ("Figure 4. Confusion matrix on held-out test data", FIG_DIR / "confusion_matrix.png"),
    ("Figure 5. Precision-recall curve", FIG_DIR / "precision_recall_curve.png"),
    ("Figure 6. Sample held-out predictions", FIG_DIR / "sample_predictions.png"),
]


def pct(value):
    return f"{100 * float(value):.2f}%"


def make_markdown():
    lines = []
    lines.append("# Deep Vision Pipeline for Brain Tumor MRI Slice Understanding")
    lines.append("")
    lines.append("Student IDs: 231192, 231200")
    lines.append("")
    lines.append("Course: AI303 Computer Vision")
    lines.append("")
    lines.append("## Abstract")
    lines.append(
        "This research paper presents a complete Computer Vision pipeline for brain tumor MRI slice understanding. "
        "The work extends three prior assignments into a single workflow that performs acquisition, radiometric "
        "pre-processing, segmentation, statistical description, and deep recognition. The final recognition model "
        "uses a 3D transformer-style architecture to learn dependencies across neighboring MRI slices. On the "
        f"held-out test split, the model achieved {pct(metrics['test_accuracy'])} accuracy, "
        f"{pct(metrics['macro_precision'])} macro precision, {pct(metrics['macro_recall'])} macro recall, and "
        f"{pct(metrics['macro_f1'])} macro F1-score. The system correctly detected all tumor test slices, with "
        "the main remaining error mode being false positives on no-tumor slices."
    )
    sections = paper_sections()
    for title, paragraphs in sections:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        for paragraph in paragraphs:
            if isinstance(paragraph, list):
                for item in paragraph:
                    lines.append(f"- {item}")
                lines.append("")
            else:
                lines.append(paragraph)
                lines.append("")
    lines.append("## References")
    lines.append("")
    for ref in references():
        lines.append(f"- {ref}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).strip() + "\n"
    ROOT_MD.write_text(text, encoding="utf-8")
    REPORT_MD.write_text(text, encoding="utf-8")


def paper_sections():
    return [
        (
            "1. Introduction",
            [
                "Medical image interpretation is a demanding Computer Vision problem because image evidence can be subtle, noisy, and dependent on the imaging modality. In magnetic resonance imaging, brain tumor analysis often requires several processing stages before a reliable decision can be made. Raw slices must be normalized, artifacts must be reduced, regions of interest must be isolated, and both local texture and spatial context must be interpreted.",
                "The objective of this project was to convert the outputs of Assignments 1, 2, and 3 into a high-level deep vision application. Assignment 1 focused on radiometric pre-processing and noise mitigation. Assignment 2 focused on segmentation, morphological cleaning, boundary representation, and convex hull analysis. Assignment 3 focused on GLCM texture descriptors, geometric features, and traditional classification. The final project integrates these components into a single end-to-end system.",
                "The assigned dataset category was brain tumor MRI. The final pipeline treats each useful MRI slice as a labeled sample and uses paired segmentation masks when available. The system exports an annotated dataset, trained model, feature vectors, figures, metrics, and reproducible notebooks. This makes the work suitable for both technical evaluation and GitHub-based submission.",
            ],
        ),
        (
            "2. Literature Review",
            [
                "Classical Computer Vision pipelines commonly begin with pre-processing and segmentation before recognition. Radiometric correction, denoising, and contrast normalization reduce nuisance variation so that later stages operate on more stable image representations. In MRI, this is important because intensity distributions vary between subjects, sequences, and scanners.",
                "Segmentation is a central task in medical image analysis. Edge-based approaches such as Canny edge detection and intensity-based methods such as Otsu thresholding provide interpretable classical baselines. Morphological operations such as erosion, dilation, opening, and closing are useful for removing isolated artifacts and improving binary masks. Although deep segmentation networks are now common, classical mask generation remains valuable in small-data settings because it is transparent and easy to debug.",
                "Texture analysis is another important tradition in medical imaging. Haralick texture features derived from the Gray-Level Co-occurrence Matrix describe how frequently gray-level pairs occur at specified distances and angles. Energy, entropy, contrast, homogeneity, and correlation are particularly useful for describing local regularity, randomness, and variation. In tumor imaging, these descriptors can capture tissue heterogeneity that may not be obvious from shape alone.",
                "Deep learning changes the recognition stage by learning task-specific representations directly from pixels. Convolutional neural networks are effective for 2D image classification because they build hierarchical features from local receptive fields. Transformers add self-attention, allowing the model to connect spatially separated tokens. In 3D medical imaging, a transformer block can model voxel-to-voxel or slice-to-slice dependencies, which is useful when the appearance of pathology spans multiple neighboring slices.",
                "This project uses the classical stages for interpretability and continuity with earlier assignments, while using a deep 3D transformer-style classifier for the final semantic recognition step. This hybrid design is appropriate for a semester project because it demonstrates both foundational Computer Vision concepts and a modern deep learning component.",
            ],
        ),
        (
            "3. Dataset and Acquisition",
            [
                "The dataset was acquired through KaggleHub using the brain tumor MRI dataset referenced in the project notebook. The final run selected paired NIfTI MRI volumes and segmentation files. The exported dataset contains PNG slice images, binary masks, overlays, and a manifest CSV describing each sample.",
                f"The final annotated dataset contains {metrics['num_samples']} samples. The labels are balanced: {label_counts.get('no_tumor', 0)} no-tumor slices and {label_counts.get('tumor', 0)} tumor slices. The train, validation, and test split sizes are {sum(split_by_label.get('train', {}).values())}, {sum(split_by_label.get('validation', {}).values())}, and {sum(split_by_label.get('test', {}).values())}, respectively.",
                "The dataset used for final testing is stored inside the repository under DeepVision Pipeline Output/annotated_dataset. This satisfies the final deliverable requirement that the annotated dataset used for training and testing be included with the project submission.",
            ],
        ),
        (
            "4. Assignment 1 Integration: Pre-processing",
            [
                "The pre-processing stage follows the Assignment 1 theme of radiometric correction and noise mitigation. Each MRI slice is converted to floating point, normalized using percentile clipping, and rescaled to the range 0 to 1. This reduces the effect of extreme intensity values and creates a stable input range for later operations.",
                "Denoising combines Gaussian filtering, median filtering, and mean filtering. Gaussian smoothing reduces high-frequency acquisition noise. Median filtering suppresses isolated salt-and-pepper artifacts while preserving stronger edges. Mean filtering provides a final local averaging step. These operations connect directly to the original Assignment 1 deliverables, where Gaussian, median, and mean filters were evaluated with visual outputs and image quality metrics.",
                "The final deep model receives pre-processed slices, not raw image values. This matters because deep networks are sensitive to input scale and distribution. Consistent pre-processing makes training more stable and improves reproducibility between Colab runs.",
            ],
        ),
        (
            "5. Assignment 2 Integration: Segmentation and Masking",
            [
                "The segmentation stage continues the Assignment 2 work. When a paired segmentation volume exists, the pipeline uses the annotated mask directly. This is preferable for the final training/testing dataset because the label source is explicit and reproducible. If masks are not available, the notebook contains a fallback classical segmentation method based on thresholding, Canny edges, morphological closing/opening, hole filling, and connected-component selection.",
                "For each exported sample, the pipeline saves the original pre-processed image, its mask, and an overlay showing the segmented region. The overlays make it possible to visually audit whether the mask is aligned with the anatomical region of interest. This is important because poor masks can damage both handcrafted feature extraction and model interpretation.",
                "The final output folder contains 158 image slices, 158 masks, and 158 overlays. This directly supports the deliverable requiring the final annotated dataset used for training and testing.",
            ],
        ),
        (
            "6. Assignment 3 Integration: Description",
            [
                "The descriptor stage continues the Assignment 3 feature extraction work. For each masked slice, the pipeline computes GLCM texture features: energy, entropy, contrast, homogeneity, and correlation. It also computes geometric descriptors including area, perimeter, centroid, and circularity.",
                "These descriptors are exported in DeepVision Pipeline Output/features/deep_pipeline_features.csv. They are not the direct input to the final 3D transformer model, but they provide interpretable measurements that connect the final project to the statistical texture and geometry work completed earlier.",
                "The descriptor distribution figure shows strong differences between tumor and no-tumor samples in several features. Tumor masks generally have larger area and perimeter, while no-tumor slices often have empty or near-empty masks. One limitation observed in the exported descriptor table is that circularity can become unstable for tiny masks with perimeter close to zero. This does not affect the deep model because the deep model learns from image tensors, but it should be handled carefully when interpreting handcrafted descriptors.",
            ],
        ),
        (
            "7. Deep Recognition Model",
            [
                "The final recognition stage applies a 3D transformer-style architecture. Instead of classifying one 2D slice alone, the model forms a local context volume using neighboring MRI slices. A small Conv3D feature extractor embeds this context, then the feature map is reshaped into voxel-patch tokens. Transformer encoder blocks with multi-head self-attention then learn dependencies among these tokens.",
                "This design satisfies the selected research challenge: 3D Transformer Integration. The attention block allows the model to relate spatial and slice-wise evidence across the local MRI context, which is more appropriate for volumetric medical imaging than treating every slice as completely independent.",
                "The output layer performs binary classification between no_tumor and tumor. A validation-tuned decision threshold was used instead of a fixed 0.50 cutoff. The selected threshold was " + f"{metrics.get('decision_threshold', 0):.4f}" + ", which was chosen from the validation set to improve the precision-recall tradeoff.",
            ],
        ),
        (
            "8. Training Protocol",
            [
                "Training was performed in Google Colab. The model used the exported MRI slice context tensors with labels derived from segmentation masks. The dataset was split into training, validation, and held-out test subsets. The test subset was not used for training or threshold selection.",
                f"The final run used {sum(split_by_label.get('train', {}).values())} training samples, {sum(split_by_label.get('validation', {}).values())} validation samples, and {sum(split_by_label.get('test', {}).values())} test samples. The test set was balanced with {split_by_label.get('test', {}).get('no_tumor', 0)} no-tumor and {split_by_label.get('test', {}).get('tumor', 0)} tumor samples.",
                f"The final epoch recorded training accuracy of {pct(train_acc)} and validation accuracy of {pct(val_acc)}. Training loss decreased to {train_loss:.4f}, while validation loss was {val_loss:.4f}. This gap indicates overfitting, which is expected because the final run is based on a limited annotated subset.",
            ],
        ),
        (
            "9. Evaluation Methodology",
            [
                "Evaluation uses a complete confusion matrix and precision/recall analysis, as required by the final project instructions. The confusion matrix records true no-tumor, false positive, false negative, and true tumor predictions. Precision measures how many predicted tumor cases are actually tumor. Recall measures how many true tumor cases were detected.",
                "Macro precision, macro recall, and macro F1-score were reported to give equal importance to both classes. This is important because medical applications should not hide poor minority-class behavior behind overall accuracy. The final test set was balanced, so accuracy is also meaningful.",
                "The pipeline exports classification_report.csv, metrics_summary.json, test_predictions.csv, confusion_matrix.png, and precision_recall_curve.png. These files make the evaluation reproducible and easy to inspect.",
            ],
        ),
        (
            "10. Results",
            [
                f"The final 3D transformer model achieved {pct(metrics['test_accuracy'])} accuracy on the held-out test set. Macro precision was {pct(metrics['macro_precision'])}, macro recall was {pct(metrics['macro_recall'])}, and macro F1-score was {pct(metrics['macro_f1'])}.",
                f"The confusion matrix was: true no-tumor predicted no-tumor = {cm[0][0]}, true no-tumor predicted tumor = {cm[0][1]}, true tumor predicted no-tumor = {cm[1][0]}, and true tumor predicted tumor = {cm[1][1]}. This means the model detected every tumor test slice, producing zero false negatives.",
                "The main error mode is false positives: seven no-tumor test slices were predicted as tumor. In a medical screening scenario, this is usually less dangerous than false negatives, but it still reduces specificity and would require follow-up review by a clinician or a stronger model.",
            ],
        ),
        (
            "11. Discussion",
            [
                "The results show that integrating classical Computer Vision stages with a deep model can produce a working high-level application. The tumor class achieved perfect recall on the test set, demonstrating that the model learned features associated with tumor-containing slices. The false positives suggest that bright anatomical structures, boundary artifacts, or slice-context ambiguity can still look tumor-like to the model.",
                "The validation curve shows overfitting. Training accuracy increased strongly, while validation accuracy remained lower. This is likely caused by the small number of subjects and the similarity between neighboring slices. Slice-level datasets can appear large, but adjacent slices from the same subject are correlated. A stronger final system should include more subjects and should split data subject-wise to measure generalization more fairly.",
                "Even with these limitations, the project meets the educational objective: it demonstrates acquisition, pre-processing, segmentation, description, and recognition in one runnable pipeline. The exported artifacts also allow another user to inspect the data, reproduce the figures, and rerun the notebook in Colab.",
            ],
        ),
        (
            "12. Limitations",
            [
                "The most important limitation is dataset size. The final run contains 158 slices from a limited annotated subset. A larger project should include more subjects, more MRI modalities, and stronger subject-level train/test separation.",
                "The second limitation is that the task is slice-level tumor presence classification, not full 3D lesion segmentation or clinical diagnosis. The system should be viewed as a Computer Vision prototype, not a medical decision system.",
                "The third limitation is descriptor instability for tiny masks. Circularity can be mathematically unstable when perimeter is near zero. Future work should clamp circularity values, ignore tiny masks for shape analysis, or use a more robust perimeter estimate.",
                "The fourth limitation is model overfitting. Regularization, augmentation, transfer learning, and larger data would likely improve validation performance and reduce false positives.",
            ],
        ),
        (
            "13. Reproducibility and GitHub Submission",
            [
                "The full project has been initialized as a Git repository and pushed to GitHub. The repository contains the original project PDF, assignment notebooks, assignment reports, final Colab notebook, final outputs, trained model, annotated data, figures, and this research paper.",
                "The final notebook is 231192_231200_Final_Deep_Vision_Pipeline_Colab.ipynb. Running it in Google Colab downloads or loads the dataset, builds the annotated slice dataset, trains the 3D transformer model, evaluates it, exports all artifacts, and downloads the final output zip.",
                "The GitHub repository link for submission is https://github.com/isMaaz/CV_Project.git.",
            ],
        ),
        (
            "14. Conclusion",
            [
                "This semester project successfully integrates the findings from Assignments 1 through 3 into a complete Deep Vision Pipeline. The system starts with acquisition, applies MRI pre-processing, uses segmentation masks, extracts texture and geometric descriptors, and performs semantic recognition with a deep 3D transformer-style model.",
                f"The final held-out test accuracy is {pct(metrics['test_accuracy'])}, with {pct(metrics['macro_precision'])} macro precision, {pct(metrics['macro_recall'])} macro recall, and {pct(metrics['macro_f1'])} macro F1-score. The model achieved complete tumor recall on the test split, which is the strongest result of the final run.",
                "Future improvements should focus on subject-level splitting, larger annotated datasets, stronger augmentation, and a dedicated segmentation network. However, as a CLO-5 high-level deep vision application, the submitted system completes the required pipeline and demonstrates both classical and modern Computer Vision techniques.",
            ],
        ),
    ]


def references():
    return [
        "Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.",
        "Haralick, R. M., Shanmugam, K., and Dinstein, I. (1973). Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics.",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics.",
        "LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE.",
        "Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.",
        "Vaswani, A. et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems.",
        "Dosovitskiy, A. et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale.",
        "Menze, B. H. et al. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Transactions on Medical Imaging.",
        "KaggleHub dataset used in this project: awansaad6797/cancer-dataset.",
    ]


def esc(text):
    return html.escape(str(text), quote=False)


def png_size(path):
    with open(path, "rb") as f:
        sig = f.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        return 800, 500
    width, height = struct.unpack(">II", sig[16:24])
    return width, height


def paragraph(text="", style=None, bold=False, italic=False):
    text = esc(text)
    ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
    rpr = ""
    if bold or italic:
        rpr = "<w:rPr>" + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + "</w:rPr>"
    return f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"


def page_break():
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def table(rows):
    out = [
        "<w:tbl>",
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders><w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/></w:tblBorders></w:tblPr>",
    ]
    for row in rows:
        out.append("<w:tr>")
        for cell in row:
            out.append("<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>")
            out.append(paragraph(cell))
            out.append("</w:tc>")
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def image_xml(rid, title, width_px, height_px):
    max_width_emu = 5_900_000
    emu_per_px = 9525
    width_emu = width_px * emu_per_px
    height_emu = height_px * emu_per_px
    if width_emu > max_width_emu:
        scale = max_width_emu / width_emu
        width_emu = int(width_emu * scale)
        height_emu = int(height_emu * scale)
    doc_pr_id = int(rid.replace("rId", "")) + 100
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{int(width_emu)}" cy="{int(height_emu)}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{doc_pr_id}" name="{esc(title)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{esc(title)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rid}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{int(width_emu)}" cy="{int(height_emu)}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def build_docx():
    image_rels = []
    image_files = []
    rid_counter = 10
    for title, path in figures:
        if path.exists():
            rid = f"rId{rid_counter}"
            image_rels.append((rid, path.name))
            image_files.append((title, path, rid))
            rid_counter += 1

    body = []
    body.append(paragraph("Deep Vision Pipeline for Brain Tumor MRI Slice Understanding", "Title"))
    body.append(paragraph("Final Semester Project Research Paper", "Subtitle"))
    body.append(paragraph("Student IDs: 231192, 231200"))
    body.append(paragraph("Course: AI303 Computer Vision"))
    body.append(paragraph("GitHub Repository: https://github.com/isMaaz/CV_Project.git"))
    body.append(paragraph("Abstract", "Heading1"))
    body.append(paragraph(
        "This paper presents an end-to-end deep vision pipeline for brain tumor MRI slice understanding. "
        "The project integrates radiometric pre-processing, segmentation, texture and geometric description, "
        "and deep recognition. The final model uses a 3D transformer-style architecture to learn local "
        "voxel-to-voxel and slice-to-slice dependencies. The held-out test accuracy is "
        f"{pct(metrics['test_accuracy'])}, with macro precision {pct(metrics['macro_precision'])}, "
        f"macro recall {pct(metrics['macro_recall'])}, and macro F1-score {pct(metrics['macro_f1'])}."
    ))
    body.append(paragraph("Keywords: Computer Vision, MRI, Brain Tumor, GLCM, 3D Transformer, Deep Learning"))
    body.append(page_break())

    body.append(paragraph("Table of Contents", "Heading1"))
    toc_items = [
        "1. Introduction",
        "2. Literature Review",
        "3. Dataset and Acquisition",
        "4. Pre-processing",
        "5. Segmentation and Masking",
        "6. Texture and Geometric Description",
        "7. Deep Recognition Model",
        "8. Training Protocol",
        "9. Evaluation Methodology",
        "10. Results",
        "11. Discussion",
        "12. Limitations",
        "13. Reproducibility",
        "14. Conclusion",
        "15. References",
    ]
    for item in toc_items:
        body.append(paragraph(item))
    body.append(page_break())

    for idx, (title, paragraphs) in enumerate(paper_sections(), start=1):
        body.append(paragraph(title, "Heading1"))
        if title.startswith("3. Dataset"):
            body.append(table([
                ["Dataset item", "Value"],
                ["Total samples", str(metrics["num_samples"])],
                ["No-tumor samples", str(label_counts.get("no_tumor", 0))],
                ["Tumor samples", str(label_counts.get("tumor", 0))],
                ["Training samples", str(sum(split_by_label.get("train", {}).values()))],
                ["Validation samples", str(sum(split_by_label.get("validation", {}).values()))],
                ["Test samples", str(sum(split_by_label.get("test", {}).values()))],
            ]))
        for para in paragraphs:
            body.append(paragraph(para))
        if title.startswith("5. Assignment 2"):
            add_figure(body, image_files, "preprocessing_segmentation_preview.png")
        if title.startswith("6. Assignment 3"):
            add_figure(body, image_files, "descriptor_distributions.png")
        if title.startswith("8. Training"):
            add_figure(body, image_files, "training_curves.png")
        if title.startswith("10. Results"):
            body.append(table([
                ["Metric", "Value"],
                ["Accuracy", pct(metrics["test_accuracy"])],
                ["Macro precision", pct(metrics["macro_precision"])],
                ["Macro recall", pct(metrics["macro_recall"])],
                ["Macro F1-score", pct(metrics["macro_f1"])],
                ["Decision threshold", f"{metrics.get('decision_threshold', 0):.4f}"],
            ]))
            body.append(table([
                ["True / Predicted", "no_tumor", "tumor"],
                ["no_tumor", str(cm[0][0]), str(cm[0][1])],
                ["tumor", str(cm[1][0]), str(cm[1][1])],
            ]))
            add_figure(body, image_files, "confusion_matrix.png")
            add_figure(body, image_files, "precision_recall_curve.png")
            add_figure(body, image_files, "sample_predictions.png")
        body.append(page_break())

    body.append(paragraph("15. References", "Heading1"))
    for ref in references():
        body.append(paragraph(ref))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
 mc:Ignorable="w14 wp14">
 <w:body>
 {''.join(body)}
 <w:sectPr>
   <w:pgSz w:w="12240" w:h="15840"/>
   <w:pgMar w:top="1008" w:right="1008" w:bottom="1008" w:left="1008" w:header="720" w:footer="720" w:gutter="0"/>
 </w:sectPr>
 </w:body>
</w:document>"""

    rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
    ]
    for rid, filename in image_rels:
        rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{filename}"/>'
        )
    rels.append("</Relationships>")

    for target in [ROOT_DOCX, REPORT_DOCX]:
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types(image_rels))
            z.writestr("_rels/.rels", package_rels())
            z.writestr("docProps/core.xml", core_props())
            z.writestr("docProps/app.xml", app_props())
            z.writestr("word/document.xml", document_xml)
            z.writestr("word/_rels/document.xml.rels", "\n".join(rels))
            z.writestr("word/styles.xml", styles_xml())
            z.writestr("word/settings.xml", settings_xml())
            for _, path, _ in image_files:
                z.write(path, f"word/media/{path.name}")


def add_figure(body, image_files, filename):
    for title, path, rid in image_files:
        if path.name == filename:
            width, height = png_size(path)
            body.append(image_xml(rid, title, width, height))
            body.append(paragraph(title, italic=True))
            return


def content_types(image_rels):
    image_overrides = "\n".join(
        f'<Override PartName="/word/media/{filename}" ContentType="image/png"/>' for _, filename in image_rels
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  {image_overrides}
</Types>"""


def package_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def core_props():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>Deep Vision Pipeline for Brain Tumor MRI Slice Understanding</dc:title>
 <dc:creator>231192, 231200</dc:creator>
 <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
 <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
 <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def app_props():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
 <Application>Microsoft Office Word</Application>
 <DocSecurity>0</DocSecurity>
 <ScaleCrop>false</ScaleCrop>
 <Company>Air University</Company>
</Properties>"""


def styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
  <w:name w:val="Normal"/>
  <w:qFormat/>
  <w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Title">
  <w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr>
  <w:rPr><w:b/><w:sz w:val="36"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Subtitle">
  <w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:qFormat/>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:rPr><w:i/><w:sz w:val="28"/></w:rPr>
 </w:style>
 <w:style w:type="paragraph" w:styleId="Heading1">
  <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
  <w:pPr><w:keepNext/><w:spacing w:before="260" w:after="160"/></w:pPr>
  <w:rPr><w:b/><w:sz w:val="30"/></w:rPr>
 </w:style>
 <w:style w:type="table" w:styleId="TableGrid">
  <w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:uiPriority w:val="59"/><w:qFormat/>
  <w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders></w:tblPr>
 </w:style>
</w:styles>"""


def settings_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:zoom w:percent="100"/>
 <w:defaultTabStop w:val="720"/>
</w:settings>"""


if __name__ == "__main__":
    make_markdown()
    build_docx()
    print(f"Wrote {ROOT_DOCX}")
    print(f"Wrote {ROOT_MD}")
    print(f"Wrote {REPORT_DOCX}")
    print(f"Wrote {REPORT_MD}")
