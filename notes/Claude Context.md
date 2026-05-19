**Project:** [[../../Past Projects|CV Past Projects]] | [[../../../Claude Context|ML Interview Prep Context]] | [[Errors With Old Code]] | [[Daily Notes/2026-05-03]] | [[Daily Notes/2026-05-04]]

# SVM-HOG Microglia Detection Project

Example file convention: github.com/forrestchang/andrej-karpathy-skills
## Project Overview

This is a machine learning project for **automated detection of microglia cells** in laboratory images (2048x2048 pixels). The system uses Support Vector Machine (SVM) classifiers trained on HOG (Histogram of Oriented Gradients) features and color information.

**Goal**: Accurately identify and localize microglia cells in microscopy images while minimizing false positives.

## Data & Preprocessing

- **Training Images**: 13 images (2048x2048 pixels) with manually labeled microglia cells
- **Labeling Tool**: MakeSense.AI
- **Positive Samples**: 64x64 pixel crops of microglia cells
- **Negative Samples**: 65x65 pixel crops of background/noise (created by masking out labeled cells and sampling remaining regions)
- **Data Augmentation**: Rotated images (90°, 180°, 270°) to increase dataset size

**Known Data Leakage Issue**: Rotation augmentation is currently applied before the train/test split, meaning rotated versions of the same cell appear in both sets. Fix: split source images first, then rotate only training images.

## Feature Extraction

- **HOG Features**: Histogram of Oriented Gradients using scikit-image
- **Color Features**: Color histograms and color bin distributions
- **LBP**: Local Binary Patterns
- **Intensity stats**: mean, std, skewness per channel
- **Spatial bins**: resized patch histogram
- **Feature vector layout**: `[spatial_bins | color_hist | lbp | intensity_stats | hog_fine | hog_coarse]`
- **Libraries**: OpenCV, scikit-learn, scikit-image, Matplotlib, NumPy

## Model Approaches

### 1. SVM_original.ipynb
Baseline SVM model with single window size (64x64). Uses HOG + color features.

### 2. SVM_window variants (50, 64, 80, 100)
Experiments with different window sizes. Some cells are larger than 64x64, so multiple sizes help.

### 3. SVM_ensemble.ipynb
Ensemble approach combining predictions from multiple window sizes with weighted voting.

### 4. SVM_rotated_data.ipynb
Uses rotationally augmented training data for better generalization.

### 5. Hard Negative Mining (`Hard Negative Mining/`)
Targeted dataset improvement using misclassified negatives from full-image inference.

### 6. SVM_original_refactored.ipynb (2026/ — current best version)
Major refactor with:
- `Config` dataclass for all hyperparameters
- Multi-scale image pyramid (scales: 0.75, 1.0, 1.5) with cross-scale NMS
- GridSearchCV for SVM hyperparameter tuning
- PCA dimensionality reduction (95% variance, ~4997 components from ~32801)
- Pre-computed HOG maps + LBP maps + integral image intensity stats for fast sliding window

## Prediction Pipeline

1. **Sliding Window**: 8x8 pixel stride over 2048x2048 image (per scale)
2. **Heatmap Generation**: Increment heatmap for each positive prediction
3. **Thresholding**: Apply threshold to filter false positives
4. **NMS**: Cross-scale non-maximum suppression
5. **Bounding Boxes**: Draw boxes around detected regions

## Model Saving — Always Save All Three

```python
joblib.dump(svm_clf, "svm_clf.pkl")
joblib.dump(scaler,  "scaler.pkl")
joblib.dump(pca,     "pca.pkl")
```

The scaler and PCA store learned statistics from training. Loading only the classifier makes predictions meaningless.

## Key Directories

```
.
├── README.md
├── LICENSE.txt
├── SVM_*.ipynb                   # Legacy model experiments
├── SVM_ensemble.ipynb
├── 2026/                         # Refactored notebooks (start here)
│   ├── SVM_original_refactored.ipynb
│   ├── SVM_original_fixed.ipynb
│   ├── SVM_window_template_fixed.ipynb
│   ├── SVM_ensemble_fixed.ipynb
│   ├── FIXES_GUIDE.md
│   └── QUICK_REFERENCE.md
├── notes/                        # Junction → Obsidian vault (Claude folder)
├── Processed_training_images/
├── Source_images/
├── test_images/
├── Heatmaps/
├── Hard Negative Mining/
└── Image_ROIs*.csv
```

## Critical Issue: Patch Metrics ≠ Detection Quality

**The most important architectural insight.**

- **Patch evaluation**: balanced split, ~50% positive patches → F1 ~0.92
- **Detection inference**: ~15,000 windows per scale on a 2048×2048 image, only 10–20 actual cells → extreme class imbalance not seen during training

A 92% accurate patch classifier still produces hundreds of false positives when sweeping a full image. The current `find_optimal_threshold` tunes on patch-level F1 — wrong signal.

## Recommended Next Steps (Priority Order)

1. **Tune threshold on detection output**: sweep `cfg.detection_threshold` and measure bounding box precision/recall (IoU ≥ 0.5) on full training/validation images, not patch F1.

2. **Hard negative mining**: run detection pipeline on training images, collect every false positive window crop, add to noise training set, retrain. Teaches the classifier about the specific backgrounds it confuses at inference.

3. **Fix data leakage**: split source images before rotation augmentation.

4. **Make detection-level precision/recall the primary metric**: patch accuracy is useful for debugging feature quality only.

## Session Prompt Template

Copy, fill in the two bracketed fields, and paste as your first message each session:

```
Role: You are a computer vision engineer helping me prepare a 7-minute interview presentation for a pharmaceutical company CV role.
Context: SVM-HOG microglia detection project, 2023–2026. Active notebook: 2026/SVM_original_refactored.ipynb
Goal: [fill in]
Constraints: 16hr total budget remaining, 7-min presentation, pharma CV audience, detection-level metrics only
Output: [fill in — e.g. notebook cells / slide content / filled Experiment Story gaps / Q&A prep]
```

---

## Current Status (Updated: 2026-05-04)

- Active notebook: `2026/SVM_original_refactored.ipynb`
- `notes/` directory is a Windows junction pointing to Obsidian vault (`ML Interview Prep/CV/SVM-HOG Microglia Detection/Claude/`)
- Session knowledge captured in daily notes under `notes/Daily Notes/`
