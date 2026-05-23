"""
Shared pipeline for the SVM-HOG microglia detection ablation series.

Each ablation notebook constructs a ``Config`` and calls the functions here.
Two switches control the row-by-row deltas:

* ``feature_mode``:
    - ``'hog_coarse'``  — baseline (row 1): HOG-coarse only on the 3 colour channels.
    - ``'full'``        — rows 2–5: spatial bins + colour hist + LBP + intensity
                          stats + HOG-fine + HOG-coarse.

* ``scale_factors``:
    - ``(1.0,)``                 — single-scale detection (rows 1–2).
    - ``(0.75, 1.0, 1.5)``       — image pyramid + cross-scale NMS (rows 3–5).

PCA is held on across every row (per project decision — needed for memory).
The leakage fix (split-before-augment) and rotation/flip augmentation are
likewise constant background, not ablated.
"""

import os
import csv
import shutil
import time
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.feature import hog, local_binary_pattern

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Paths — inputs (shared across all rows) ────────────────────────────
    image_rois_csv: str = "../Image_ROIs.csv"
    source_images:  str = "../Source_images"
    test_dir:       str = "../test_images"

    # ── Paths — patch folders (shared; same random_state=42 split) ────────
    microglia_folder:      str = "./Processed_training_images/Train/Microglia"
    noise_folder:          str = "./Processed_training_images/Train/Noise"
    microglia_val_folder:  str = "./Processed_training_images/Validate/Microglia"
    noise_val_folder:      str = "./Processed_training_images/Validate/Noise"
    microglia_test_folder: str = "./Processed_training_images/Test/Microglia"
    noise_test_folder:     str = "./Processed_training_images/Test/Noise"
    hnm_folder:            str = "./Processed_training_images/HardNegatives"

    # ── Paths — artifact dir (per-row override) ────────────────────────────
    # Inner paths default to None and are filled in by __post_init__ from
    # artifact_dir so each notebook only sets one path.
    artifact_dir:    str = "./microglia-artifacts"
    features_cache:  str = None
    svm_clf_path:    str = None
    scaler_path:     str = None
    pca_path:        str = None
    val_paths_cache: str = None
    hnm_train_final: str = None

    # ── Preprocessing ──────────────────────────────────────────────────────
    image_size:  int = 2048
    window_size: int = 64

    # ── Ablation switches ──────────────────────────────────────────────────
    feature_mode: str   = 'full'        # 'hog_coarse' (row 1) or 'full' (rows 2-5)
    scale_factors: tuple = (1.0,)       # (1.0,) single-scale; (0.75, 1.0, 1.5) pyramid

    # ── Feature extraction ─────────────────────────────────────────────────
    color_conv:            str   = 'BGR2HSV'
    hog_channel:           str   = 'ALL'
    hog_orient:            int   = 9
    hog_pix_per_cell:      int   = 8
    hog_pix_per_cell_fine: int   = 4
    hog_cell_per_block:    int   = 2
    spatial_size:          tuple = (32, 32)
    hist_bins:             int   = 32
    lbp_radius:            int   = 2
    lbp_n_points:          int   = 16
    lbp_n_bins:            int   = 32

    # ── Dimensionality reduction ───────────────────────────────────────────
    pca_n_components: int = 4677   # 90% variance for the full feature space

    # ── Train / validate split ─────────────────────────────────────────────
    val_size:     float = 0.2
    random_state: int   = 42

    # ── SVM ────────────────────────────────────────────────────────────────
    svm_C:        float = 1.0
    svm_max_iter: int   = 10000

    # ── Detection ──────────────────────────────────────────────────────────
    cells_per_step:      int   = 2      # 16-px stride at ppc=8
    detection_threshold: float = 1.5
    nms_iou_thresh:      float = 0.3

    def __post_init__(self):
        # Derived path defaults — only filled in when not explicitly set so a
        # notebook can override a single field without unsetting the rest.
        if self.features_cache  is None: self.features_cache  = f"{self.artifact_dir}/features_cache.npz"
        if self.svm_clf_path    is None: self.svm_clf_path    = f"{self.artifact_dir}/svm_clf.pkl"
        if self.scaler_path     is None: self.scaler_path     = f"{self.artifact_dir}/scaler.pkl"
        if self.pca_path        is None: self.pca_path        = f"{self.artifact_dir}/pca.pkl"
        if self.val_paths_cache is None: self.val_paths_cache = f"{self.artifact_dir}/val_paths.txt"
        if self.hnm_train_final is None: self.hnm_train_final = f"{self.artifact_dir}/hnm_train_final.npz"


# ════════════════════════════════════════════════════════════════════════════
# I/O utilities
# ════════════════════════════════════════════════════════════════════════════

_COLOR_CODE = {
    'RGB2GRAY': cv2.COLOR_RGB2GRAY, 'RGB2RGBA': cv2.COLOR_RGB2RGBA,
    'RGB2BGR':  cv2.COLOR_RGB2BGR,  'RGB2BGRA': cv2.COLOR_RGB2BGRA,
    'RGB2HSV':  cv2.COLOR_RGB2HSV,  'RGB2HLS':  cv2.COLOR_RGB2HLS,
    'RGB2LUV':  cv2.COLOR_RGB2LUV,  'RGB2YUV':  cv2.COLOR_RGB2YUV,
    'RGB2YCrCb':cv2.COLOR_RGB2YCrCb,
    'BGR2GRAY': cv2.COLOR_BGR2GRAY, 'BGR2BGRA': cv2.COLOR_BGR2BGRA,
    'BGR2RGB':  cv2.COLOR_BGR2RGB,  'BGR2RGBA': cv2.COLOR_BGR2RGBA,
    'BGR2HSV':  cv2.COLOR_BGR2HSV,  'BGR2HLS':  cv2.COLOR_BGR2HLS,
    'BGR2LUV':  cv2.COLOR_BGR2LUV,  'BGR2YUV':  cv2.COLOR_BGR2YUV,
    'BGR2YCrCb':cv2.COLOR_BGR2YCrCb,
}


def convert_color(img: np.ndarray, conv_code: str) -> np.ndarray:
    """Convert *img* to the colour space named by *conv_code*."""
    return cv2.cvtColor(img, _COLOR_CODE[conv_code])


def clear_folder(folder_path: str) -> None:
    """Delete all contents of *folder_path*, creating it if absent."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return
    for name in os.listdir(folder_path):
        entry = os.path.join(folder_path, name)
        shutil.rmtree(entry) if os.path.isdir(entry) else os.remove(entry)


def save_patch(img: np.ndarray, folder: str, filename: str) -> None:
    """Write *img* as JPEG to *folder/filename*."""
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, filename), img)


def load_rois_for_image(image_name: str, csv_path: str) -> dict:
    """Return a dict of bbox dicts (keys x, y, w, h) for *image_name*."""
    rois, count = {}, 0
    try:
        with open(csv_path, newline='') as fh:
            for row in csv.reader(fh):
                if len(row) > 5 and row[5] == image_name + ".png" and row[0] == "0":
                    rois[count] = {"x": row[1], "y": row[2], "w": row[3], "h": row[4]}
                    count += 1
    except FileNotFoundError:
        print(f"Warning: CSV not found — {csv_path}")
    except Exception as exc:
        print(f"Warning: could not read CSV — {exc}")
    return rois


def list_images(folder: str) -> list:
    """Return sorted list of image paths in *folder*."""
    exts = ('.jpg', '.jpeg', '.png')
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )


def plot_image(img: np.ndarray, title: str = "", show: bool = True) -> None:
    """Display *img* with an optional *title* when *show* is True."""
    if not show:
        return
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Image processing — patch cropping + augmentation
# ════════════════════════════════════════════════════════════════════════════

def _clamp_crop(start: int, size: int, window: int):
    """Clamp start so a window of *window* pixels stays inside [0, size]."""
    start = max(0, start)
    end   = start + window
    if end > size:
        end   = size
        start = max(0, end - window)
    return start, end


def crop_centered(img: np.ndarray, bbox: dict, window_size: int, image_size: int) -> np.ndarray:
    """Crop a window_size × window_size patch centred on the bbox centre."""
    x = int(float(bbox["x"]));  y = int(float(bbox["y"]))
    w = int(float(bbox["w"]));  h = int(float(bbox["h"]))
    cx = x + w // 2
    cy = y + h // 2
    half = window_size // 2
    ystart, yend = _clamp_crop(cy - half, image_size, window_size)
    xstart, xend = _clamp_crop(cx - half, image_size, window_size)
    return img[ystart:yend, xstart:xend]


def crop_grid(img: np.ndarray, col: int, row: int, window_size: int, image_size: int) -> np.ndarray:
    """Crop patch at grid position (*col*, *row*) in pixel space."""
    ystart, yend = _clamp_crop(row, image_size, window_size)
    xstart, xend = _clamp_crop(col, image_size, window_size)
    return img[ystart:yend, xstart:xend]


def build_inv_mask(bbox: dict, mask: np.ndarray) -> np.ndarray:
    """Fill the microglia bounding box on *mask* (in-place)."""
    x = int(float(bbox["x"]));  y = int(float(bbox["y"]))
    w = int(float(bbox["w"]));  h = int(float(bbox["h"]))
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def save_augmented_patches(patch: np.ndarray, folder: str, stem: str, idx: int) -> None:
    """Save 8 augmented variants: 4 rotations × 2 flip states."""
    rotations = [
        patch,
        cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(patch, cv2.ROTATE_180),
        cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    flipped      = [cv2.flip(r, 1) for r in rotations]
    all_variants = rotations + flipped
    for i, variant in enumerate(all_variants):
        save_patch(variant, folder, f"{stem}{idx}_a{i}.jpg")


# ════════════════════════════════════════════════════════════════════════════
# Preprocessing pipeline
# ════════════════════════════════════════════════════════════════════════════

def extract_patches_from_image(
    img_path: str,
    microglia_folder: str,
    noise_folder: str,
    cfg: Config,
    augment: bool = True,
) -> None:
    """Extract positive (microglia) + negative (noise) patches from one image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read {img_path}")
        return

    stem = os.path.splitext(os.path.basename(img_path))[0]
    rois = load_rois_for_image(stem, cfg.image_rois_csv)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for idx, bbox in rois.items():
        mask  = build_inv_mask(bbox, mask)
        patch = crop_centered(img, bbox, cfg.window_size, cfg.image_size)
        if augment:
            save_augmented_patches(patch, microglia_folder, stem, idx)
        else:
            save_patch(patch, microglia_folder, f"{stem}{idx}.jpg")

    bg_img  = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    n_steps = (cfg.image_size // cfg.window_size) - 1

    noise_idx = 0
    for col_i in range(n_steps):
        for row_i in range(n_steps):
            patch = crop_grid(
                bg_img,
                col_i * cfg.window_size,
                row_i * cfg.window_size,
                cfg.window_size,
                cfg.image_size,
            )
            save_patch(patch, noise_folder, f"{stem}{noise_idx}.jpg")
            noise_idx += 1


def preprocess_all_images(cfg: Config) -> tuple:
    """Build train/validate/test patch datasets. Split happens BEFORE augmentation."""
    for folder in (
        cfg.microglia_folder,     cfg.noise_folder,
        cfg.microglia_val_folder, cfg.noise_val_folder,
        cfg.microglia_test_folder, cfg.noise_test_folder,
    ):
        clear_folder(folder)

    source_paths = list_images(cfg.source_images)
    train_paths, val_paths = train_test_split(
        source_paths, test_size=cfg.val_size, random_state=cfg.random_state
    )
    test_image_paths = list_images(cfg.test_dir)

    print(f"Source — train: {len(train_paths)}, val: {len(val_paths)}, "
          f"test (fixed): {len(test_image_paths)}\n")

    print("── Train images (augmented) ──")
    for path in train_paths:
        print(f"  {os.path.basename(path)}")
        extract_patches_from_image(path, cfg.microglia_folder, cfg.noise_folder, cfg, augment=True)

    print("\n── Validate images (no augmentation) ──")
    for path in val_paths:
        print(f"  {os.path.basename(path)}")
        extract_patches_from_image(path, cfg.microglia_val_folder, cfg.noise_val_folder, cfg, augment=False)

    print("\n── Test images (held-out) ──")
    for path in test_image_paths:
        print(f"  {os.path.basename(path)}")
        extract_patches_from_image(path, cfg.microglia_test_folder, cfg.noise_test_folder, cfg, augment=False)

    return train_paths, val_paths


# ════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ════════════════════════════════════════════════════════════════════════════

def bin_spatial(img: np.ndarray, size: tuple) -> np.ndarray:
    """Flatten resized per-channel pixel values into a 1-D vector."""
    return np.hstack([
        cv2.resize(img[:, :, ch], size).ravel()
        for ch in range(img.shape[2])
    ])


def color_hist(img: np.ndarray, nbins: int) -> np.ndarray:
    """Concatenated per-channel colour histogram."""
    return np.concatenate([
        np.histogram(img[:, :, ch], bins=nbins)[0]
        for ch in range(img.shape[2])
    ])


def lbp_features(img_gray: np.ndarray, radius: int, n_points: int, n_bins: int) -> np.ndarray:
    """Normalised LBP histogram for a single-channel image."""
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_points + 2), density=True)
    return hist


def intensity_stats(img: np.ndarray) -> np.ndarray:
    """Per-channel mean, standard deviation, and skewness."""
    stats = []
    for ch in range(img.shape[2]):
        ch_vals = img[:, :, ch].ravel().astype(np.float32)
        mean    = ch_vals.mean()
        std     = ch_vals.std() + 1e-6
        skew    = np.mean(((ch_vals - mean) / std) ** 3)
        stats.extend([mean, std, skew])
    return np.array(stats)


def hog_features(
    img: np.ndarray,
    orient: int,
    pix_per_cell: int,
    cell_per_block: int,
    feature_vector: bool = True,
) -> np.ndarray:
    """HOG features for a single-channel image."""
    return hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=True,
        feature_vector=feature_vector,
    )


def _hog_for_channels(img: np.ndarray, orient: int, ppc: int, cpb: int,
                      hog_channel: str) -> np.ndarray:
    """Compute HOG and flatten across the requested channels."""
    if hog_channel == 'ALL':
        return np.ravel([
            hog_features(img[:, :, ch], orient, ppc, cpb, feature_vector=True)
            for ch in range(img.shape[2])
        ])
    ch = int(hog_channel)
    return hog_features(img[:, :, ch], orient, ppc, cpb, feature_vector=True)


def _feature_vec_full(raw_rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Full feature vector: spatial + color hist + LBP + intensity + HOG fine/coarse."""
    img      = convert_color(raw_rgb, cfg.color_conv)
    img_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)

    spatial    = bin_spatial(img, cfg.spatial_size)
    chist      = color_hist(img, cfg.hist_bins)
    lbp_feat   = lbp_features(img_gray, cfg.lbp_radius, cfg.lbp_n_points, cfg.lbp_n_bins)
    int_stats  = intensity_stats(img)
    hog_fine   = _hog_for_channels(img, cfg.hog_orient, cfg.hog_pix_per_cell_fine,
                                   cfg.hog_cell_per_block, cfg.hog_channel)
    hog_coarse = _hog_for_channels(img, cfg.hog_orient, cfg.hog_pix_per_cell,
                                   cfg.hog_cell_per_block, cfg.hog_channel)
    return np.concatenate((spatial, chist, lbp_feat, int_stats, hog_fine, hog_coarse))


def _feature_vec_hog_coarse(raw_rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """Baseline feature vector: HOG-coarse only on 3 colour channels."""
    img = convert_color(raw_rgb, cfg.color_conv)
    return _hog_for_channels(img, cfg.hog_orient, cfg.hog_pix_per_cell,
                             cfg.hog_cell_per_block, cfg.hog_channel)


def extract_features(image_paths: list, cfg: Config, label: str = "") -> list:
    """
    Extract feature vectors for each image path. Branches on cfg.feature_mode:

    * ``'hog_coarse'`` — HOG-coarse only on 3 channels (~5292-d).
    * ``'full'``       — spatial bins + colour hist + LBP + intensity stats +
                         HOG-fine + HOG-coarse (~32801-d).
    """
    if cfg.feature_mode == 'full':
        compute = _feature_vec_full
    elif cfg.feature_mode == 'hog_coarse':
        compute = _feature_vec_hog_coarse
    else:
        raise ValueError(f"Unknown cfg.feature_mode: {cfg.feature_mode!r}")

    features    = []
    n           = len(image_paths)
    prefix      = f"  [{label}] " if label else "  "
    report_step = max(1, n // 10)

    for i, path in enumerate(image_paths):
        if i % report_step == 0:
            print(f"{prefix}{i}/{n} ({100 * i // n:3d}%)", end="\r", flush=True)
        try:
            raw = mpimg.imread(path)
            features.append(compute(raw, cfg))
        except Exception as exc:
            print(f"\nSkipping {path}: {exc}")

    print(f"{prefix}{n}/{n} (100%) — {n} feature vectors extracted")
    return features


# ════════════════════════════════════════════════════════════════════════════
# Dataset setup — extract raw features + fit scaler/PCA
# ════════════════════════════════════════════════════════════════════════════

def extract_raw_features(cfg: Config):
    """Extract unscaled feature vectors from train + validate folders."""
    train_pos_paths = list_images(cfg.microglia_folder)
    train_neg_paths = list_images(cfg.noise_folder)
    val_pos_paths   = list_images(cfg.microglia_val_folder)
    val_neg_paths   = list_images(cfg.noise_val_folder)

    print(f"Train    — microglia: {len(train_pos_paths)}, noise: {len(train_neg_paths)}")
    print(f"Validate — microglia: {len(val_pos_paths)},  noise: {len(val_neg_paths)}")

    print("\nExtracting train features...")
    train_pos_feats = extract_features(train_pos_paths, cfg, label="train microglia")
    train_neg_feats = extract_features(train_neg_paths, cfg, label="train noise")

    print("Extracting validate features...")
    val_pos_feats = extract_features(val_pos_paths, cfg, label="val microglia")
    val_neg_feats = extract_features(val_neg_paths, cfg, label="val noise")

    X_train_raw = np.vstack((train_pos_feats, train_neg_feats)).astype(np.float32)
    y_train_raw = np.hstack((np.ones(len(train_pos_feats)), np.zeros(len(train_neg_feats))))
    X_val_raw   = np.vstack((val_pos_feats, val_neg_feats)).astype(np.float32)
    y_val_raw   = np.hstack((np.ones(len(val_pos_feats)), np.zeros(len(val_neg_feats))))

    print(f"\nRaw feature dim : {X_train_raw.shape[1]}")
    print(f"Train: {len(X_train_raw)} | Validate: {len(X_val_raw)}")
    return X_train_raw, X_val_raw, y_train_raw, y_val_raw


def fit_pipeline(X_train_raw: np.ndarray, X_val_raw: np.ndarray,
                 y_train_raw: np.ndarray, y_val_raw: np.ndarray, cfg: Config):
    """Fit scaler + PCA on X_train_raw; transform train and val."""
    scaler  = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)

    pca = None
    if cfg.pca_n_components > 0:
        n_components = min(int(cfg.pca_n_components), X_train.shape[1], X_train.shape[0])
        pca = PCA(
            n_components=n_components,
            svd_solver='randomized',
            random_state=cfg.random_state,
            copy=False,
        )
        X_train = pca.fit_transform(X_train)
        X_val   = pca.transform(X_val)
        var_retained = pca.explained_variance_ratio_.sum()
        print(f"PCA: {X_train_raw.shape[1]} → {pca.n_components_} components "
              f"({var_retained:.1%} variance retained)")

    print(f"Train: {len(X_train)} | Validate: {len(X_val)}")
    return X_train, X_val, y_train_raw, y_val_raw, scaler, pca


# ════════════════════════════════════════════════════════════════════════════
# Hyperparameter tuning + training + evaluation
# ════════════════════════════════════════════════════════════════════════════

def tune_svm(X_train: np.ndarray, y_train: np.ndarray, cfg: Config) -> dict:
    """GridSearchCV over LinearSVC(C). Updates cfg.svm_C in place."""
    param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    base_clf = LinearSVC(
        max_iter=cfg.svm_max_iter,
        class_weight=None,
        dual=False,
        random_state=cfg.random_state,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    grid = GridSearchCV(base_clf, param_grid, cv=cv, scoring='f1', n_jobs=1, verbose=1)
    grid.fit(X_train, y_train)

    print("\nC        mean-F1  std-F1")
    for c, mean, std in zip(
        param_grid["C"],
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"],
    ):
        marker = " ← best" if c == grid.best_params_["C"] else ""
        print(f"{c:<8} {mean:.4f}   ±{std:.4f}{marker}")

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV F1  : {grid.best_score_:.4f}")
    cfg.svm_C = grid.best_params_["C"]
    return grid.best_params_


def train_svm(X_train: np.ndarray, y_train: np.ndarray, cfg: Config) -> LinearSVC:
    """Fit LinearSVC with cfg.svm_C and return the trained classifier."""
    clf = LinearSVC(
        C=cfg.svm_C,
        max_iter=cfg.svm_max_iter,
        class_weight=None,
        dual=False,
        random_state=cfg.random_state,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"Training time : {time.time() - t0:.2f}s")
    return clf


def evaluate_classifier(clf: LinearSVC, X: np.ndarray, y: np.ndarray, label: str = "") -> None:
    """Print accuracy, precision, recall, F1 for a given split."""
    y_pred = clf.predict(X)
    acc    = (y_pred == y).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Accuracy  : {acc:.4f}")
    print(f"{prefix}Precision : {prec:.4f}")
    print(f"{prefix}Recall    : {rec:.4f}")
    print(f"{prefix}F1-Score  : {f1:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# Detection pipeline
# ════════════════════════════════════════════════════════════════════════════

def _precompute_full(img_c: np.ndarray, img_gray: np.ndarray, cfg: Config):
    """Precompute LBP map + integral images used by the full feature vector."""
    lbp_map = local_binary_pattern(
        img_gray.astype(np.float64), cfg.lbp_n_points, cfg.lbp_radius, method='uniform'
    )
    integ_sum, integ_sq, integ_cu = [], [], []
    for ch in range(img_c.shape[2]):
        ch_f = img_c[:, :, ch].astype(np.float64)
        integ_sum.append(cv2.integral(ch_f))
        integ_sq.append(cv2.integral(ch_f * ch_f))
        integ_cu.append(cv2.integral(ch_f * ch_f * ch_f))
    return lbp_map, integ_sum, integ_sq, integ_cu


def _intensity_stats_fast(integ_sum, integ_sq, integ_cu, ytop, xleft, win, n_ch):
    """O(1) per-window intensity stats from integral images."""
    def _query(integ, y0, x0, y1, x1):
        return integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    n = win * win
    stats = []
    for ch in range(n_ch):
        s1 = _query(integ_sum[ch], ytop, xleft, ytop + win, xleft + win)
        s2 = _query(integ_sq[ch],  ytop, xleft, ytop + win, xleft + win)
        s3 = _query(integ_cu[ch],  ytop, xleft, ytop + win, xleft + win)
        mean = s1 / n
        var  = max(s2 / n - mean * mean, 0.0)
        std  = var ** 0.5 + 1e-6
        skew = (s3 / n - 3 * mean * (s2 / n) + 2 * mean ** 3) / (std ** 3)
        stats.extend([mean, std, skew])
    return np.array(stats, dtype=np.float32)


def sliding_window_detect(
    img: np.ndarray,
    clf: LinearSVC,
    scaler: StandardScaler,
    cfg: Config,
    pca: PCA = None,
    _batch_size: int = 512,
) -> tuple:
    """
    Sweep *img* with a sliding window; return scored detections.

    Branches on cfg.feature_mode — the precompute and per-window feature build
    only run the components needed by the active mode.
    """
    img_uint8 = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
    img_c     = convert_color(img_uint8, cfg.color_conv)

    full_mode = cfg.feature_mode == 'full'

    ppc_c = cfg.hog_pix_per_cell
    ppc_f = cfg.hog_pix_per_cell_fine
    cpb   = cfg.hog_cell_per_block
    win   = cfg.window_size
    cps   = cfg.cells_per_step

    pixel_stride  = cps * ppc_c
    cps_f         = pixel_stride // ppc_f
    nblocks_win_c = (win // ppc_c) - (cpb - 1)
    nxblocks_c    = (img_c.shape[1] // ppc_c) - 1
    nyblocks_c    = (img_c.shape[0] // ppc_c) - 1
    nxsteps       = (nxblocks_c - nblocks_win_c) // cps
    nysteps       = (nyblocks_c - nblocks_win_c) // cps
    n_ch          = img_c.shape[2]

    hog_maps_c = [
        hog_features(img_c[:, :, ch], cfg.hog_orient, ppc_c, cpb, feature_vector=False)
        for ch in range(n_ch)
    ]

    if full_mode:
        nblocks_win_f = (win // ppc_f) - (cpb - 1)
        hog_maps_f = [
            hog_features(img_c[:, :, ch], cfg.hog_orient, ppc_f, cpb, feature_vector=False)
            for ch in range(n_ch)
        ]
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        lbp_map, integ_sum, integ_sq, integ_cu = _precompute_full(img_c, img_gray, cfg)

    detections, det_scores = [], []
    batch_feats, batch_pos = [], []
    report_every = max(1, nxsteps // 10)

    def _flush():
        if not batch_feats:
            return
        X = np.array(batch_feats, dtype=np.float32)
        X = scaler.transform(X)
        if pca is not None:
            X = pca.transform(X)
        for (xleft, ytop), score in zip(batch_pos, clf.decision_function(X)):
            if score > cfg.detection_threshold:
                detections.append(((xleft, ytop), (xleft + win, ytop + win)))
                det_scores.append(float(score))
        batch_feats.clear()
        batch_pos.clear()

    for xb in range(nxsteps):
        if xb % report_every == 0:
            print(f"    {100 * xb // nxsteps:3d}%", end=" ", flush=True)
        for yb in range(nysteps):
            xpos_c, ypos_c = xb * cps, yb * cps
            xleft = xpos_c * ppc_c
            ytop  = ypos_c * ppc_c

            hog_coarse = np.hstack([
                hm[ypos_c:ypos_c + nblocks_win_c,
                   xpos_c:xpos_c + nblocks_win_c].ravel()
                for hm in hog_maps_c
            ])

            if full_mode:
                xpos_f, ypos_f = xb * cps_f, yb * cps_f
                hog_fine = np.hstack([
                    hm[ypos_f:ypos_f + nblocks_win_f,
                       xpos_f:xpos_f + nblocks_win_f].ravel()
                    for hm in hog_maps_f
                ])
                lbp_feat = np.histogram(
                    lbp_map[ytop:ytop + win, xleft:xleft + win],
                    bins=cfg.lbp_n_bins, range=(0, cfg.lbp_n_points + 2), density=True,
                )[0].astype(np.float32)
                int_st = _intensity_stats_fast(integ_sum, integ_sq, integ_cu,
                                               ytop, xleft, win, n_ch)
                sub     = img_c[ytop:ytop + win, xleft:xleft + win]
                spatial = bin_spatial(sub, cfg.spatial_size)
                chist   = color_hist(sub, cfg.hist_bins)
                feat = np.hstack((spatial, chist, lbp_feat, int_st, hog_fine, hog_coarse))
            else:
                feat = hog_coarse

            batch_feats.append(feat)
            batch_pos.append((xleft, ytop))

            if len(batch_feats) >= _batch_size:
                _flush()

    _flush()
    print("100%")
    return detections, det_scores


def non_max_suppression(detections: list, scores: list, iou_thresh: float) -> list:
    """Greedy score-ordered NMS — suppress boxes with IoU > iou_thresh."""
    if not detections:
        return []
    boxes  = np.array([[x0, y0, x1, y1] for (x0, y0), (x1, y1) in detections], dtype=float)
    scores = np.array(scores)
    order  = scores.argsort()[::-1]
    kept   = []
    while order.size > 0:
        i = order[0]
        kept.append(i)
        if order.size == 1:
            break
        rest  = order[1:]
        xi1   = np.maximum(boxes[i, 0], boxes[rest, 0])
        yi1   = np.maximum(boxes[i, 1], boxes[rest, 1])
        xi2   = np.minimum(boxes[i, 2], boxes[rest, 2])
        yi2   = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        area_i    = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou       = inter / (area_i + area_rest - inter + 1e-6)
        order     = rest[iou <= iou_thresh]
    return [detections[i] for i in kept]


def multi_scale_detect(
    img: np.ndarray,
    clf: LinearSVC,
    scaler: StandardScaler,
    cfg: Config,
    pca: PCA = None,
) -> list:
    """
    Sliding-window detection across cfg.scale_factors; map boxes back to original
    coordinates; apply cross-scale NMS. With ``cfg.scale_factors == (1.0,)`` this
    is equivalent to single-scale + NMS.
    """
    all_dets, all_scores = [], []
    for scale in cfg.scale_factors:
        h, w   = img.shape[:2]
        sh, sw = int(h * scale), int(w * scale)
        print(f"  scale={scale:.2f} ({sw}×{sh}px)", end="  ", flush=True)
        scaled = cv2.resize(img, (sw, sh)) if scale != 1.0 else img
        dets, scores = sliding_window_detect(scaled, clf, scaler, cfg, pca)
        mapped = [
            ((int(x0 / scale), int(y0 / scale)),
             (int(x1 / scale), int(y1 / scale)))
            for (x0, y0), (x1, y1) in dets
        ]
        all_dets.extend(mapped)
        all_scores.extend(scores)
        print(f"  → {len(mapped)} candidates at this scale")
    print(f"  Total candidates : {len(all_dets)}")
    final = non_max_suppression(all_dets, all_scores, cfg.nms_iou_thresh)
    print(f"  After NMS        : {len(final)}")
    return final, all_dets, all_scores


def draw_boxes(img: np.ndarray, boxes: list) -> np.ndarray:
    """Draw bounding boxes on a copy of img."""
    out = img.copy()
    for (x0, y0), (x1, y1) in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 6)
    return out


def process_image(
    img: np.ndarray,
    clf: LinearSVC,
    scaler: StandardScaler,
    cfg: Config,
    pca: PCA = None,
) -> tuple:
    """Run detection (single- or multi-scale per cfg) + NMS. Returns (annotated, boxes)."""
    boxes, _, _ = multi_scale_detect(img, clf, scaler, cfg, pca)
    return draw_boxes(img, boxes), boxes


# ════════════════════════════════════════════════════════════════════════════
# Ground-truth evaluation
# ════════════════════════════════════════════════════════════════════════════

def compute_iou(boxA: tuple, boxB: tuple) -> float:
    """boxA, boxB: (x0, y0, x1, y1). Returns IoU in [0, 1]."""
    xi1 = max(boxA[0], boxB[0]);  yi1 = max(boxA[1], boxB[1])
    xi2 = min(boxA[2], boxB[2]);  yi2 = min(boxA[3], boxB[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter + 1e-6)


def load_gt_boxes(image_stem: str, csv_path: str) -> list:
    """Return list of (x0, y0, x1, y1) ground-truth boxes from the ROI CSV."""
    boxes = []
    try:
        with open(csv_path, newline='') as fh:
            for row in csv.reader(fh):
                if len(row) > 5 and row[5] == image_stem + ".png" and row[0] == "0":
                    x, y, w, h = int(row[1]), int(row[2]), int(row[3]), int(row[4])
                    boxes.append((x, y, x + w, y + h))
    except FileNotFoundError:
        print(f"Warning: CSV not found — {csv_path}")
    return boxes


def evaluate_detections(pred_boxes: list, gt_boxes: list, iou_thresh: float = 0.5) -> dict:
    """Greedy IoU matching. Returns tp/fp/fn/precision/recall."""
    matched_gt = set()
    tp = 0
    for (x0, y0), (x1, y1) in pred_boxes:
        pred_flat = (x0, y0, x1, y1)
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pred_flat, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
    fp        = len(pred_boxes) - tp
    fn        = len(gt_boxes)   - tp
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def plot_gt_vs_pred(img: np.ndarray, pred_boxes: list, gt_boxes: list, title: str = "") -> None:
    """Side-by-side: GT boxes (green) left, predicted boxes (red) right."""
    def _draw(base, boxes, colour):
        out = base.copy()
        for box in boxes:
            if isinstance(box[0], tuple):
                (x0, y0), (x1, y1) = box
            else:
                x0, y0, x1, y1 = box
            cv2.rectangle(out, (x0, y0), (x1, y1), colour, 6)
        return out
    left  = _draw(img, gt_boxes,   (0, 255, 0))
    right = _draw(img, pred_boxes, (255, 0, 0))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(left);  axes[0].set_title(f"Ground truth ({len(gt_boxes)} cells)")
    axes[1].imshow(right); axes[1].set_title(f"Predicted ({len(pred_boxes)} boxes)")
    for ax in axes:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Hard negative mining (used by row 4)
# ════════════════════════════════════════════════════════════════════════════

def save_hard_negatives(
    image: np.ndarray,
    detections: list,
    gt_boxes: list,
    hnm_folder: str,
    stem: str,
    hnm_round: int,
    iou_thresh: float = 0.5,
    window_size: int = 64,
) -> list:
    """
    Save each FP detection crop to hnm_folder. Filenames are prefixed with the
    HNM round so crops mined in different rounds never collide. Crops are
    RGB→BGR converted before saving so they match the base-patch JPEG convention.
    """
    saved_paths = []
    for i, ((x0, y0), (x1, y1)) in enumerate(detections):
        pred_flat = (x0, y0, x1, y1)
        is_tp = any(compute_iou(pred_flat, gt) >= iou_thresh for gt in gt_boxes)
        if not is_tp:
            crop = image[y0:y1, x0:x1]
            if crop.dtype != np.uint8:
                crop = (crop * 255).astype(np.uint8)
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                if crop.shape[0] != window_size or crop.shape[1] != window_size:
                    crop = cv2.resize(crop, (window_size, window_size))
                fname = f"hn_r{hnm_round}_{stem}_{i}.jpg"
                save_patch(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), hnm_folder, fname)
                saved_paths.append(os.path.join(hnm_folder, fname))
    return saved_paths


# ════════════════════════════════════════════════════════════════════════════
# Detection-level threshold tuning (used by row 5)
# ════════════════════════════════════════════════════════════════════════════

def tune_detection_threshold(
    val_paths: list,
    clf: LinearSVC,
    scaler: StandardScaler,
    cfg: Config,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pca: PCA = None,
    n_steps: int = 15,
) -> float:
    """
    Sweep cfg.detection_threshold and pick the value that maximises detection-
    level F1 (TP/FP/FN via IoU >= 0.5 matching).

    Cached-score sweep: run the multi-scale window ONCE per validate image to
    capture all (box, score) pairs pre-NMS, then the threshold sweep just
    filters those cached scores and re-runs NMS on the survivors. Updates
    cfg.detection_threshold in place and returns the best value.
    """
    pos_scores = clf.decision_function(X_val[y_val == 1])
    t_min      = max(0.0, float(pos_scores.min()))
    t_max      = float(pos_scores.max())
    thresholds = np.linspace(t_min, t_max, n_steps)

    saved_thresh = cfg.detection_threshold
    cfg.detection_threshold = -np.inf

    cached = []
    t0 = time.time()
    print("Caching candidates (one detection pass per image)...")
    for path in val_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img  = mpimg.imread(path)
        print(f"  {stem}")
        # Inline the pyramid loop to capture scores (multi_scale_detect's
        # NMS-then-return drops them).
        all_dets, all_scores = [], []
        for scale in cfg.scale_factors:
            h, w   = img.shape[:2]
            sh, sw = int(h * scale), int(w * scale)
            scaled = cv2.resize(img, (sw, sh)) if scale != 1.0 else img
            dets, scores = sliding_window_detect(scaled, clf, scaler, cfg, pca)
            mapped = [
                ((int(x0 / scale), int(y0 / scale)),
                 (int(x1 / scale), int(y1 / scale)))
                for (x0, y0), (x1, y1) in dets
            ]
            all_dets.extend(mapped)
            all_scores.extend(scores)
        gt_boxes = load_gt_boxes(stem, cfg.image_rois_csv)
        cached.append((all_dets, all_scores, gt_boxes))
        print(f"    {len(all_dets)} raw candidates, {len(gt_boxes)} GT boxes")

    cfg.detection_threshold = saved_thresh
    print(f"Cache built in {time.time() - t0:.1f}s.\n")

    print(f"Sweeping threshold {t_min:.2f} → {t_max:.2f} ({n_steps} steps)")
    results = []
    for t in thresholds:
        total_tp = total_fp = total_fn = 0
        for all_dets, all_scores, gt_boxes in cached:
            kept_dets, kept_scores = [], []
            for d, s in zip(all_dets, all_scores):
                if s > t:
                    kept_dets.append(d)
                    kept_scores.append(s)
            boxes = non_max_suppression(kept_dets, kept_scores, cfg.nms_iou_thresh)
            m = evaluate_detections(boxes, gt_boxes, iou_thresh=0.5)
            total_tp += m['tp']; total_fp += m['fp']; total_fn += m['fn']

        prec = total_tp / (total_tp + total_fp + 1e-6)
        rec  = total_tp / (total_tp + total_fn + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)
        results.append((float(t), prec, rec, f1))
        print(f"  t={t:.2f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    best_t, best_p, best_r, best_f1 = max(results, key=lambda x: x[3])
    cfg.detection_threshold = best_t

    ts, ps, rs, fs = zip(*results)
    plt.figure(figsize=(10, 4))
    plt.plot(ts, ps, label='Precision')
    plt.plot(ts, rs, label='Recall')
    plt.plot(ts, fs, label='F1')
    plt.axvline(best_t, color='r', linestyle='--', label=f'Best t={best_t:.2f}')
    plt.xlabel('Detection threshold (SVM decision score)')
    plt.ylabel('Score')
    plt.title('Detection-level P / R / F1 vs Threshold (validate)')
    plt.legend(); plt.tight_layout(); plt.show()

    print(f"\nBest threshold: {best_t:.2f}  "
          f"P={best_p:.3f}  R={best_r:.3f}  F1={best_f1:.3f}")
    return best_t


# ════════════════════════════════════════════════════════════════════════════
# Convenience runners (used by every notebook)
# ════════════════════════════════════════════════════════════════════════════

def ensure_test_patches(cfg: Config) -> None:
    """Idempotent: populate Test/{Microglia,Noise} from test_dir/ if empty."""
    if list_images(cfg.microglia_test_folder):
        return
    print("Extracting test patches from test_dir/ ...")
    for path in list_images(cfg.test_dir):
        extract_patches_from_image(
            path, cfg.microglia_test_folder, cfg.noise_test_folder, cfg, augment=False,
        )
    n_pos = len(list_images(cfg.microglia_test_folder))
    n_neg = len(list_images(cfg.noise_test_folder))
    print(f"Extracted — microglia: {n_pos}, noise: {n_neg}\n")
    if n_pos == 0:
        raise RuntimeError(
            f"No test microglia patches extracted from {cfg.test_dir!r}. "
            f"Check that {cfg.image_rois_csv} has GT entries matching the image stems."
        )


def patch_level_test_eval(
    svm_clf: LinearSVC, scaler: StandardScaler, pca: PCA, cfg: Config,
) -> None:
    """Extract test patch features, transform, evaluate. Prints metrics only."""
    test_pos_paths = list_images(cfg.microglia_test_folder)
    test_neg_paths = list_images(cfg.noise_test_folder)
    test_pos_feats = extract_features(test_pos_paths, cfg, label="test microglia")
    test_neg_feats = extract_features(test_neg_paths, cfg, label="test noise")
    X_test = np.vstack((test_pos_feats, test_neg_feats)).astype(np.float32)
    y_test = np.hstack((np.ones(len(test_pos_feats)), np.zeros(len(test_neg_feats))))
    X_test = scaler.transform(X_test)
    if pca is not None:
        X_test = pca.transform(X_test)
    evaluate_classifier(svm_clf, X_test, y_test, label="test")


def detection_level_test_eval(
    svm_clf: LinearSVC, scaler: StandardScaler, pca: PCA, cfg: Config,
    show_plots: bool = True,
) -> dict:
    """
    Run detection (single- or multi-scale per cfg) on every test image; aggregate
    TP/FP/FN at IoU >= 0.5. Returns the aggregate dict.
    """
    final_test_paths = list_images(cfg.test_dir)
    print(f"Running detection on {len(final_test_paths)} test images "
          f"at scales {cfg.scale_factors}, threshold={cfg.detection_threshold:.2f}\n")

    total_tp = total_fp = total_fn = 0
    for path in final_test_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img  = mpimg.imread(path)
        print(f"── {stem} ──")
        t0 = time.time()
        result, boxes = process_image(img, svm_clf, scaler, cfg, pca=pca)
        print(f"  Detection time: {time.time() - t0:.1f}s")

        gt_boxes = load_gt_boxes(stem, cfg.image_rois_csv)
        if gt_boxes:
            m = evaluate_detections(boxes, gt_boxes, iou_thresh=0.5)
            total_tp += m["tp"]; total_fp += m["fp"]; total_fn += m["fn"]
            print(f"  GT={len(gt_boxes)}  Pred={len(boxes)}  "
                  f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}")
            if show_plots:
                plot_gt_vs_pred(img, boxes, gt_boxes, title=stem)
        else:
            print("  (no GT)")
            if show_plots:
                plot_image(result, title=f"{stem} — {len(boxes)} cells detected")
        print()

    prec = total_tp / (total_tp + total_fp + 1e-6)
    rec  = total_tp / (total_tp + total_fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    print(f"== AGGREGATE  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"   Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
    return {"tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": prec, "recall": rec, "f1": f1}
