"""Plot detection P-R curves for row 1 (HOG-coarse) vs row 2 (full features)
on the validate split. Uses cached pre-NMS scores per image, then sweeps
thresholds in [0, max(score)] and runs NMS+evaluation on survivors at each.

Output: 2026/pr_curve_row1_vs_row2.png + stdout table.
"""
import os
import sys
import time
import joblib
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

REPO = "/home/ec2-user/lx-repos/SVM-HOG-microglia-detection"
sys.path.insert(0, f"{REPO}/2026")
os.chdir(f"{REPO}/2026")

from pipeline import (
    Config,
    sliding_window_detect, non_max_suppression,
    load_gt_boxes, evaluate_detections,
    list_images,
)
from sklearn.model_selection import train_test_split


def cache_validate_scores(cfg):
    svm_clf = joblib.load(cfg.svm_clf_path)
    scaler  = joblib.load(cfg.scaler_path)
    pca     = joblib.load(cfg.pca_path)

    source_paths = list_images(cfg.source_images)
    _, val_paths = train_test_split(
        source_paths, test_size=cfg.val_size, random_state=cfg.random_state
    )

    saved_thresh = cfg.detection_threshold
    cfg.detection_threshold = -np.inf
    cached = []
    t0 = time.time()
    for path in val_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = mpimg.imread(path)
        all_dets, all_scores = [], []
        for scale in cfg.scale_factors:
            h, w = img.shape[:2]
            sh, sw = int(h * scale), int(w * scale)
            scaled = cv2.resize(img, (sw, sh)) if scale != 1.0 else img
            dets, scores = sliding_window_detect(scaled, svm_clf, scaler, cfg, pca)
            mapped = [
                ((int(x0 / scale), int(y0 / scale)), (int(x1 / scale), int(y1 / scale)))
                for (x0, y0), (x1, y1) in dets
            ]
            all_dets.extend(mapped)
            all_scores.extend(scores)
        gt_boxes = load_gt_boxes(stem, cfg.image_rois_csv)
        cached.append((all_dets, all_scores, gt_boxes))
        print(f"  {stem}: {len(all_dets)} candidates, {len(gt_boxes)} GT")
    cfg.detection_threshold = saved_thresh
    print(f"Cache built in {time.time() - t0:.1f}s")
    return cached


def sweep(cached, thresholds):
    results = []
    for t in thresholds:
        total_tp = total_fp = total_fn = 0
        for all_dets, all_scores, gt_boxes in cached:
            kept_dets, kept_scores = [], []
            for d, s in zip(all_dets, all_scores):
                if s > t:
                    kept_dets.append(d)
                    kept_scores.append(s)
            boxes = non_max_suppression(kept_dets, kept_scores, 0.3)
            m = evaluate_detections(boxes, gt_boxes, iou_thresh=0.5)
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
        p = total_tp / (total_tp + total_fp + 1e-6)
        r = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        results.append((float(t), p, r, f1, total_tp, total_fp, total_fn))
    return results


def build_thresholds(cached, n=40):
    all_scores = np.concatenate([np.asarray(c[1]) for c in cached]) if cached else np.array([])
    if len(all_scores) == 0:
        return np.array([0.0])
    t_min = max(0.0, float(np.percentile(all_scores, 5)))
    t_max = float(np.percentile(all_scores, 99.5))
    ts = list(np.linspace(t_min, t_max, n))
    if 1.5 not in ts:
        ts.append(1.5)
    return np.array(sorted(set(ts)))


print("=== Row 1 (HOG-coarse, single scale) ===")
cfg1 = Config(
    artifact_dir='./microglia-artifacts-row1',
    feature_mode='hog_coarse',
    scale_factors=(1.0,),
    pca_n_components=1215,
)
cached1 = cache_validate_scores(cfg1)
ts1 = build_thresholds(cached1)
r1 = sweep(cached1, ts1)

print("\n=== Row 2 (full features, single scale) ===")
cfg2 = Config(
    artifact_dir='./microglia-artifacts-row2',
    feature_mode='full',
    scale_factors=(1.0,),
    pca_n_components=4677,
)
cached2 = cache_validate_scores(cfg2)
ts2 = build_thresholds(cached2)
r2 = sweep(cached2, ts2)


def at_threshold(results, t_target):
    for t, p, r, f1, *_ in results:
        if abs(t - t_target) < 1e-6:
            return t, p, r, f1
    return None


def best_f1(results):
    return max(results, key=lambda x: x[3])


# Plot P-R curve
fig, ax = plt.subplots(figsize=(8, 6))
_, p1, r1_rec, f1_1, *_ = zip(*r1)
_, p2, r2_rec, f2_, *_ = zip(*r2)

# Sort by recall for clean curve
order1 = np.argsort(r1_rec)
order2 = np.argsort(r2_rec)
ax.plot(np.array(r1_rec)[order1], np.array(p1)[order1], "o-",
        label="Row 1 — HOG-coarse, PCA(1215)", color="C0", alpha=0.85)
ax.plot(np.array(r2_rec)[order2], np.array(p2)[order2], "s-",
        label="Row 2 — full features, PCA(4677)", color="C1", alpha=0.85)

# Mark t=1.5 operating points on each
op1 = at_threshold(r1, 1.5)
op2 = at_threshold(r2, 1.5)
if op1:
    ax.plot(op1[2], op1[1], "P", color="C0", markersize=14, markeredgecolor="black", markeredgewidth=1.5,
            label=f"Row 1 @ t=1.5  (P={op1[1]:.3f}, R={op1[2]:.3f})")
if op2:
    ax.plot(op2[2], op2[1], "P", color="C1", markersize=14, markeredgecolor="black", markeredgewidth=1.5,
            label=f"Row 2 @ t=1.5  (P={op2[1]:.3f}, R={op2[2]:.3f})")

# Best F1 markers
b1 = best_f1(r1)
b2 = best_f1(r2)
ax.plot(b1[2], b1[1], "*", color="C0", markersize=18, markeredgecolor="black", markeredgewidth=1.2,
        label=f"Row 1 best F1={b1[3]:.3f} @ t={b1[0]:.2f}")
ax.plot(b2[2], b2[1], "*", color="C1", markersize=18, markeredgecolor="black", markeredgewidth=1.2,
        label=f"Row 2 best F1={b2[3]:.3f} @ t={b2[0]:.2f}")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Detection P-R curves (validate, IoU ≥ 0.5)\nRow 1 vs Row 2 — same single scale, same NMS")
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(0.05, max(max(r1_rec), max(r2_rec)) * 1.05))
ax.set_ylim(0, max(0.05, max(max(p1), max(p2)) * 1.05))
plt.tight_layout()

OUT = f"{REPO}/2026/pr_curve_row1_vs_row2.png"
plt.savefig(OUT, dpi=130)
print(f"\nSaved P-R curve to {OUT}")

print("\n=== Row 1 sweep ===")
print("t        P       R       F1     TP    FP    FN")
for t, p, r, f1, tp, fp, fn in r1:
    print(f"{t:6.3f}  {p:.3f}  {r:.3f}  {f1:.3f}  {tp:4d}  {fp:4d}  {fn:4d}")

print("\n=== Row 2 sweep ===")
print("t        P       R       F1     TP    FP    FN")
for t, p, r, f1, tp, fp, fn in r2:
    print(f"{t:6.3f}  {p:.3f}  {r:.3f}  {f1:.3f}  {tp:4d}  {fp:4d}  {fn:4d}")

print("\n=== Summary ===")
print(f"Row 1 best F1 = {b1[3]:.3f} at t={b1[0]:.2f}  (P={b1[1]:.3f}, R={b1[2]:.3f})")
print(f"Row 2 best F1 = {b2[3]:.3f} at t={b2[0]:.2f}  (P={b2[1]:.3f}, R={b2[2]:.3f})")
if op1 and op2:
    print(f"Row 1 @ t=1.5: P={op1[1]:.3f}, R={op1[2]:.3f}, F1={op1[3]:.3f}")
    print(f"Row 2 @ t=1.5: P={op2[1]:.3f}, R={op2[2]:.3f}, F1={op2[3]:.3f}")
