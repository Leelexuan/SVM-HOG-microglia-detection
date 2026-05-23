"""
One-off PCA variance curve via the Gram trick.

Run once per ``feature_mode`` to find the integer ``n_components`` that retains
a desired variance fraction. Paste the resulting int into the matching
notebook's ``Config.pca_n_components``.

The Gram trick: for X centered (n_samples, n_features), the non-zero
eigenvalues of X·X^T equal the eigenvalues of X^T·X (and equal PCA's explained
variances). Compute the smaller of the two — quadratic in ``min(n_samples,
n_features)`` instead of running a full SVD on the original matrix.

Usage (from inside 2026/):

    /home/ec2-user/lx-svm-venv/bin/python compute_variance_curve.py \\
        --feature-mode hog_coarse \\
        --artifact-dir ./microglia-artifacts-row1

Output:
    - printed crossover table for 50/80/85/90/95/99% variance targets
    - PNG plot at <artifact-dir>/pca_variance_curve_<feature-mode>.png

Memory:
    - hog_coarse (~5292-d, 14914 samples): X^T·X is 5292×5292 → ~110 MB peak
    - full (~32801-d, 14914 samples):       X·X^T is 14914×14914 → ~1.8 GB peak
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless, no GUI required
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import pipeline as pl
from pipeline import Config, extract_raw_features


def gram_variance_curve(X: np.ndarray) -> np.ndarray:
    """Return PCA-explained eigenvalues in descending order via the Gram trick."""
    n_samples, n_features = X.shape
    print(f"X shape: ({n_samples}, {n_features})  dtype={X.dtype}")
    Xf = X.astype(np.float32, copy=False)

    if n_samples <= n_features:
        print(f"  Computing X @ X.T ({n_samples}×{n_samples} float32)...")
        K = Xf @ Xf.T
    else:
        print(f"  Computing X.T @ X ({n_features}×{n_features} float32)...")
        K = Xf.T @ Xf

    print(f"  Gram dtype={K.dtype}, shape={K.shape}, "
          f"size={K.nbytes / 1e6:.0f} MB")

    print("  Upcasting Gram to float64 for eigvalsh...")
    K = K.astype(np.float64)

    print("  Running np.linalg.eigvalsh...")
    t0 = time.time()
    eigvals = np.linalg.eigvalsh(K)
    print(f"    eigvalsh done in {time.time() - t0:.1f}s")

    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals[eigvals > 0]   # discard numerical zeros / negatives
    return eigvals


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-mode", choices=["hog_coarse", "full"], required=True)
    parser.add_argument("--artifact-dir", default=None,
                        help="Where to read/write features_cache.npz and the PNG. "
                             "Defaults to ./microglia-artifacts-variance-<mode>/.")
    parser.add_argument("--out-png", default=None,
                        help="Override the output PNG path.")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or f"./microglia-artifacts-variance-{args.feature_mode}"
    os.makedirs(artifact_dir, exist_ok=True)

    cfg = Config(artifact_dir=artifact_dir, feature_mode=args.feature_mode)
    print(f"Config: feature_mode={cfg.feature_mode!r}, artifact_dir={cfg.artifact_dir!r}")

    # ── Load or extract raw features ──────────────────────────────────────
    if os.path.exists(cfg.features_cache):
        print(f"\nLoading cached features from {cfg.features_cache}")
        cache = np.load(cfg.features_cache)
        X_train_raw = cache["X_train_raw"]
        print(f"  X_train_raw shape: {X_train_raw.shape}, dtype: {X_train_raw.dtype}")
    else:
        print(f"\nNo cache at {cfg.features_cache} — extracting features (slow).")
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = extract_raw_features(cfg)
        np.savez_compressed(
            cfg.features_cache,
            X_train_raw=X_train_raw, X_val_raw=X_val_raw,
            y_train_raw=y_train_raw, y_val_raw=y_val_raw,
        )
        print(f"  Cached features → {cfg.features_cache}")

    # ── Standardise then compute the curve ────────────────────────────────
    print("\nStandardising with StandardScaler (matches fit_pipeline)...")
    t0 = time.time()
    scaler = StandardScaler().fit(X_train_raw)
    X      = scaler.transform(X_train_raw)
    print(f"  scaler.fit_transform done in {time.time() - t0:.1f}s")

    print("\nComputing PCA variance curve via the Gram trick:")
    eigvals = gram_variance_curve(X)
    cumvar  = np.cumsum(eigvals) / eigvals.sum()

    # ── Crossover table ───────────────────────────────────────────────────
    print(f"\nVariance crossovers ({len(eigvals)} non-zero eigenvalues):")
    print(f"  {'target':>8s}   {'n_components':>12s}")
    crossovers = {}
    for target in [0.50, 0.80, 0.85, 0.90, 0.95, 0.99]:
        n = int(np.searchsorted(cumvar, target) + 1)
        crossovers[target] = n
        marker = "  ← chosen for ablation" if target == 0.90 else ""
        print(f"  {target * 100:>7.0f}%   {n:>12d}{marker}")

    # ── Plot ──────────────────────────────────────────────────────────────
    out_png = args.out_png or os.path.join(
        artifact_dir, f"pca_variance_curve_{args.feature_mode}.png"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=1.5)
    for target, color in [(0.80, "tab:blue"), (0.90, "tab:green"), (0.95, "tab:orange")]:
        n = crossovers[target]
        plt.axhline(target, color=color, linestyle="--", alpha=0.5,
                    label=f"{target * 100:.0f}% → n={n}")
        plt.axvline(n, color=color, linestyle=":", alpha=0.3)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(
        f"PCA variance curve — feature_mode={args.feature_mode!r}, "
        f"X={X.shape}"
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"\nPlot saved → {out_png}")
    print(f"\n→ Paste into Config: pca_n_components = {crossovers[0.90]}   # 90% variance")


if __name__ == "__main__":
    sys.exit(main())
