[[Claude Context]] | [[Experiment Story]]

# Research Notes — SVM-HOG Sliding Window Detection

> Focus: how bounding boxes are defined during training and used during inference across the classical detection lineage.

---

## Core Theme

No box regression exists in classical sliding window pipelines. The bounding box is always the window rectangle — position + scale from the image pyramid. The field's progression was about making the window *score* more discriminative, not about predicting tighter boxes.

---

## Papers

---

### Dalal & Triggs (2005) — Histograms of Oriented Gradients for Human Detection
**CVPR 2005**

**Training:** Positive samples are 64×128 crops tightly aligned to ground-truth annotations. No box coordinates are learned — the SVM learns a binary window-level decision. Hard negative mining runs in two passes: random background crops first, then false positives from a first-pass SVM are added and the SVM is retrained.

**Inference:** Dense sliding window at 8px stride over a multi-scale image pyramid (scale factor ~1.05 per level). Each window position produces a scalar SVM decision score. Greedy NMS collapses overlapping high-score windows to the single best candidate.

**Key insight:** The output box *is* the window rectangle — no regression step, ever. Every subsequent classical paper inherits this assumption.

---

### Viola & Jones (2001) — Rapid Object Detection Using a Boosted Cascade of Simple Features
**CVPR 2001**

**Training:** Fixed 24×24 patches aligned to ground-truth annotations. AdaBoost builds a cascade of weak classifiers (Haar features). Each cascade stage is trained to maximize recall while discarding a fixed fraction of negatives; survivors feed the next stage's hard negative pool.

**Inference:** Sliding window at 1–2px stride; scale variation handled by downsampling the *image* (not resizing the window). Windows passing all cascade stages are flagged. Multiple overlapping detections are merged by spatial grouping and coordinate averaging — not score-based NMS.

**Key insight:** ~99% of windows are rejected after the first 1–2 cascade stages, giving ~15× speedup. The coordinate-averaging merge was later replaced by greedy IoU-NMS in HOG/SVM pipelines.

---

### Felzenszwalb, McAllester & Ramanan (2008) — A Discriminatively Trained, Multiscale, Deformable Part Model
**CVPR 2008**

**Training:** Only ground-truth bounding boxes required — no part annotations. A root HOG filter covers the annotated box; part filters are initialized by clustering HOG energy inside it, then refined via Latent SVM. Part locations within positive examples are latent variables updated iteratively alongside the SVM weights.

**Inference:** HOG pyramid at scale factor ~2^(1/10). Root and part filters convolved at their respective pyramid levels. Dynamic programming finds optimal part placement for each root position, yielding one scalar score per (position, scale). Greedy IoU-NMS (threshold ~0.5) suppresses lower-scoring overlapping boxes.

**Key insight:** Two-resolution HOG pyramid — part filters run at double the root resolution. Fine-grained part scoring without bloating root filter size. Gave a 2× improvement over PASCAL 2006 person detection.

---

### Felzenszwalb, Girshick, McAllester & Ramanan (2010) — Object Detection with Discriminatively Trained Part-Based Models
**TPAMI 2010**

**Training:** Same L-SVM framework as 2008, matured. Total score = `root_filter_score + Σ max(part_score_i − deformation_cost_i)`. Hard negative mining is fully integrated into the training loop. Part placements inside positive boxes are latent and re-inferred each L-SVM iteration.

**Inference:** Efficient dense score maps via filter convolution on the HOG pyramid. Deformation cost handled by distance transforms — O(n) not O(n²). Threshold → collect candidates → greedy NMS.

**Key insight:** Part positions are inferred at inference time by maximizing joint score within a spatial penalty. The bounding box still comes from the root filter window, but the score integrating deformable parts is far more discriminative than a rigid template.

---

### Neubeck & Van Gool (2006) — Efficient Non-Maximum Suppression
**ICPR 2006**

**Training:** N/A — purely a post-processing algorithm.

**Inference (the algorithm):**
1. Sort all candidate boxes by descending classifier score.
2. Greedily select top-scoring box M as a confirmed detection.
3. Suppress all remaining boxes with IoU(box, M) > threshold (typically 0.3–0.5).
4. Repeat on surviving boxes.

Also derives block-based variants that avoid O(n²) pairwise comparisons — critical when a 2048×2048 image at 8px stride produces ~65,000 candidate windows.

**Key insight:** This greedy IoU-NMS became the universal default for all sliding window detectors. Its central flaw — silently discarding true positives in crowded/overlapping scenes — motivated Soft-NMS (Bodla et al., 2017), which decays scores by a continuous IoU function instead of hard suppression.

---

## Summary Table

| Paper | Box Source | Box Regression? | NMS Strategy |
|---|---|---|---|
| Viola & Jones 2001 | Fixed 24×24 window position | No | Spatial group-merge + coord averaging |
| Dalal & Triggs 2005 | Fixed 64×128 window position | No | Greedy score-based NMS |
| Felzenszwalb et al. 2008 | Root filter position in HOG pyramid | No (parts deformable, root fixed) | Greedy IoU-NMS ~0.5 |
| Felzenszwalb et al. 2010 | Root filter position in HOG pyramid | No | Greedy IoU-NMS |
| Neubeck & Van Gool 2006 | N/A (post-processing only) | N/A | Greedy IoU-NMS (formalized) |

---

## Relevance to This Project

This pipeline sits squarely in the Dalal & Triggs tradition — fixed window, SVM score, heatmap thresholding, NMS. The multi-scale pyramid in `2026/SVM_original_refactored.ipynb` mirrors the DPM approach. The detection-level precision/recall at IoU ≥ 0.5 used in these papers is exactly the metric needed to fill the gaps in `Experiment Story.md`.
