[[Claude Context]] | [[Daily Notes/2026-05-06]]

# Experiment Story — SVM-HOG Microglia Detection

> Use this note to map every experiment before touching slides.
> Step 1: fill in all experiment entries. Step 2: the 4 beats are already identified — just fill the gaps. Step 3: use the slide map at the bottom to build the deck.

---

## Full Experiment Map

*For each experiment: what you thought would happen, what actually happened, what you learned.*

---

### 2023 — Initial Work

#### SVM_original.ipynb
- **Hypothesis:** HOG + colour features on 64×64 crops can distinguish microglia from background
- **What I did:** manually labelled cells with MakeSense.AI, extracted 64×64 positive crops, sampled negative crops from background regions
- **What happened:** _[fill in — patch F1, any qualitative observations]_
- **Insight:** _[fill in]_

---

#### Reverse Masking for Negatives
- **Hypothesis:** sampling negatives from regions outside labelled cells gives clean background examples
- **What I did:** _[fill in — how the mask was applied, how negatives were sampled]_
- **What happened:** _[fill in]_
- **Insight:** _[fill in — did this produce good negatives? any edge cases?]_

---

#### Rotation Augmentation (SVM_rotated_data.ipynb)
- **Hypothesis:** rotating training images 90°/180°/270° will improve generalisation since microglia have no fixed orientation
- **What I did:** applied rotation to all 13 images before train/test split, tripling dataset size
- **What happened:** _[fill in — did F1 improve?]_
- **Insight:** ⚠️ Data leakage — rotated versions of the same cell appeared in both train and test. The improvement was partly artificial.

---

#### Window Size Variants (SVM_window 50, 64, 80, 100)
- **Hypothesis:** not all microglia fit in a 64×64 crop — different window sizes may catch different cells
- **What I did:** _[fill in — trained separate models at each size, same feature pipeline]_
- **What happened:** _[fill in — which size performed best? any tradeoffs?]_
- **Insight:** _[fill in]_

---

#### SVM_ensemble.ipynb
- **Hypothesis:** combining predictions from multiple window sizes via weighted voting should outperform any single size
- **What I did:** _[fill in — how voting was weighted, how detections were merged]_
- **What happened:** _[fill in]_
- **Insight:** _[fill in — did ensemble help? was it worth the complexity?]_

---

#### Hard Negative Mining (Hard Negative Mining/)
- **Hypothesis:** the classifier confuses specific background textures for cells — adding those as negatives should fix it
- **What I did:** ran detection pipeline on training images, collected false positive crops, added to negative training set, retrained
- **What happened:** _[fill in]_
- **Insight:** _[fill in — how many false positives were reduced?]_

---

### 2026 — Refactor + Fixes

#### Realising Patch Metrics ≠ Detection Quality
- **What triggered it:** running inference on a full 2048×2048 image and seeing hundreds of false positive boxes despite ~0.92 patch F1
- **Root cause:** patch evaluation uses a balanced 50/50 split. Full-image inference sweeps ~15,000 windows per scale with only 10–20 real cells — extreme class imbalance the model never saw during training
- **What changed:** _[fill in — when did you switch to detection-level metrics?]_

---

#### SVM_original_refactored.ipynb — Key Changes
- **Data leakage fix:** split source images before applying rotation augmentation
  - Before: _[fill in patch F1 with leaky split]_
  - After: _[fill in patch F1 with clean split]_ ← the honest number
- **PCA added:** reduced ~32,801 features → ~4,997 components (95% variance)
  - Effect on training time: _[fill in]_
  - Effect on accuracy: _[fill in]_
- **Detection threshold retuned:** swept `cfg.detection_threshold` against bounding box precision/recall at IoU ≥ 0.5 on full images
  - Old threshold (tuned on patch F1): _[fill in]_ → detection precision: _[fill in]_
  - New threshold (tuned on detection output): _[fill in]_ → detection precision: _[fill in]_
- **Multi-scale pyramid added:** scales 0.75×, 1.0×, 1.5× with cross-scale NMS
  - Effect: _[fill in]_

---

### 2026 — Modern Baseline

#### YOLOv8 Fine-tuned Comparison
- **Hypothesis:** a modern detector fine-tuned on the same labels should outperform a classical pipeline — or reveal where classical still wins at small data scale
- **What I did:** _[fill in after running — export labels to YOLO format, fine-tune YOLOv8-nano on Colab]_
- **What happened:** _[fill in — detection precision/recall, inference time]_
- **Insight:** _[fill in — did YOLOv8 win? where did SVM+HOG hold up?]_

---

## 4 Presentation Beats

*Compress the full map above into these beats for the slides. Fill in the bracketed fields.*

---

### Beat 1 — "It works… sort of" *(2023 — ~90 seconds)*

**The setup:** Manual microglia counting is slow and varies between analysts. Goal: automate detection on 2048×2048 microscopy images.

**What I built:**
- HOG + colour features on 64×64 crops
- Reverse masking to generate clean negative samples
- Rotation augmentation to handle orientation variance
- Patch F1: _[fill in]_ ← looks impressive

**Key visual:** labelled image from MakeSense.AI showing positive crops overlaid on a raw microscopy image

---

### Beat 2 — "It doesn't actually work" *(2023–2024 — ~60 seconds)*

**The moment:** ran the pipeline on a full image — _[fill in: how many false positives?]_ bounding boxes on an image with ~_[fill in]_ real cells.

**Why:** 15,000 windows per scale, only 10–20 positives. The model never saw this imbalance during training. Patch F1 of 0.92 is meaningless here.

**Key visual:** side-by-side — ground truth boxes vs. pipeline output (the "messy" result)

---

### Beat 3 — "Here's what was actually wrong" *(2026 — ~90 seconds)*

Three fixable engineering errors:

| Error | Impact | Fix |
|---|---|---|
| Data leakage in augmentation | F1 was inflated — rotated versions of same cell in train and test | Split source images first, then augment training only |
| Wrong threshold signal | Threshold tuned on patch F1, not detection output | Sweep threshold against IoU-matched bounding box precision/recall |
| No dimensionality reduction | _[fill in — overfitting? slow training?]_ | PCA to 95% variance — ~32,801 → ~4,997 features |

**Key visual:** before/after detection output — same image, old threshold vs. new threshold

---

### Beat 4 — "Fixed + where it stands today" *(2026 — ~90 seconds)*

**Results after fixes:**
- Detection precision: _[fill in]_ / Recall: _[fill in]_ at IoU ≥ 0.5
- Compared to YOLOv8 fine-tuned on same data: _[fill in]_

**The comparison insight:** _[fill in — e.g. "SVM+HOG still outperforms YOLOv8-nano at this data scale because X" or "YOLOv8 wins on precision but SVM is 3× faster at inference"]_

**What's next:** hard negative mining, more labelled images, CellViT as domain-specific baseline

**Key visual:** three-column comparison — raw image / SVM output / YOLOv8 output

---

## Slide Map

| Slide | Beat | Content | Target time |
|---|---|---|---|
| 1 | Intro | Problem — why microglia, why automation | 30s |
| 2 | Beat 1 | Dataset + initial approach (HOG, masking, augmentation) | 45s |
| 3 | Beat 1 | Patch F1 result — "looks good" | 15s |
| 4 | Beat 2 | Full-image inference — the false positive problem | 60s |
| 5 | Beat 3 | Three errors table + fixes | 60s |
| 6 | Beat 3 | Before/after detection images | 30s |
| 7 | Beat 4 | Final metrics + YOLOv8 comparison | 60s |
| 8 | Beat 4 | What's next + closing | 30s |
| | | **Total** | **~7 min** |
