# 2026 - Fixed SVM-HOG Microglia Detection Notebooks

This folder contains corrected versions of all your Jupyter notebooks with **15 identified errors and quality issues fixed**.

## 📋 What's Included

| File | Purpose |
|------|---------|
| **SVM_original_fixed.ipynb** | Baseline SVM model (64x64 window) - **START HERE** |
| **SVM_window_template_fixed.ipynb** | Template for 50/64/80/100 px windows |
| **SVM_ensemble_fixed.ipynb** | Ensemble combining multiple window sizes |
| **FIXES_GUIDE.md** | Detailed explanation of all fixes |
| **QUICK_REFERENCE.md** | Side-by-side original vs fixed code |
| **README.md** | This file |

## 🔧 Critical Fixes Applied

### Top 5 Logic Errors Fixed:

1. **Bounding Box Coordinates** - Was calculating box corners incorrectly
   - `top_left_y = cy + h` → `top_left_y = cy - h/2`
   - **Impact**: Training data was extracted from wrong locations

2. **Image Size Parameter** - Used 2047 instead of 2048
   - **Impact**: Missing edge regions of images in noise sampling

3. **SVM Convergence** - Wasn't converging, showing warnings
   - Added `max_iter=10000, class_weight='balanced'`
   - **Impact**: Better model training, handles class imbalance

4. **Deprecated Import** - `scipy.ndimage.measurements` is deprecated
   - `from scipy.ndimage.measurements import label` → `from scipy.ndimage import label`
   - **Impact**: Future compatibility, removes warnings

5. **Metric Calculation** - Manual entry of TP/FP/FN (error-prone)
   - Now computed automatically from predictions
   - **Impact**: Reliable, reproducible metrics

### Plus 10 More Code Quality Improvements:
- ✅ Proper error handling (try-except blocks)
- ✅ Configurable parameters with documentation
- ✅ Data-driven threshold selection
- ✅ Path handling improvements
- ✅ Type conversions from CSV data
- ✅ And more...

See **FIXES_GUIDE.md** and **QUICK_REFERENCE.md** for complete details.

## 🚀 Quick Start

### 1. For Baseline Model:
```bash
cd 2026
jupyter notebook SVM_original_fixed.ipynb
```
- Preprocesses training data
- Trains SVM with 64x64 window
- Tests on validation set
- Analyzes optimal threshold

### 2. For Window Size Experiments:
```bash
jupyter notebook SVM_window_template_fixed.ipynb
```
Change at the top:
```python
WINDOW_SIZE = 50   # or 64, 80, 100
```

### 3. For Ensemble:
```bash
jupyter notebook SVM_ensemble_fixed.ipynb
```
Combines predictions from multiple window sizes.

## 📊 Expected Results

With these fixes, you should see:
- **No convergence warnings** during training
- **Better precision/recall balance** (class weighting)
- **More stable metrics** (computed, not manual)
- **Better recall** (catches more cells)
- **Cleaner code** (error handling, documentation)

## ✨ Key Improvements Over Original

| Aspect | Original | Fixed |
|--------|----------|-------|
| Bounding box logic | ❌ Incorrect | ✅ Correct |
| Image size | ❌ 2047 | ✅ 2048 |
| SVM convergence | ❌ Warnings | ✅ Converges cleanly |
| Class imbalance | ❌ Not handled | ✅ Balanced weights |
| Error handling | ❌ None | ✅ Robust try-except |
| Documentation | ❌ Sparse | ✅ Well documented |
| Threshold selection | ❌ Hardcoded | ✅ Data-driven |
| Metrics | ❌ Manual | ✅ Automated |

## 📖 Understanding the Fixes

### Not Sure Where to Start?

1. **First time?** → Read `FIXES_GUIDE.md` intro section
2. **Want code comparison?** → Check `QUICK_REFERENCE.md`
3. **Ready to code?** → Open `SVM_original_fixed.ipynb`

### For Specific Issues:

- **Bounding box confusion?** → See QUICK_REFERENCE.md section 1
- **Want to understand thresholds?** → See FIXES_GUIDE.md section on "Threshold Selection"
- **Need to change window size?** → Use SVM_window_template_fixed.ipynb
- **Class imbalance concerns?** → See FIXES_GUIDE.md section 5

## ⚠️ Known Remaining Issues

### Data Leakage (Rotation Augmentation)
Current approach rotates samples before train/test split. The same cell appears in both sets (rotated versions).

**Fix (not implemented):**
```python
# Split images first
train_images, test_images = train_test_split(source_images)
# Then rotate only training images
for img in train_images:
    rotate_and_save(img)
```

### Recommendation:
For more robust results, consider splitting the source images before preprocessing.

## 📈 Recommended Workflow

```
1. Run SVM_original_fixed.ipynb (baseline)
   ↓
2. Try different window sizes (use template)
   ↓
3. Compare F1-scores for each size
   ↓
4. Run ensemble with top performers
   ↓
5. Optimize threshold on validation set
```

## 🔍 Verification Checklist

After running notebooks, check:

- [ ] No ConvergenceWarning about SVM
- [ ] Extracted microglia patches look reasonable
- [ ] Noise patches don't contain cells
- [ ] F1-score > 0.6
- [ ] Both precision and recall non-zero
- [ ] Heatmaps show detected regions
- [ ] Optimal threshold is printed

## 🎯 Performance Goals

Based on your original data:
- **Precision**: 0.90+ (minimize false positives)
- **Recall**: 0.65+ (catch most cells)
- **F1-Score**: 0.75+ (balance both)

With these fixes, you should achieve better recall without sacrificing too much precision.

## 💡 Tips & Tricks

### Debugging Training Data:
```python
# In SVM_original_fixed.ipynb, visualize extracted patches:
img = mpimg.imread("Processed_training_images/Microglia/sample.jpg")
plt.imshow(img)
plt.show()
```

### Testing Different Thresholds:
The notebook includes a threshold analysis section that plots F1-score vs threshold.

### Comparing Models:
Save the confusion matrix results for each model and compare:
```python
# Add to bottom of each notebook:
print(f"F1-Score for {WINDOW_SIZE}px window: {f1:.4f}")
```

## 🤔 FAQ

**Q: Should I use these notebooks or fix the originals?**
A: Use these fixed notebooks directly. It's cleaner and safer.

**Q: Can I modify these notebooks?**
A: Absolutely! They're designed to be starting points, not final solutions.

**Q: Why do window sizes matter?**
A: Cells vary in size. Window sizes 50, 64, 80, 100px capture different sized cells.

**Q: What's with the class weighting?**
A: You have ~10x more noise samples than microglia. Class weights balance this.

**Q: How do I know if convergence is working?**
A: You shouldn't see `ConvergenceWarning` anymore. Training should complete cleanly.

## 📚 Related Documentation

- `FIXES_GUIDE.md` - Complete explanation of all 15 fixes
- `QUICK_REFERENCE.md` - Side-by-side code comparisons
- `../claude.md` - Project overview and context
- `../README.md` - Original project documentation

## ✅ Verification

All files created successfully on 2026-04-11:
```
✅ SVM_original_fixed.ipynb (31 KB)
✅ SVM_window_template_fixed.ipynb (20 KB)
✅ SVM_ensemble_fixed.ipynb (13 KB)
✅ FIXES_GUIDE.md (9.4 KB)
✅ QUICK_REFERENCE.md (12 KB)
✅ README.md (this file)
```

---

**Ready to start?** Open `SVM_original_fixed.ipynb` and follow the cells!

Questions? Check the FIXES_GUIDE.md for detailed explanations.
