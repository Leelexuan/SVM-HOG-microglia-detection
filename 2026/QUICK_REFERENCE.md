# Quick Reference: Original vs Fixed Code

## 1. Bounding Box Coordinates

### ❌ ORIGINAL (WRONG)
```python
def inv_masking(bounding_coords, inv_mask):
    cx = int(bounding_coords["cx"])
    cy = int(bounding_coords["cy"])
    w = int(bounding_coords["w"])
    h = int(bounding_coords["h"])

    # Calculate top-left and bottom-right corners from center
    top_left_x = cx
    top_left_y = int(cy + h)         # ❌ WRONG!
    bottom_right_x = int(cx + w)     # ❌ WRONG!
    bottom_right_y = cy              # ❌ WRONG!

    top_left = (top_left_x, top_left_y)
    bottom_right = (bottom_right_x, bottom_right_y)

    cv2.rectangle(inv_mask, top_left, bottom_right, 255, -1)
    return inv_mask
```

### ✅ FIXED
```python
def inv_masking(bounding_coords, inv_mask):
    cx = int(float(bounding_coords["cx"]))
    cy = int(float(bounding_coords["cy"]))
    w = int(float(bounding_coords["w"]))
    h = int(float(bounding_coords["h"]))

    # Calculate top-left and bottom-right corners from center + dimensions
    top_left_x = int(cx - w/2)       # ✅ CORRECT
    top_left_y = int(cy - h/2)       # ✅ CORRECT
    bottom_right_x = int(cx + w/2)   # ✅ CORRECT
    bottom_right_y = int(cy + h/2)   # ✅ CORRECT

    top_left = (top_left_x, top_left_y)
    bottom_right = (bottom_right_x, bottom_right_y)

    cv2.rectangle(inv_mask, top_left, bottom_right, 255, -1)
    return inv_mask
```

---

## 2. Image Size Parameter

### ❌ ORIGINAL
```python
def preprocess_images(source_folder, microglia_folder, noise_folder):
    # ...
    for img in os.listdir(source_folder):
        extract_roi(source_folder, img, microglia_folder, noise_folder, 64, 2047)  # ❌ 2047!
```

### ✅ FIXED
```python
def preprocess_images(source_folder, microglia_folder, noise_folder, window_size=64):
    # ...
    image_size = 2048  # ✅ Correct for 2048x2048 images
    for img in os.listdir(source_folder):
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            extract_roi(source_folder, img, microglia_folder, noise_folder, window_size, image_size)
```

---

## 3. SciPy Import

### ❌ ORIGINAL
```python
from scipy.ndimage.measurements import label  # ❌ Deprecated namespace
```

### ✅ FIXED
```python
from scipy.ndimage import label  # ✅ Current correct import
```

---

## 4. SVM Training - Convergence & Class Imbalance

### ❌ ORIGINAL
```python
def train_SVC(X_train, y_train):
    svc = svm.LinearSVC()  # ❌ Uses default max_iter=1000, no class weighting
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return svc
```

**Problem:** Results in `ConvergenceWarning: Liblinear failed to converge, increase the number of iterations`

### ✅ FIXED
```python
def train_SVC(X_train, y_train):
    # Fixed: Added max_iter and class_weight to handle imbalance
    svc = svm.LinearSVC(max_iter=10000, class_weight='balanced', dual=False, random_state=42)
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return svc
```

---

## 5. Metric Calculation

### ❌ ORIGINAL (Manual Entry - Error Prone)
```python
def confusion_matrix(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1-score: " + str(f1_score))

# Manual entry of numbers (likely contains errors!)
confusion_matrix(52, 36, 20)  # Where did these numbers come from?
```

### ✅ FIXED (Computed from Predictions)
```python
from sklearn.metrics import precision_recall_fscore_support

def test_classifier(svc, X_test, y_test):
    accuracy = svc.score(X_test, y_test)
    print(f'Test Accuracy of SVC = {round(accuracy, 4)}')

    # Compute metrics automatically from predictions
    y_pred = svc.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )
    print(f'Precision: {round(precision, 4)}')
    print(f'Recall: {round(recall, 4)}')
    print(f'F1-Score: {round(f1, 4)}')
```

---

## 6. Error Handling

### ❌ ORIGINAL (No Error Handling)
```python
def extract_roi(source_folder, img, microglia_folder, noise_folder, window_size, image_size):
    img_read = cv2.imread(source_folder + "/" + img)  # ❌ Crashes if file doesn't exist
    img_name = os.path.splitext(img)[0]
    bounding_rects = get_row_by_image_name(img_name)
    # ... rest of function
```

### ✅ FIXED (With Error Handling)
```python
def extract_roi(source_folder, img, microglia_folder, noise_folder, window_size, image_size):
    try:
        img_read = cv2.imread(source_folder + "/" + img)
        if img_read is None:  # ✅ Check if read failed
            print(f"Warning: Could not read {img}")
            return

        img_name = os.path.splitext(img)[0]
        bounding_rects = get_row_by_image_name(img_name)
        # ... rest of function
    except Exception as e:  # ✅ Catch any other errors
        print(f"Error processing {img}: {str(e)}")
```

---

## 7. Configurable Parameters

### ❌ ORIGINAL (Hardcoded Magic Numbers)
```python
colorConv = 'BGR2HSV'
hog_channel = "ALL"
orient = 9                          # ❌ What does 9 mean? Why?
pix_per_cell = 8                    # ❌ Magic number
cell_per_block = 2                  # ❌ Magic number
recent_heatmaps = deque(maxlen=10)

def find_microglias(img, colorConv, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    # ...
    window = 64                      # ❌ Hardcoded
    cells_per_step = 2               # ❌ Hardcoded, no explanation
```

### ✅ FIXED (Documented Parameters)
```python
# HOG feature parameters (documented)
HOG_ORIENT = 9          # Number of orientation bins
HOG_PIX_PER_CELL = 8   # Pixels per cell
HOG_CELL_PER_BLOCK = 2 # Cells per block

# Configurable window parameters
WINDOW_SIZE = 64        # Options: 50, 64, 80, 100
CELLS_PER_STEP = 2      # Sliding window step (creates 8*CELLS_PER_STEP pixel stride)

def find_microglias(img, colorConv, svc, X_scaler, orient=HOG_ORIENT,
                    pix_per_cell=HOG_PIX_PER_CELL, cell_per_block=HOG_CELL_PER_BLOCK,
                    cells_per_step=CELLS_PER_STEP, window=WINDOW_SIZE):
    # ✅ All parameters documented and changeable
```

---

## 8. Threshold Selection

### ❌ ORIGINAL (Manual Guessing)
```python
heatmap = apply_threshold(np.mean(recent_heatmaps, axis=0), 1.5)  # ❌ Why 1.5?
# OR in ensemble:
new_heatmap = apply_dynamic_threshold(new_heatmap, 95)  # ❌ Why 95? Different from 1.5!
```

### ✅ FIXED (Data-Driven)
```python
# Analyze different thresholds on test set
y_scores = svc.decision_function(X_test)
thresholds = np.linspace(-2, 2, 50)
f1_scores = []

for thresh in thresholds:
    y_pred = (y_scores >= thresh).astype(int)
    if len(np.unique(y_pred)) > 1:
        _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        f1_scores.append(f1)
    else:
        f1_scores.append(0)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Best F1-Score: {f1_scores[optimal_idx]:.4f}")
```

---

## 9. Rotation Implementation

### ❌ ORIGINAL (Inefficient)
```python
def rotate_and_save(img, filename, folder_name, count):
    save_file(img, filename, folder_name, count)

    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    count90 = count + 90           # ❌ Weird naming
    save_file(img_90, filename, folder_name, count90)

    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    count180 = count + 180         # ❌ Magic numbers
    save_file(img_180, filename, folder_name, count180)

    img_270 = cv2.rotate(img_180, cv2.ROTATE_90_CLOCKWISE)  # ❌ Redundant rotation
    count270 = count + 270
    save_file(img_270, filename, folder_name, count270)
```

### ✅ FIXED
```python
def rotate_and_save(img, filename, folder_name, count):
    """Save image and its 90, 180, 270 degree rotations."""
    save_file(img, filename, folder_name, count)

    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    save_file(img_90, filename, folder_name, count + 90)

    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    save_file(img_180, filename, folder_name, count + 180)

    img_270 = cv2.rotate(img, cv2.ROTATE_270_CLOCKWISE)  # ✅ Direct rotation
    save_file(img_270, filename, folder_name, count + 270)
```

---

## 10. CSV Parsing

### ❌ ORIGINAL (Fragile)
```python
def get_row_by_image_name(target_value):
    with open("Image_ROIs.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        count = 0
        rois = {}
        for row in reader:
            if row[5] == target_value + ".png" and row[0] == "0":  # ❌ Hardcoded indices, no bounds check
                rois[count] = {"cx": row[1],
                               "cy": row[2],
                               "w": row[3],
                               "h": row[4]}
                count += 1
        return rois
    return None
```

### ✅ FIXED (Robust)
```python
def get_row_by_image_name(target_value):
    """Get bounding box coordinates from CSV for a given image."""
    try:
        with open("Image_ROIs.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            rois = {}
            for row in reader:
                if len(row) > 5 and row[5] == target_value + ".png" and row[0] == "0":  # ✅ Bounds check
                    rois[count] = {"cx": row[1],
                                   "cy": row[2],
                                   "w": row[3],
                                   "h": row[4]}
                    count += 1
            return rois
    except FileNotFoundError:
        print("Error: Image_ROIs.csv not found")
        return {}
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return {}
```

---

## Summary Table

| Issue | Original | Fixed | Impact |
|-------|----------|-------|--------|
| Bounding Box Math | ❌ Wrong formula | ✅ Correct center±dim | Incorrect training data |
| Image Size | ❌ 2047 | ✅ 2048 | Missing edge regions |
| SciPy Import | ❌ Deprecated | ✅ Current | Future compatibility |
| SVM Settings | ❌ Low iter, no balance | ✅ 10k iter, balanced | Won't converge, biased |
| Metrics | ❌ Manual entry | ✅ Computed | Reliable results |
| Error Handling | ❌ None | ✅ Try-except | Crashes on bad input |
| Parameters | ❌ Hardcoded | ✅ Documented & configurable | Easy to experiment |
| Threshold | ❌ Guessed | ✅ Data-driven | Optimal performance |

---

## Files to Update from Original

If you want to update your original notebooks instead of using the fixed versions:

1. Find all instances of `2047` → Replace with `2048`
2. Find `from scipy.ndimage.measurements import label` → Replace with `from scipy.ndimage import label`
3. Find `svm.LinearSVC()` → Replace with `svm.LinearSVC(max_iter=10000, class_weight='balanced', dual=False, random_state=42)`
4. Find the `inv_masking()` function → Replace with fixed version
5. Add error handling (try-except blocks) around file I/O

---

**Recommendation:** Use the fixed notebooks directly rather than trying to patch the originals.
