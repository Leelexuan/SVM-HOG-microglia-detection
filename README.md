# microglia-detection-project

## Model Overview

The aim of this model is to detect microglia cells from lab images with a resolution of 2048x2048 pixels. The model utilizes machine learning techniques to accurately identify and differentiate microglia cells from noise in the images. The following sections describe the detailed steps involved in data preprocessing, feature extraction, training, and prediction.

## Data Preprocessing

### Labeling and Cropping

1. **Labeling Microglia Cells**: Microglia cells were manually labeled using the MakeSense.AI tool. This step involved identifying and marking the cells in the large 2048x2048 images.
2. **Image Cropping**: Labeled microglia cells were cropped out of the original images to create smaller images of size 64x64 pixels. This step ensures that each smaller image contains a single microglia cell for better focus and accuracy during feature extraction.
3. **Creating Noise Data**: A mask was applied to the source images to remove the labeled microglia cells. The remaining parts of the images, which contain no microglia cells, were split into 65x65 pixel images and labeled as noise. This step is crucial for training the model to distinguish between microglia cells and noise.

## Features Extracted

### Histogram of Oriented Gradients (HOG)
- **HOG Features**: HOG features were extracted from both microglia and noise images. This method captures the shape and structure of the cells by computing gradients and orientations, making it effective for object detection.

### Color Features
- **Image Color Bins**: Color bins were used to categorize the pixel values of the images. This method helps in capturing the color distribution within the microglia cells and noise images.
- **Color Histogram**: A color histogram was generated for each image to represent the distribution of colors. This feature aids in distinguishing microglia cells based on their color characteristics.

## Libraries Used

The following libraries were used to implement the model:
- **OpenCV**: Utilized for feature extraction, reading images, and image pre-processing tasks.
- **Scikit-learn**: Employed for implementing the Support Vector Machine (SVM) classifier.
- **Scikit-image**: Used for extracting HOG features.
- **Matplotlib**: Utilized for visualizing images and results.
- **Numpy**: Used for numerical computations and handling image data arrays.

## Training Process

### Feature Extraction
- **Microglia and Noise Images**: Features were extracted from both microglia (64x64) and noise (65x65) images using the methods described above.
- **Data Segregation**: The dataset consisted of 13 images for training. After feature extraction, the model automatically segregated the data into training and testing sets to evaluate performance.

### Model Training
- **Support Vector Machine (SVM)**: An SVM classifier was trained using the extracted features. The classifier was trained to differentiate between microglia cells and noise images.

## Prediction Process

### Sliding Window Technique
- **Sliding Windows**: During prediction, an 8x8 pixel sliding window was applied over the entire 2048x2048 image to scan for potential microglia cells.
- **Heatmap Generation**: Each time a region was recognized as containing a microglia cell, the corresponding location in a heatmap was incremented by 1.
- **Threshold Application**: A dynamic threshold was applied to the heatmap to filter out false positives. Regions with values below the threshold were set to 0.
- **Bounding Boxes**: Finally, boxes were drawn around the detected regions to indicate the presence of microglia cells.

### Data for Prediction
- **Prediction Images**: The model was tested on 2 images to validate its performance in detecting microglia cells.

## Improvements Exploration

Here are some ways that are being explored to improve the precision of the model:
- **Ensemble of Window Sizes**: Since some microglia cells were larger than the original window size of 64x64, an ensemble model of various window sizes (50, 80, 100) with different weights is being explored 
- **Data Permutation**: Due to the small size of data (only 15 2048x2048 images), the data size was increased by rotating the images 90, 180 and 270 degrees respectively 
- **Hard Negative Mining**: Since the background of lab images are quite similar, hard negative mining can be used to do targeted improvements to the dataset instead of adding more similar images. This may improve efficiency too

## Credits:

Base Code:
https://github.com/neerajd12/object-detection-with-svm-and-opencv?tab=readme-ov-file





