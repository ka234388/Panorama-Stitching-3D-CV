# Panorama Stitching - Computer Vision Assignment 4

**Automated Image Stitching and Panorama Generation using Feature Matching and Homography Transformation**

## 📋 Assignment Overview

This assignment focuses on implementing a **panorama stitching application** that automatically combines multiple overlapping images into a single seamless panoramic image. The project demonstrates fundamental computer vision concepts including feature detection, feature matching, geometric transformations, and image blending techniques.

### Why This Assignment?

Panorama stitching is a classic and practical application of computer vision that involves:
- **Real-world relevance**: Used in smartphones (panorama mode), UAV/drone imaging, and surveying
- **Core CV concepts**: Combines multiple computer vision techniques (feature detection, registration, warping, blending)
- **Algorithm understanding**: Teaches geometric transformations, homography matrices, and robust estimation using RANSAC
- **Image processing**: Applies perspective transformations, image warping, and multi-image blending

---

## 🎯 Learning Objectives

By completing this assignment, you will:

1. **Detect and extract distinctive features** using Scale-Invariant Feature Transform (SIFT)
2. **Match corresponding features** between overlapping images using descriptor matching
3. **Estimate geometric transformations** (homography) using RANSAC for robust matching
4. **Compute perspective transformations** to warp images into a common coordinate system
5. **Blend overlapping regions** seamlessly to create final panorama
6. **Handle practical challenges**: varying lighting, distortions, and imperfect overlaps

---

## 🔬 Technical Approach

### Step 1: Feature Detection
- Extract SIFT keypoints and descriptors from each input image
- SIFT features are scale-invariant and rotation-invariant, making them ideal for matching across images with different scales and orientations
- Each keypoint has 128-dimensional descriptor capturing local image information

### Step 2: Feature Matching
- Use brute-force or FLANN (Fast Library for Approximate Nearest Neighbors) matching to find correspondences between features
- Apply **Lowe's ratio test**: ratio of nearest neighbor to second-nearest neighbor distance < 0.75
- This filtering removes ambiguous matches and keeps only high-confidence correspondences

### Step 3: Homography Estimation
- Compute homography matrix H using matched feature pairs
- Homography relates pixel coordinates: \(p_2 = H \cdot p_1\)
- Use **RANSAC (Random Sample Consensus)** for robust estimation:
  - Randomly sample minimum feature set (4 point pairs)
  - Compute homography and count inliers (reprojection error < threshold)
  - Repeat and keep homography with most inliers
  - This handles outliers in feature matches

### Step 4: Image Warping
- Apply perspective transformation using computed homography
- Warp all images into a common panorama coordinate system
- Use bilinear interpolation for smooth pixel values

### Step 5: Image Blending
- Handle overlapping regions where multiple images contribute to output
- Apply weighted blending or multi-band blending for seamless transitions
- Avoid harsh seams at image boundaries

---

## 🛠️ Requirements

### System Requirements
- Python 3.7+
- OpenCV with SIFT support (opencv-contrib-python)
- Sufficient RAM for image processing
- 2GB+ disk space for sample images and outputs

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies:
```
opencv-contrib-python>=4.5.0
numpy>=1.19.0
scipy>=1.5.0
scikit-image>=0.17.0
Pillow>=8.0.0
matplotlib>=3.3.0
tqdm
imutils
```

---

## 🚀 How to Setup and Run the Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/ka234388/Panorama.git
cd Panorama
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using Python venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Input Images

Create an `inputs/` directory and place overlapping images:

```bash
mkdir -p inputs outputs
# Add your overlapping images to the inputs/ folder
# Images should be ordered from left to right (or top to bottom)
# Example: image1.jpg, image2.jpg, image3.jpg
```

**Image Requirements:**
- Minimum 30% overlap between consecutive images
- Similar exposure/lighting (or pre-processed for consistency)
- JPEG, PNG, or BMP format
- Recommended size: 1024×768 or smaller (larger = slower processing)

### Step 5: Run Panorama Stitching

#### Basic Usage (Stitch two images)

```bash
python panorama_stitch.py --input1 inputs/image1.jpg --input2 inputs/image2.jpg --output outputs/panorama.jpg
```

#### Stitch Multiple Images (Recommended)

```bash
python panorama_stitch.py --images inputs/ --output outputs/panorama.jpg
```

#### Advanced Options

```bash
python panorama_stitch.py \
    --images inputs/ \
    --output outputs/panorama.jpg \
    --ratio 0.75 \
    --reprojThresh 4.0 \
    --showMatches \
    --blur_percent 2
```

**Parameters:**
- `--ratio`: Lowe's ratio test threshold (0.75 default, lower = stricter matching)
- `--reprojThresh`: RANSAC reprojection error threshold in pixels (4.0 default)
- `--showMatches`: Display feature matches visualization
- `--blur_percent`: Blending smoothness (2-10, higher = softer blending)

### Step 6: View Results

Output panorama is saved to `outputs/panorama.jpg`

Additional visualizations generated:
```
outputs/
├── panorama.jpg                    # Final stitched panorama
├── feature_matches_1_2.jpg         # Feature matches visualization
├── feature_matches_2_3.jpg         # (if more than 2 images)
└── matching_info.txt               # Matching statistics
```

---

## 📂 Project Structure

```
Panorama/
├── inputs/                         # Place input images here
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── outputs/                        # Generated panorama results
│   ├── panorama.jpg
│   └── matching_stats.txt
├── src/                            # Source code
│   ├── panorama_stitch.py         # Main stitching script
│   ├── stitcher.py                # Stitcher class
│   ├── feature_matcher.py         # Feature matching logic
│   └── image_blender.py           # Blending utilities
├── scripts/                        # Utility scripts
│   ├── test_panorama.py           # Test script with sample images
│   └── visualize_matches.py       # Visualization tools
├── data/                          # Sample test images
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── sample3.jpg
├── requirements.txt               # Python dependencies
├── Assignment4_Report.pdf         # Detailed assignment report
└── README.md                      # This file
```

---

## 📊 Algorithm Workflow

```
Input Images (Overlapping)
        ↓
    ┌───────────────────────────────┐
    │  FEATURE DETECTION (SIFT)     │
    │  Extract keypoints & descriptors
    └───────────────────────────────┘
        ↓
    ┌───────────────────────────────┐
    │  FEATURE MATCHING             │
    │  Find correspondences between   │
    │  descriptor vectors            │
    └───────────────────────────────┘
        ↓
    ┌───────────────────────────────┐
    │  LOWE'S RATIO TEST            │
    │  Filter ambiguous matches      │
    │  Keep high-confidence pairs    │
    └───────────────────────────────┘
        ↓
    ┌───────────────────────────────┐
    │  HOMOGRAPHY ESTIMATION        │
    │  (RANSAC)                     │
    │  Robust geometric transform   │
    └───────────────────────────────┘
        ↓
    ┌───────────────────────────────┐
    │  PERSPECTIVE WARPING          │
    │  Transform images to common   │
    │  coordinate system            │
    └───────────────────────────────┘
        ↓
    ┌───────────────────────────────┐
    │  IMAGE BLENDING               │
    │  Smooth overlapping regions   │
    │  Handle exposure differences  │
    └───────────────────────────────┘
        ↓
    Output Panorama
```

---

## 🖼️ Sample Workflow

### Example: Stitching Mount Rainier Images

1. **Capture 3 overlapping photos** taken from left to right

2. **Feature Detection**: SIFT finds ~500-1000 distinctive keypoints in each image

3. **Feature Matching**: 
   - Initial matches: ~300-500
   - After ratio test: ~100-150 good matches
   
4. **Homography Estimation**:
   - RANSAC iterations: ~1000-5000
   - Inliers found: ~80-120
   
5. **Warping & Stitching**:
   - Image 1 stays in reference frame
   - Image 2 warped relative to Image 1
   - Image 3 warped relative to Image 1-2 combined
   
6. **Blending**: Smooth transitions in overlap regions

7. **Result**: Single wide panoramic image of Mount Rainier

---

## 📈 Performance Metrics

### Evaluation Criteria

| Metric | Description | Target |
|--------|-------------|--------|
| **Number of Matches** | Valid feature correspondences after filtering | >50 matches |
| **RANSAC Inliers** | Matches consistent with computed homography | >80% inliers |
| **Reprojection Error** | Pixel error of reprojected points | <5 pixels |
| **Blend Quality** | Smoothness of overlapping regions | Visual inspection |
| **Panorama Dimensions** | Output resolution | Depends on inputs |
| **Processing Time** | Speed of stitching | <30 seconds for typical images |

### Typical Results

- **Two images**: 5-15 seconds
- **Three images**: 15-30 seconds
- **Quality**: Depends on overlap quality and image alignment

---

## ⚠️ Common Issues & Troubleshooting

### Issue 1: "Not enough features matched"
**Causes**: Poor image overlap, low texture content, very different scales
**Solution**:
```bash
# Lower ratio test threshold for less strict matching
python panorama_stitch.py --images inputs/ --ratio 0.85 --output outputs/panorama.jpg
```

### Issue 2: "Homography estimation failed"
**Causes**: Insufficient good matches, rotation >45 degrees
**Solution**:
- Ensure minimum 30% overlap between images
- Take images from same height/angle
- Increase image texture/detail (avoid blank walls)

### Issue 3: "Visible seams in panorama"
**Causes**: Exposure differences, blending issue
**Solution**:
```bash
# Adjust blending window
python panorama_stitch.py --images inputs/ --blur_percent 5 --output outputs/panorama.jpg
```

### Issue 4: "Output image dimensions too large"
**Causes**: Many images or very large input sizes
**Solution**:
```python
# Resize images before stitching
import cv2
img = cv2.imread('image.jpg')
resized = cv2.resize(img, (1024, 768))
cv2.imwrite('image_small.jpg', resized)
```

### Issue 5: "Memory error / Out of memory"
**Causes**: Very large images or insufficient system RAM
**Solution**:
- Resize input images to smaller resolution
- Close other applications
- Run on machine with more RAM

---

## 🎓 Course Information

- **Course**: CAP 6411 - Computer Vision Systems
- **Assignment**: A4 - Panorama Stitching
- **Institution**: University of Central Florida (UCF)
- **Semester**: Fall 2025

---

## 📚 Key Computer Vision Concepts

### 1. Scale-Invariant Feature Transform (SIFT)
- Detects keypoints at multiple scales using Difference of Gaussians (DoG)
- Computes orientation-invariant descriptors
- Robust to rotation, scale, and illumination changes

### 2. Feature Matching
- Brute-force: Compare all descriptor pairs
- FLANN: Approximate nearest neighbor for speed
- Lowe's ratio test: Confidence filtering

### 3. RANSAC (Random Sample Consensus)
- Robust estimation method for outlier handling
- Iteratively samples minimum feature sets
- Selects model with most consensus (inliers)
- Probability of success: \(N = \log(1-w^n) / \log(1-(1-\epsilon)^n)\)
  where w = inlier rate, n = sample size

### 4. Homography Matrix
- 3×3 matrix relating corresponding points: \(p' = H \cdot p\)
- Represents perspective transformation
- Computed from ≥4 point correspondences

### 5. Image Blending
- Linear blend: \(I_{out} = \alpha I_1 + (1-\alpha) I_2\)
- Multi-band blending: Apply blending at multiple image scales
- Feathering: Smooth transition using Gaussian weights

---

## 🔗 References & Resources

- [SIFT Algorithm - David Lowe](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [OpenCV Panorama Documentation](https://docs.opencv.org/master/d7/d4d/tutorial_py_panorama.html)
- [RANSAC Paper](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [Image Stitching Tutorial - PyImageSearch](https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/)
- [Computer Vision: Algorithms and Applications - Richard Szeliski](https://szeliski.org/Book/)

---

## 📧 Support

For questions or issues:
- Check existing issues in repository
- Refer to this README and assignment report
- Consult course materials and lectures
- Reach out during office hours

---

## 📝 Notes for Best Results

1. **Image Capture Tips**:
   - Keep camera at same height
   - Maintain ~30-50% overlap between consecutive images
   - Use consistent focus and exposure
   - Avoid moving objects

2. **Pre-processing**:
   - Resize very large images (>2000 pixels)
   - Ensure consistent brightness across images
   - Remove lens distortion if severe

3. **Debugging**:
   - Use `--showMatches` flag to visualize feature correspondences
   - Check `matching_info.txt` for statistics
   - Verify input image ordering (left to right)

4. **Advanced**:
   - Implement bundle adjustment for multi-image stitching
   - Use deep learning-based feature descriptors (SuperPoint, DISK)
   - Apply cylindrical projection for wider panoramas

---

**Last Updated**: November 21, 2025  
**Repository**: https://github.com/ka234388/Panorama

## License
This project is for academic purposes as part of CAP 6411 course at UCF.
