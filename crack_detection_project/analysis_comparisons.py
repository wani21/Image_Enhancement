"""
Analysis Comparisons
Produces 5 dedicated comparison/analysis figures covering the full image set.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

os.makedirs("output_figures", exist_ok=True)
image_paths = sorted(glob.glob("images/*.jpg") + glob.glob("images/*.jpeg"))
if not image_paths:
    print("No images found in images/ directory. Please add .jpg images and run again.")
    sys.exit()

num_images = min(5, len(image_paths))

# Figure A: Filter Comparison (Gaussian vs Median)
fig_A = plt.figure(figsize=(15, 4 * num_images))
for idx in range(num_images):
    img = cv2.imread(image_paths[idx])
    resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    gaussian = cv2.GaussianBlur(gray, (5, 5), 1)
    median = cv2.medianBlur(gray, 5)
    
    ax1 = fig_A.add_subplot(num_images, 3, idx*3 + 1)
    ax1.imshow(gray, cmap='gray')
    if idx == 0: ax1.set_title("Original Grayscale")
    ax1.axis('off')
    
    ax2 = fig_A.add_subplot(num_images, 3, idx*3 + 2)
    ax2.imshow(gaussian, cmap='gray')
    if idx == 0: ax2.set_title("Gaussian")
    ax2.axis('off')
    
    ax3 = fig_A.add_subplot(num_images, 3, idx*3 + 3)
    ax3.imshow(median, cmap='gray')
    if idx == 0: ax3.set_title("Median")
    ax3.axis('off')

fig_A.suptitle("Spatial Filter Comparison: Gaussian vs Median", fontsize=16)
plt.tight_layout()
plt.savefig("output_figures/analysis_A_filter_comparison.png")
plt.show()

# Figure B: Segmentation Comparison (Otsu vs Adaptive)
fig_B = plt.figure(figsize=(15, 4 * num_images))
for idx in range(num_images):
    img = cv2.imread(image_paths[idx])
    resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    enhanced = cv2.equalizeHist(gray)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    ax1 = fig_B.add_subplot(num_images, 3, idx*3 + 1)
    ax1.imshow(enhanced, cmap='gray')
    if idx == 0: ax1.set_title("Enhanced Grayscale")
    ax1.axis('off')
    
    ax2 = fig_B.add_subplot(num_images, 3, idx*3 + 2)
    ax2.imshow(otsu, cmap='gray')
    if idx == 0: ax2.set_title("Otsu Result")
    ax2.axis('off')
    # Adding annotation under Otsu result about over/under-segmentation
    ax2.text(0.5, -0.15, "Check for over/under-segmentation", transform=ax2.transAxes, ha='center', fontsize=10)
    
    ax3 = fig_B.add_subplot(num_images, 3, idx*3 + 3)
    ax3.imshow(adaptive, cmap='gray')
    if idx == 0: ax3.set_title("Adaptive Result")
    ax3.axis('off')

fig_B.suptitle("Segmentation Comparison: Otsu vs Adaptive", fontsize=16)
plt.tight_layout()
plt.savefig("output_figures/analysis_B_segmentation_comparison.png")
plt.show()

# Figure C: Edge Detection Comparison (Canny vs Sobel)
fig_C = plt.figure(figsize=(15, 4 * num_images))
for idx in range(num_images):
    img = cv2.imread(image_paths[idx])
    resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blurred, 50, 150)
    
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    ax1 = fig_C.add_subplot(num_images, 3, idx*3 + 1)
    ax1.imshow(blurred, cmap='gray')
    if idx == 0: ax1.set_title("Blurred Input")
    ax1.axis('off')
    
    ax2 = fig_C.add_subplot(num_images, 3, idx*3 + 2)
    ax2.imshow(canny, cmap='gray')
    if idx == 0: ax2.set_title("Canny Edges")
    ax2.axis('off')
    
    ax3 = fig_C.add_subplot(num_images, 3, idx*3 + 3)
    ax3.imshow(sobel, cmap='gray')
    if idx == 0: ax3.set_title("Sobel Edges")
    ax3.axis('off')

fig_C.suptitle("Edge Detection Comparison: Canny vs Sobel", fontsize=16)
plt.tight_layout()
plt.savefig("output_figures/analysis_C_edge_comparison.png")
plt.show()

# Figure D: Canny Parameter Sensitivity Study (Image 1 only)
img1 = cv2.imread(image_paths[0])
resized1 = cv2.resize(img1, (512, 512))
gray1 = cv2.cvtColor(resized1, cv2.COLOR_BGR2GRAY)
blurred1 = cv2.GaussianBlur(gray1, (5, 5), 1)

canny_thresh1 = cv2.Canny(blurred1, 30, 90)
canny_thresh2 = cv2.Canny(blurred1, 50, 150)
canny_thresh3 = cv2.Canny(blurred1, 100, 200)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(canny_thresh1, cmap='gray')
plt.title("Canny (30, 90)")
plt.axis('off')

plt.subplot(132)
plt.imshow(canny_thresh2, cmap='gray')
plt.title("Canny (50, 150)")
plt.axis('off')

plt.subplot(133)
plt.imshow(canny_thresh3, cmap='gray')
plt.title("Canny (100, 200)")
plt.axis('off')

plt.suptitle("Parameter Sensitivity: Effect of Canny Thresholds on Crack Edge Detection", fontsize=16)
plt.tight_layout()
plt.savefig("output_figures/analysis_D_parameter_sensitivity.png")
plt.show()

# Figure E: Full Pipeline Summary (Image 1 only)
enhanced1 = cv2.equalizeHist(gray1)
_, otsu1 = cv2.threshold(blurred1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
closed1 = cv2.morphologyEx(canny_thresh2, cv2.MORPH_CLOSE, kernel)

hough_overlay1 = resized1.copy()
lines = cv2.HoughLinesP(closed1, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough_overlay1, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(16, 8))

plt.subplot(241), plt.imshow(cv2.cvtColor(resized1, cv2.COLOR_BGR2RGB)), plt.title("1. Original Color")
plt.axis('off')

plt.subplot(242), plt.imshow(gray1, cmap='gray'), plt.title("2. Grayscale")
plt.axis('off')

plt.subplot(243), plt.imshow(enhanced1, cmap='gray'), plt.title("3. Enhanced (Hist Eq)")
plt.axis('off')

plt.subplot(244), plt.imshow(blurred1, cmap='gray'), plt.title("4. Gaussian Filtered")
plt.axis('off')

plt.subplot(245), plt.imshow(otsu1, cmap='gray'), plt.title("5. Otsu Segmented")
plt.axis('off')

plt.subplot(246), plt.imshow(canny_thresh2, cmap='gray'), plt.title("6. Canny Edges")
plt.axis('off')

plt.subplot(247), plt.imshow(closed1, cmap='gray'), plt.title("7. Morphological Closing")
plt.axis('off')

plt.subplot(248), plt.imshow(cv2.cvtColor(hough_overlay1, cv2.COLOR_BGR2RGB)), plt.title("8. Hough Lines Overlay")
plt.axis('off')

plt.suptitle("Full Pipeline Summary", fontsize=18)
plt.tight_layout()
plt.savefig("output_figures/analysis_E_full_pipeline_summary.png")
plt.show()
