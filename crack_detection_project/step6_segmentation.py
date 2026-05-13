"""
Step 6: Segmentation
Applies Otsu's and Adaptive Thresholding to equalized images.
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

for idx, path in enumerate(image_paths, 1):
    img = cv2.imread(path)
    if img is None: continue
    resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Preprocess: Histogram Equalization
    enhanced = cv2.equalizeHist(gray)
    
    # Otsu's Thresholding
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced Grayscale")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(otsu, cmap='gray')
    plt.title("Otsu Result")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(adaptive, cmap='gray')
    plt.title("Adaptive Result")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Segmentation\nAdaptive thresholding handles uneven illumination in drone images better than global Otsu — reduces over-segmentation in bright regions")
    plt.tight_layout()
    plt.savefig(f"output_figures/step6_segmentation_img{idx}.png")
    plt.show()
