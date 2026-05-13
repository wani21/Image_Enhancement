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
    
    # Improve segmentation quality (morphological noise removal)
    kernel = np.ones((3, 3), np.uint8)
    improved_adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced Grayscale")
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(otsu, cmap='gray')
    plt.title("Otsu Result\n(Identify: Over/Under-segmentation)")
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(adaptive, cmap='gray')
    plt.title("Adaptive Result\n(Detail preserved, noisy)")
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(improved_adaptive, cmap='gray')
    plt.title("Improved Adaptive\n(Noise removal & Detail preservation)")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Segmentation\nAnalyze: Adaptive prevents Otsu's over/under-segmentation. Morphological opening improves quality by removing noise while preserving crack details.")
    plt.tight_layout()
    plt.savefig(f"output_figures/step6_segmentation_img{idx}.png")
    plt.show()
