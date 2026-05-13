"""
Step 3: Spatial Filtering
Applies Gaussian Blur and Median Filter.
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
    
    gaussian = cv2.GaussianBlur(gray, (5, 5), 1)
    median = cv2.medianBlur(gray, 5)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(gaussian, cmap='gray')
    plt.title("Gaussian Blurred")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(median, cmap='gray')
    plt.title("Median Filtered")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Spatial Filtering\nMedian filter preserves crack edges better than Gaussian — preferred for thin crack structures as it removes salt-and-pepper noise without blurring boundaries")
    plt.tight_layout()
    plt.savefig(f"output_figures/step3_spatial_filtering_img{idx}.png")
    plt.show()
