"""
Step 2: Enhancement
Applies Histogram Equalization and Contrast Stretching.
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
    
    eq_img = cv2.equalizeHist(gray)
    
    min_val, max_val = np.min(gray), np.max(gray)
    if max_val > min_val:
        stretched_img = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        stretched_img = gray
        
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(eq_img, cmap='gray')
    plt.title("Hist Equalized")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(stretched_img, cmap='gray')
    plt.title("Contrast Stretched")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Enhancement\nHistogram Equalization chosen for crack detection — redistributes intensity to improve visibility of low-contrast crack regions in drone images")
    plt.tight_layout()
    plt.savefig(f"output_figures/step2_enhancement_img{idx}.png")
    plt.show()
