"""
Step 5: Histogram Analysis
Analyzes histograms of Original, Equalized, and Stretched images.
Histogram analysis verifies enhancement effectiveness — equalized histogram should show flatter distribution indicating improved contrast.
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
        
    hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_eq = cv2.calcHist([eq_img], [0], None, [256], [0, 256])
    hist_stretch = cv2.calcHist([stretched_img], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(hist_orig, color='gray', label='Original')
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.subplot(132)
    plt.plot(hist_eq, color='blue', label='Equalized')
    plt.title("Hist Equalized")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.subplot(133)
    plt.plot(hist_stretch, color='orange', label='Stretched')
    plt.title("Contrast Stretched")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.suptitle(f"Image {idx}: Histogram Analysis\nHistogram analysis verifies enhancement effectiveness — equalized histogram should show flatter distribution indicating improved contrast")
    plt.tight_layout()
    plt.savefig(f"output_figures/step5_histogram_img{idx}.png")
    plt.show()
