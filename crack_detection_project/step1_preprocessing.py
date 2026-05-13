"""
Step 1: Preprocessing
Preprocessing standardizes input dimensions and converts to grayscale for consistent processing across the crack detection pipeline.
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
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Color")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title("Resized Color")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Preprocessing")
    plt.tight_layout()
    plt.savefig(f"output_figures/step1_preprocessing_img{idx}.png")
    plt.show()
