"""
Step 9: Feature Extraction
Applies Hough Line Transform and Contour Detection to morphologically closed edges.
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
    
    # Run pipeline to Morphological Closing inline
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    
    # Feature Extraction
    hough_overlay = resized.copy()
    lines = cv2.HoughLinesP(closed_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    contour_overlay = resized.copy()
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_overlay, contours, -1, (0, 0, 255), 2)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title("Original Color")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(hough_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Hough Lines Overlay")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Contours Overlay")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Feature Extraction\nHough Transform maps linear crack geometry; Contours outline full crack region boundaries for area estimation")
    plt.tight_layout()
    plt.savefig(f"output_figures/step9_features_img{idx}.png")
    plt.show()
