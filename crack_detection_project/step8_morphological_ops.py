"""
Step 8: Morphological Operations
Applies Dilation, Erosion, and Closing to Canny edge results.
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
    
    # Run pipeline to Canny inline
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blurred, 50, 150)
    
    # Morphological Operations
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=2)
    eroded = cv2.erode(canny, kernel, iterations=1)
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.imshow(canny, cmap='gray')
    plt.title("Canny\n(Base Edges)")
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(dilated, cmap='gray')
    plt.title("Dilated\n(Thickens crack lines)")
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(eroded, cmap='gray')
    plt.title("Eroded\n(Removes small noise dots)")
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(closed, cmap='gray')
    plt.title("Closed\n(Closes small gaps in crack lines)")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Morphological Operations")
    plt.tight_layout()
    plt.savefig(f"output_figures/step8_morphology_img{idx}.png")
    plt.show()
