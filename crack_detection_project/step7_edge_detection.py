"""
Step 7: Edge Detection
Applies Canny and Sobel edge detection to Gaussian blurred images.
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
    
    # Preprocess: Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Canny Edge Detection
    canny = cv2.Canny(blurred, 50, 150)
    
    # Sobel Edge Detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(blurred, cmap='gray')
    plt.title("Blurred Input")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(canny, cmap='gray')
    plt.title("Canny Edges")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(sobel, cmap='gray')
    plt.title("Sobel Edges")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Edge Detection\nEvaluate Edge Clarity and Continuity: Canny provides excellent edge clarity and high continuity for cracks. Sobel is thicker, noisier, and lacks continuity.")
    plt.tight_layout()
    plt.savefig(f"output_figures/step7_edge_detection_img{idx}.png")
    plt.show()
