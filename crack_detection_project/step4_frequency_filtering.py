"""
Step 4: Frequency Filtering
Applies 2D DFT and Low-pass filtering.
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
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), 60, 1, -1)
    
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(img_back, cmap='gray')
    plt.title("Low-pass Filtered Result")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Frequency Filtering\nLow-pass DFT filter removes high-frequency noise while retaining large crack structures")
    plt.tight_layout()
    plt.savefig(f"output_figures/step4_frequency_filtering_img{idx}.png")
    plt.show()
