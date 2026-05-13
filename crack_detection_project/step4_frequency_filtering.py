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
    
    # Low-pass filter (removes high-frequency noise)
    fshift_lp = fshift * mask
    f_ishift_lp = np.fft.ifftshift(fshift_lp)
    img_back_lp = np.abs(np.fft.ifft2(f_ishift_lp))
    
    # High-pass filter (enhances edges/details)
    mask_hp = 1 - mask
    fshift_hp = fshift * mask_hp
    f_ishift_hp = np.fft.ifftshift(fshift_hp)
    img_back_hp = np.abs(np.fft.ifft2(f_ishift_hp))
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum")
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(img_back_lp, cmap='gray')
    plt.title("Low-pass (Noise Removal)")
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(img_back_hp, cmap='gray')
    plt.title("High-pass (Detail Preserved)")
    plt.axis('off')
    
    plt.suptitle(f"Image {idx}: Frequency Filtering\nAnalyze: Low-pass effectively removes noise but blurs details; High-pass preserves crack edges and details but keeps high-frequency noise.")
    plt.tight_layout()
    plt.savefig(f"output_figures/step4_frequency_filtering_img{idx}.png")
    plt.show()
