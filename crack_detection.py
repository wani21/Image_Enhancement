import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a directory to save figures if it doesn't exist
output_dir = "output_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_synthetic_images(num_images=3, size=512):
    """
    Generate synthetic test images of concrete surfaces with cracks.
    Creates a noisy gray background with thin dark lines simulating cracks.
    """
    images = []
    for i in range(num_images):
        img = np.ones((size, size), dtype=np.uint8) * 160
        noise = np.random.normal(0, 15, (size, size)).astype(np.int8)
        img = cv2.add(img, noise.view(np.uint8))
        
        num_cracks = np.random.randint(1, 4)
        for _ in range(num_cracks):
            pts = []
            x, y = np.random.randint(50, size-50, 2)
            for _ in range(np.random.randint(3, 8)):
                pts.append([x, y])
                x += np.random.randint(-40, 40)
                y += np.random.randint(20, 80)
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            
            thickness = np.random.randint(1, 4)
            cv2.polylines(img, [pts], isClosed=False, color=(30), thickness=thickness, lineType=cv2.LINE_AA)
            
            for pt in pts[1:-1]:
                if np.random.random() > 0.5:
                    bx, by = pt[0]
                    bx += np.random.randint(-30, 30)
                    by += np.random.randint(-30, 30)
                    cv2.line(img, tuple(pt[0]), (bx, by), (50), np.random.randint(1, 3))
                    
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append((f"synthetic_crack_{i+1}", img_bgr))
    return images

# ==========================================
# PART 1 — Fundamental Image Processing
# ==========================================

def load_and_preprocess(image_bgr, name):
    """
    Step 1 — Image Preprocessing
    Reads image, resizes to 512x512, converts to Grayscale.
    Displays original vs grayscale side-by-side.
    """
    resized = cv2.resize(image_bgr, (512, 512))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), plt.title('Original RGB')
    plt.subplot(122), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
    plt.suptitle("Step 1: Preprocessing")
    plt.savefig(os.path.join(output_dir, f"{name}_step1_preprocessing.png"))
    plt.show()
    
    return resized, gray

def enhancement(gray_img, name):
    """
    Step 2 — Point Processing / Enhancement
    Applies Histogram Equalization and Contrast Stretching.
    JUSTIFICATION: Histogram equalization is preferred for crack visibility in low-contrast drone images
    because it redistributes pixel intensities globally, making dark cracks stand out more against the background.
    """
    eq_img = cv2.equalizeHist(gray_img)
    
    min_val = np.min(gray_img)
    max_val = np.max(gray_img)
    if max_val == min_val:
        stretched_img = gray_img
    else:
        stretched_img = ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(eq_img, cmap='gray'), plt.title('Histogram Equalization')
    plt.subplot(122), plt.imshow(stretched_img, cmap='gray'), plt.title('Contrast Stretching')
    plt.suptitle("Step 2: Enhancement\n(Hist. Eq. preferred for crack visibility)")
    plt.savefig(os.path.join(output_dir, f"{name}_step2_enhancement.png"))
    plt.show()
    
    return eq_img, stretched_img

def spatial_filtering(gray_img, name):
    """
    Step 3 — Spatial Domain Filtering
    Applies Gaussian Blur and Median Filter.
    JUSTIFICATION: Median filter preserves edges better, which is good for crack-like thin structures.
    """
    gaussian = cv2.GaussianBlur(gray_img, (5, 5), 1)
    median = cv2.medianBlur(gray_img, 5)
    
    # The actual plot for Step 3 will be generated in the comparison figures section as requested
    return gaussian, median

def frequency_filtering(gray_img, name):
    """
    Step 4 — Frequency Domain Filtering
    Applies DFT, Low-pass filter (zeros high frequencies).
    """
    dft = np.fft.fft2(gray_img)
    dft_shift = np.fft.fftshift(dft)
    
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    rows, cols = gray_img.shape
    crow, ccol = rows // 2, cols // 2
    r = 60
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), r, 1, -1)
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(gray_img, cmap='gray'), plt.title('Original')
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
    plt.subplot(133), plt.imshow(img_back, cmap='gray'), plt.title('Filtered Result (Low-pass)')
    plt.suptitle("Step 4: Frequency Domain Filtering")
    plt.savefig(os.path.join(output_dir, f"{name}_step4_frequency.png"))
    plt.show()
    
    return magnitude_spectrum, img_back

def histogram_analysis(gray_img, eq_img, stretched_img, name):
    """
    Step 5 — Histogram Analysis
    Plots histograms: Original vs Equalized vs Stretched on one figure.
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist(gray_img.ravel(), 256, [0, 256], color='blue')
    plt.title('Original Histogram')
    
    plt.subplot(132)
    plt.hist(eq_img.ravel(), 256, [0, 256], color='green')
    plt.title('Equalized Histogram')
    
    plt.subplot(133)
    plt.hist(stretched_img.ravel(), 256, [0, 256], color='red')
    plt.title('Stretched Histogram')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_step5_histograms.png"))
    plt.show()

# ==========================================
# PART 2 — Advanced Image Processing
# ==========================================

def segmentation(filtered_img, name):
    """
    Step 6 — Segmentation
    Applies Otsu's Thresholding and Adaptive Thresholding.
    JUSTIFICATION: Adaptive thresholding handles uneven illumination better.
    """
    _, otsu = cv2.threshold(filtered_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(filtered_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # The actual plot for Step 6 will be generated in the comparison figures section
    return otsu, adaptive

def edge_detection(filtered_img, name):
    """
    Step 7 — Edge Detection
    Applies Canny and Sobel Edge Detection.
    JUSTIFICATION: Canny gives cleaner, continuous crack edges; Sobel is noisier but fast.
    """
    canny = cv2.Canny(filtered_img, 50, 150)
    
    sobelx = cv2.Sobel(filtered_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(filtered_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    
    # The actual plot for Step 7 will be generated in the comparison figures section
    return canny, sobel

def morphological_ops(canny_img, name):
    """
    Step 8 — Morphological Operations
    Applies Dilation (thicken), Erosion (clean noise), and Closing (close gaps).
    """
    kernel = np.ones((3, 3), np.uint8)
    
    dilated = cv2.dilate(canny_img, kernel, iterations=2)
    eroded = cv2.erode(canny_img, kernel, iterations=1)
    closed = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
    
    plt.figure(figsize=(16, 4))
    plt.subplot(141), plt.imshow(canny_img, cmap='gray'), plt.title('Canny Edge')
    plt.subplot(142), plt.imshow(dilated, cmap='gray'), plt.title('Dilated (Thickened)')
    plt.subplot(143), plt.imshow(eroded, cmap='gray'), plt.title('Eroded (Cleaned)')
    plt.subplot(144), plt.imshow(closed, cmap='gray'), plt.title('Closed (Gaps Filled)')
    plt.suptitle("Step 8: Morphological Operations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_step8_morphology.png"))
    plt.show()
    
    return dilated, eroded, closed

def feature_extraction(original_bgr, morph_img, name):
    """
    Step 9 — Feature Extraction
    Applies Hough Line Transform and Contour Detection.
    """
    hough_output = original_bgr.copy()
    lines = cv2.HoughLinesP(morph_img, 1, np.pi/180, threshold=40, minLineLength=20, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    contour_output = original_bgr.copy()
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_output, contours, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(hough_output, cv2.COLOR_BGR2RGB)), plt.title('Hough Lines Overlay')
    plt.subplot(122), plt.imshow(cv2.cvtColor(contour_output, cv2.COLOR_BGR2RGB)), plt.title('Contours Overlay')
    plt.suptitle("Step 9: Feature Extraction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_step9_features.png"))
    plt.show()
    
    return hough_output, contour_output

# ==========================================
# ANALYSIS & COMPARISON VISUALIZATIONS
# ==========================================

def analysis_comparisons(name, img_dict):
    """
    Generates all 5 mandatory comparison figures as requested for the lab report.
    """
    
    # 1. Filter Comparison Figure: Gaussian vs Median
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img_dict['gray'], cmap='gray'), plt.title('Original Grayscale')
    plt.subplot(132), plt.imshow(img_dict['gaussian'], cmap='gray'), plt.title('Gaussian Blur\n(Edges blurred)')
    plt.subplot(133), plt.imshow(img_dict['median'], cmap='gray'), plt.title('Median Filter\n(Edges preserved)')
    plt.suptitle("Filter Comparison: Median preserves edges better, good for crack-like thin structures", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_comp1_filtering.png"))
    plt.show()

    # 2. Segmentation Comparison Figure: Otsu vs Adaptive
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img_dict['gray'], cmap='gray'), plt.title('Original Grayscale')
    plt.subplot(132), plt.imshow(img_dict['otsu'], cmap='gray'), plt.title("Otsu's Thresholding\n(Global)")
    plt.subplot(133), plt.imshow(img_dict['adaptive'], cmap='gray'), plt.title('Adaptive Thresholding\n(Local)')
    plt.suptitle("Segmentation Comparison: Adaptive handles uneven illumination better", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_comp2_segmentation.png"))
    plt.show()

    # 3. Edge Detection Comparison Figure: Canny vs Sobel
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img_dict['gray'], cmap='gray'), plt.title('Original Grayscale')
    plt.subplot(132), plt.imshow(img_dict['canny'], cmap='gray'), plt.title('Canny Edge\n(Cleaner edges)')
    plt.subplot(133), plt.imshow(img_dict['sobel'], cmap='gray'), plt.title('Sobel Edge\n(Noisier)')
    plt.suptitle("Edge Detection Comparison: Canny gives cleaner, continuous crack edges; Sobel is noisier but fast", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_comp3_edges.png"))
    plt.show()

    # 4. Parameter Sensitivity Figure: Canny with 3 thresholds
    c1 = cv2.Canny(img_dict['median'], 30, 90)
    c2 = cv2.Canny(img_dict['median'], 50, 150)
    c3 = cv2.Canny(img_dict['median'], 100, 200)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(c1, cmap='gray'), plt.title('Canny (30, 90)\n(More False Positives)')
    plt.subplot(132), plt.imshow(c2, cmap='gray'), plt.title('Canny (50, 150)\n(Balanced)')
    plt.subplot(133), plt.imshow(c3, cmap='gray'), plt.title('Canny (100, 200)\n(Misses details)')
    plt.suptitle("Parameter Sensitivity: Impact of Canny Thresholds", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_comp4_canny_sensitivity.png"))
    plt.show()

    # 5. Full Pipeline Summary Figure
    plt.figure(figsize=(16, 8))
    
    plt.subplot(241), plt.imshow(cv2.cvtColor(img_dict['original'], cv2.COLOR_BGR2RGB)), plt.title('1. Original RGB')
    plt.axis('off')
    
    plt.subplot(242), plt.imshow(img_dict['gray'], cmap='gray'), plt.title('2. Grayscale')
    plt.axis('off')
    
    plt.subplot(243), plt.imshow(img_dict['eq'], cmap='gray'), plt.title('3. Enhanced (Hist Eq)')
    plt.axis('off')
    
    plt.subplot(244), plt.imshow(img_dict['median'], cmap='gray'), plt.title('4. Filtered (Median)')
    plt.axis('off')
    
    plt.subplot(245), plt.imshow(img_dict['adaptive'], cmap='gray'), plt.title('5. Segmented (Adaptive)')
    plt.axis('off')
    
    plt.subplot(246), plt.imshow(img_dict['canny'], cmap='gray'), plt.title('6. Edges (Canny)')
    plt.axis('off')
    
    plt.subplot(247), plt.imshow(img_dict['closed'], cmap='gray'), plt.title('7. Morphological (Closed)')
    plt.axis('off')
    
    plt.subplot(248), plt.imshow(cv2.cvtColor(img_dict['hough'], cv2.COLOR_BGR2RGB)), plt.title('8. Features (Hough)')
    plt.axis('off')
    
    plt.suptitle(f"Full Pipeline Summary - {name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_comp5_full_pipeline.png"))
    plt.show()

def main():
    print("Starting Autonomous Crack Detection Pipeline...")
    
    # Generate 3 synthetic test images
    images = create_synthetic_images(num_images=3)
    
    for idx, (name, img_bgr) in enumerate(images):
        print(f"\nProcessing image {idx+1}/3: {name}")
        
        # Step 1: Preprocess
        original, gray = load_and_preprocess(img_bgr, name)
        
        # Step 2: Enhancement
        eq_img, stretched_img = enhancement(gray, name)
        
        # Step 3: Spatial Filtering
        gaussian, median = spatial_filtering(gray, name) 
        
        # Step 4: Frequency Filtering
        mag_spec, freq_filtered = frequency_filtering(gray, name)
        
        # Step 5: Histogram Analysis
        histogram_analysis(gray, eq_img, stretched_img, name)
        
        # Step 6: Segmentation (using median filtered image for cleaner segmentation)
        otsu, adaptive = segmentation(median, name)
        
        # Step 7: Edge Detection (using median filtered image)
        canny, sobel = edge_detection(median, name)
        
        # Step 8: Morphological Operations
        dilated, eroded, closed = morphological_ops(canny, name)
        
        # Step 9: Feature Extraction
        hough_output, contour_output = feature_extraction(original, closed, name)
        
        # Collect outputs for analysis figures
        img_dict = {
            'original': original,
            'gray': gray,
            'eq': eq_img,
            'stretched': stretched_img,
            'gaussian': gaussian,
            'median': median,
            'otsu': otsu,
            'adaptive': adaptive,
            'canny': canny,
            'sobel': sobel,
            'closed': closed,
            'hough': hough_output,
            'contour': contour_output
        }
        
        # Generate Analysis & Comparison Visualizations
        analysis_comparisons(name, img_dict)
        
    print("\nPipeline completed successfully. All figures saved in 'output_figures' directory.")

if __name__ == "__main__":
    main()
