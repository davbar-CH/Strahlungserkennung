import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import imutils
import os
from config import Config

def load_image_files(folder_path):
    """Load all image files from the specified folder."""
    try:
        bilder = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        bilder = [os.path.join(folder_path, f) for f in bilder]
        print(f"Found {len(bilder)} image files")
        return bilder
    except Exception as e:
        print(f"Error loading images: {e}")
        return []

def load_and_prepare_image(image_path, rotation_angle=51):
    """Load image, convert to grayscale, and rotate."""
    try:
        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        

        # #Debug your transformation matrix

        # (h, w) = image.shape[:2]
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # print("Transformation matrix M:")
        # print(M)
        # print("Matrix shape:", M.shape)
        # print("Matrix values range:", M.min(), "to", M.max())

        # # Check input image
        # print("Input image shape:", image.shape)
        # print("Input image dtype:", image.dtype)
        # print("Input image value range:", image.min(), "to", image.max())
        # print("Input image non-zero pixels:", np.count_nonzero(image))

        # # Check output dimensions
        # print("Output dimensions:", (w, h))
        # myrotated = cv2.warpAffine(image, M, (w, h))
        
        # if np.count_nonzero(myrotated) == 0:
        #  print("WARNING: Result is all zeros!")
        
        # # Try with different border mode
        # rotated_reflected = cv2.warpAffine(image, M, (w, h), 
        #                                  borderMode=cv2.BORDER_REFLECT)
        # # Rotate image

        # rotated = imutils.rotate(image, rotation_angle)
        # print(f"Image loaded and rotated by {rotation_angle} degrees")
        # return rotated_reflected

        # Rotate image
        #rotated = imutils.rotate(image, rotation_angle)
        #print(f"Image loaded and rotated by {rotation_angle} degrees")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def crop_image(image, x_start, y_start, width, height):
    """
    Crops a given image (NumPy array) to the specified rectangle.
    Parameters:
        image (numpy.ndarray): The input image to crop.
        x_start (int): The starting x-coordinate (column) for the crop.
        y_start (int): The starting y-coordinate (row) for the crop.
        width (int): The width of the crop rectangle.
        height (int): The height of the crop rectangle.
    Returns:
        numpy.ndarray or None: The cropped image as a NumPy array if successful, 
        otherwise None if an error occurs.
    Notes:
        - The function ensures the crop does not exceed the image boundaries.
        - If an exception occurs during cropping, an error message is printed and None is returned.
    """
    """Crop image to specified dimensions."""
    try:
        y_end = min(y_start + height, image.shape[0])
        x_end = min(x_start + width, image.shape[1])
        
        cropped = image[y_start:y_end, x_start:x_end]
        print(f"Image cropped to {cropped.shape}")
        return cropped
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None

def detect_lines_hough(image, threshold=10, line_length=30, line_gap=5, debug=True):
    """
    Detect lines in an image using the Hough transform with multiple edge detection and thresholding strategies.
    Parameters:
        image (numpy.ndarray): Input grayscale image for line detection.
        threshold (int, optional): Base threshold for the Hough transform accumulator. Default is 50.
        line_length (int, optional): Minimum accepted length of detected lines (used in probabilistic Hough fallback). Default is 30.
        line_gap (int, optional): Maximum allowed gap between line segments to treat them as a single line (used in probabilistic Hough fallback). Default is 5.
        debug (bool, optional): If True, prints debugging information about edge and line detection. Default is True.
    Returns:
        tuple:
            - hough_peaks (tuple or None): Peaks detected by the Hough transform (or probabilistic Hough fallback), or None if detection fails.
            - edges (numpy.ndarray or None): Edge image used for detection, or None if detection fails.
    Notes:
        - Applies Canny edge detection with multiple parameter sets to improve robustness.
        - Tries several Hough accumulator thresholds to maximize line detection.
        - Falls back to probabilistic Hough transform if standard Hough fails.
        - Requires `cv2`, `numpy`, and `skimage.transform` (for `hough_line` and `hough_line_peaks`).
    """
    """Detect lines using Hough transform with debugging."""
    try:
        # Apply edge detection with different parameters
        edges1 = cv2.Canny(image, 30, 100, apertureSize=3)
        edges2 = cv2.Canny(image, 50, 150, apertureSize=3)
        edges3 = cv2.Canny(image, 100, 200, apertureSize=3)
        
        if debug:
            print(f"Edge pixels detected:")
            print(f"  Low threshold: {np.sum(edges1 > 0)} pixels")
            print(f"  Medium threshold: {np.sum(edges2 > 0)} pixels") 
            print(f"  High threshold: {np.sum(edges3 > 0)} pixels")
        
        # Try different edge detection results
        for i, edges in enumerate([edges1, edges2, edges3], 1):
            if np.sum(edges > 0) < 100:  # Too few edge pixels
                continue
                
            # Standard Hough Transform
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
            h, theta, d = hough_line(edges, theta=tested_angles)
            
            # Try different thresholds
            for hough_threshold in [threshold // 4, threshold // 2, threshold, threshold * 2]:
                hough_peaks = hough_line_peaks(h, theta, d, 
                                             threshold=hough_threshold,
                                             min_distance=30, 
                                             min_angle=5)
                
                num_lines = len(hough_peaks[0])
                if debug:
                    print(f"Edges {i}, threshold {hough_threshold}: {num_lines} lines")
                
                if num_lines > 0:
                    return hough_peaks, edges
        
        # If standard Hough failed, try Probabilistic Hough
        return detect_lines_probabilistic(image, debug=debug)
        
    except Exception as e:
        print(f"Error in line detection: {e}")
        return None, None

def detect_lines_probabilistic(image, debug=True):
    """Alternative line detection using Probabilistic Hough Transform."""
    try:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(edges, 
                               rho=1,           # Distance resolution
                               theta=np.pi/180, # Angle resolution
                               threshold=50,    # Minimum votes
                               minLineLength=30, # Minimum line length
                               maxLineGap=10)   # Maximum gap between line segments
        
        if debug:
            print(f"Probabilistic Hough detected: {len(lines) if lines is not None else 0} lines")
        
        if lines is not None:
            # Convert to format compatible with visualization
            hough_peaks = (np.arange(len(lines)), [], [])
            return hough_peaks, edges
        
        return None, edges
        
    except Exception as e:
        print(f"Error in probabilistic line detection: {e}")
        return None, None

def calculate_radiation_path(cropped_image, x0, y0, threshold=200):
    """Calculate radiation path from starting point."""
    try:
        height, width = cropped_image.shape[:2]
        x_strahl = x0
        y_ende = y0  # Initialize with starting point
        
        # Check if we can analyze the image
        if np.mean(cropped_image) < threshold:
            print(f"Image too dark (mean: {np.mean(cropped_image):.1f}), skipping analysis")
            return x_strahl, y_ende
        
        # Find radiation path
        for y_strahl_int in range(y0, height):
            if x_strahl >= width or y_strahl_int >= height:
                break
                
            pixel_value = cropped_image[y_strahl_int, x_strahl]
            
            if pixel_value >= threshold:
                x_strahl += 1
                y_ende = y_strahl_int
            else:
                break
        
        print(f"Radiation path calculated: start=({x0}, {y0}), end=({x_strahl}, {y_ende})")
        return x_strahl, y_ende
    except Exception as e:
        print(f"Error calculating radiation path: {e}")
        return x0, y0

def visualize_results(original_image, cropped_image, hough_peaks, edges, 
                     x0, y0, x_end, y_end, save_path=None):
    """Create visualization of the analysis results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original rotated image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Rotated Image')
        axes[0, 0].axis('off')
        
        # Cropped image with radiation path
        axes[0, 1].imshow(cropped_image, cmap='gray')
        axes[0, 1].plot([x0, x_end], [y0, y_end], 'r-', linewidth=3, label='Radiation Path')
        axes[0, 1].plot(x0, y0, 'go', markersize=8, label='Start Point')
        axes[0, 1].plot(x_end, y_end, 'ro', markersize=8, label='End Point')
        axes[0, 1].set_title('Cropped Image with Radiation Path')
        axes[0, 1].legend()
        axes[0, 1].axis('off')
        
        # Edge detection result
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection (Canny)')
        axes[1, 0].axis('off')
        
        # Hough lines on cropped image
        axes[1, 1].imshow(cropped_image, cmap='gray')
        if hough_peaks is not None:
            for angle, dist in zip(hough_peaks[1], hough_peaks[2]):
                y0_line = (dist - 0 * np.cos(angle)) / np.sin(angle)
                y1_line = (dist - cropped_image.shape[1] * np.cos(angle)) / np.sin(angle)
                axes[1, 1].plot([0, cropped_image.shape[1]], [y0_line, y1_line], 'r-', alpha=0.7)
        axes[1, 1].set_title('Detected Lines (Hough Transform)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Config.OUTPUT_FOLDER_PATH, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {Config.OUTPUT_FOLDER_PATH}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
      

def analyze_radiation_image(image_path, crop_params=None, rotation_angle=51, 
                          radiation_start=(100, 100), threshold=200, debug=True):
    """
    Analyze a radiation image to detect lines and calculate radiation path.
    This function processes an image by loading, rotating, cropping, and analyzing it
    to detect radiation patterns and calculate the path of radiation from a starting point.
    Args:
        image_path (str): Path to the input image file.
        crop_params (dict, optional): Dictionary containing crop parameters with keys:
            - 'x_start': Starting x coordinate for cropping
            - 'y_start': Starting y coordinate for cropping  
            - 'width': Width of the crop area
            - 'height': Height of the crop area
            If None, defaults to center quarter of the image.
        rotation_angle (int, optional): Angle in degrees to rotate the image. Defaults to 51.
        radiation_start (tuple, optional): Starting point (x, y) for radiation path calculation. 
            Defaults to (100, 100).
        threshold (int, optional): Threshold value for radiation path calculation. Defaults to 200.
        debug (bool, optional): Whether to enable debug output and visualization. Defaults to True.
    Returns:
        dict or None: Dictionary containing analysis results with keys:
            - 'image_path': Path to the input image
            - 'radiation_start': Starting coordinates of radiation path
            - 'radiation_end': End coordinates of calculated radiation path
            - 'detected_lines': Number of lines detected using Hough transform
            Returns None if analysis fails.
    Side Effects:
        - Creates debug visualization file if debug=True
        - Creates analyzed output image with visualization overlay
        - Prints debug information to console if debug=True
    Raises:
        Exception: Catches and prints any errors during image processing, returning None.
    """
    """Main function with debugging enabled."""
    try:
        # Load and prepare image
        image = load_and_prepare_image(image_path, rotation_angle)
        if image is None:
            return None
        
        # Set default crop parameters
        if crop_params is None:
            height, width = image.shape[:2]
            crop_params = {
                'x_start': width // 4,
                'y_start': height // 4, 
                'width': width // 2,
                'height': height // 2
            }
        
        # Crop image
        cropped = crop_image(image, **crop_params)
        save_cropped_image(cropped, "C:\\temp\\cropped_radiation_image.jpg")

        if cropped is None:
            return None
        
        if debug:
            print(f"Cropped image shape: {cropped.shape}")
            print(f"Cropped image stats: min={cropped.min()}, max={cropped.max()}, mean={cropped.mean():.1f}")
            
            # Run debug visualization
            debug_path = image_path.replace('.', '_debug.')
            debug_line_detection(cropped, debug_path)
        
        # Try multiple line detection approaches
        hough_peaks, edges = detect_lines_hough(cropped, debug=debug)
        
        # Calculate radiation path
        x0, y0 = radiation_start
        x_end, y_end = calculate_radiation_path(cropped, x0, y0, threshold)
        
        # Create visualization
        output_path = image_path.replace('.', '_analyzed.')
        visualize_results(image, cropped, hough_peaks, edges,
                         x0, y0, x_end, y_end, output_path)
        
        return {
            'image_path': image_path,
            'radiation_start': (x0, y0),
            'radiation_end': (x_end, y_end),
            'detected_lines': len(hough_peaks[0]) if hough_peaks else 0
        }
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def debug_line_detection(image, save_path=None):
    """Debug line detection with comprehensive visualization."""
    try:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Preprocessing
        enhanced, sobel = preprocess_image_for_lines(image)
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('Enhanced (CLAHE)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sobel, cmap='gray')
        axes[0, 2].set_title('Sobel Edges')
        axes[0, 2].axis('off')
        
        # Different Canny thresholds
        canny_params = [(30, 100), (50, 150), (100, 200)]
        for i, (low, high) in enumerate(canny_params):
            edges = cv2.Canny(enhanced, low, high)
            axes[1, i].imshow(edges, cmap='gray')
            axes[1, i].set_title(f'Canny {low}-{high}')
            axes[1, i].axis('off')
        
        # Line detection attempts
        hough_peaks, best_edges = detect_lines_hough(enhanced, debug=True)
        
        # Show results
        axes[2, 0].imshow(best_edges, cmap='gray')
        axes[2, 0].set_title('Best Edge Detection')
        axes[2, 0].axis('off')
        
        # Probabilistic Hough
        prob_lines = cv2.HoughLinesP(best_edges, 1, np.pi/180, 50, 
                                    minLineLength=30, maxLineGap=10)
        
        line_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        if prob_lines is not None:
            for line in prob_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[2, 1].imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        axes[2, 1].set_title(f'Detected Lines ({len(prob_lines) if prob_lines is not None else 0})')
        axes[2, 1].axis('off')
        
        # Image statistics
        stats_text = f"""
        Image Stats:
        Size: {image.shape}
        Min: {image.min()}
        Max: {image.max()}
        Mean: {image.mean():.1f}
        Std: {image.std():.1f}
        
        Edge Stats:
        Edge pixels: {np.sum(best_edges > 0)}
        Edge ratio: {np.sum(best_edges > 0) / best_edges.size:.3f}
        """
        
        axes[2, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                        verticalalignment='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Statistics')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Config.OUTPUT_FOLDER_PATH, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        print(f"Error in debug visualization: {e}")

def preprocess_image_for_lines(image):
    """Preprocess image to enhance line detection."""
    try:
        # Apply different preprocessing techniques
        
        # 1. Histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # 2. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 3. Morphological operations to enhance lines
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # 4. Try different filters
        # Sobel edge detection
        sobelx = cv2.Sobel(morph, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(morph, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        
        return morph, sobel_combined.astype(np.uint8)
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return image, image

def save_cropped_image(cropped_image, output_path):
    """Save cropped image using OpenCV."""
    try:
        success = cv2.imwrite(output_path, cropped_image)
        if success:
            print(f"✅ Cropped image saved to: {output_path}")
        else:
            print(f"❌ Failed to save image to: {output_path}")
        return success
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def main():
    """Main execution function."""
    try:
        # Load configuration
        image_folder = Config.IMAGE_FOLDER_PATH
        
        # Load all images
        image_files = load_image_files(image_folder)
        if not image_files:
            print("No images found!")
            return
        
        # Analyze first image (you can modify this to process all images)
        results = analyze_radiation_image(
            image_files[0],
            crop_params={'x_start': 200, 'y_start': 150, 'width': 6000, 'height': 4000},
            rotation_angle=51,
            radiation_start=(100, 100),
            threshold=100
        )
        
        if results:
            print("\nAnalysis Results:")
            print(f"Image: {results['image_path']}")
            print(f"Radiation path: {results['radiation_start']} -> {results['radiation_end']}")
            print(f"Radiation length: {results['radiation_length']:.2f} pixels")
            print(f"Detected lines: {results['detected_lines']}")
        
        # Process all images (optional)
        # all_results = []
        # for img_path in image_files:
        #     result = analyze_radiation_image(img_path)
        #     if result:
        #         all_results.append(result)
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
