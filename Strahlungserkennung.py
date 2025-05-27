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
        
        # Rotate image
        rotated = imutils.rotate(image, rotation_angle)
        print(f"Image loaded and rotated by {rotation_angle} degrees")
        return rotated
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

def detect_lines_hough(image, threshold=150, line_length=100, line_gap=10):
    """
    Detect lines in an image using the Hough transform.
    Parameters:
        image (numpy.ndarray): Input image (grayscale) on which to perform line detection.
        threshold (int, optional): Minimum number of votes required to consider a line. Default is 150.
        line_length (int, optional): Minimum accepted length of detected lines. Default is 100.
        line_gap (int, optional): Maximum allowed gap between line segments to treat them as a single line. Default is 10.
    Returns:
        tuple: 
            - hough_peaks (tuple): Peaks detected by the Hough transform, containing arrays of accumulator values, angles, and distances.
            - edges (numpy.ndarray): Edge map of the input image as detected by the Canny edge detector.
    Raises:
        Exception: If an error occurs during line detection, prints the error and returns (None, None).
    """
    """Detect lines using Hough transform."""
    try:
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough line detection
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(edges, theta=tested_angles)
        
        # Find peaks
        hough_peaks = hough_line_peaks(h, theta, d, threshold=threshold, 
                                     min_distance=50, min_angle=10)
        
        print(f"Detected {len(hough_peaks[0])} lines")
        return hough_peaks, edges
    except Exception as e:
        print(f"Error in line detection: {e}")
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def analyze_radiation_image(image_path, crop_params=None, rotation_angle=51, 
                          radiation_start=(100, 100), threshold=200):
    """Main function to analyze a single radiation image."""
    try:
        # Load and prepare image
        image = load_and_prepare_image(image_path, rotation_angle)
        if image is None:
            return None
        
        # Set default crop parameters if not provided
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
        if cropped is None:
            return None
        
        # Detect lines
        hough_peaks, edges = detect_lines_hough(cropped)
        
        # Calculate radiation path
        x0, y0 = radiation_start
        x_end, y_end = calculate_radiation_path(cropped, x0, y0, threshold)
        
        # Create visualization
        output_path = image_path.replace('.', '_analyzed.')
        visualize_results(image, cropped, hough_peaks, edges, 
                         x0, y0, x_end, y_end, output_path)
        
        # Return results
        results = {
            'image_path': image_path,
            'radiation_start': (x0, y0),
            'radiation_end': (x_end, y_end),
            'radiation_length': np.sqrt((x_end - x0)**2 + (y_end - y0)**2),
            'detected_lines': len(hough_peaks[0]) if hough_peaks else 0
        }
        
        return results
        
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return None

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
            crop_params={'x_start': 200, 'y_start': 150, 'width': 400, 'height': 300},
            rotation_angle=51,
            radiation_start=(100, 100),
            threshold=200
        )
        
        if results:
            print("\nAnalysis Results:")
            print(f"Image: {results['image_path']}")
            print(f"Radiation path: {results['radiation_start']} -> {results['radiation_end']}")
            print(f"Radiation length: {results['radiation_length']:.2f} pixels")
            print(f"Detected lines: {results['detected_lines']}")
        
        # Test with first image
        image_files = load_image_files(Config.IMAGE_FOLDER_PATH)
        if image_files:
            analyze_radiation_image(image_files[0], debug=True)
        
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
