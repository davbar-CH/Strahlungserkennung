#!/usr/bin/env python3
"""
Automated radiation detection using image processing.
This module processes images to detect and analyze radiation traces.
"""

import os
import glob
from typing import List, Tuple, Optional
import logging

import numpy as np
import cv2
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Point, LineString
from scipy import ndimage
from sklearn.decomposition import PCA
from pandas import DataFrame

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RadiationDetector:
    """Main class for radiation detection in images."""
    
    def __init__(self, folder_path: str, scaling_factor: float = 0.5, threshold: int = 220):
        """
        Initialize the radiation detector.
        
        Args:
            folder_path: Path to folder containing images
            scaling_factor: Factor to scale images for faster processing
            threshold: Threshold value for binary conversion
        """
        self.folder_path = folder_path
        self.scaling_factor = scaling_factor
        self.threshold = threshold
        self.crop_points = np.array([
            [970, 1550],   # top left
            [3000, 440],   # top right
            [4400, 2187],  # bottom right
            [2400, 3760]   # bottom left
        ], dtype=np.int32)
        
    def load_images(self, pattern: str = "*.JPG") -> List[str]:
        """
        Load image paths from the folder.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of image file paths
            
        Raises:
            FileNotFoundError: If no images are found
        """
        try:
            image_paths = glob.glob(os.path.join(self.folder_path, pattern))
            if not image_paths:
                raise FileNotFoundError(f"No images found matching pattern {pattern} in {self.folder_path}")
            
            logger.info(f"Found {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            raise
    
    def read_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (cropped_image, image_array)
            
        Raises:
            cv2.error: If image cannot be read
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise cv2.error(f"Could not read image: {image_path}")
            
            # Scale image for faster processing
            new_width = int(image.shape[1] * self.scaling_factor)
            new_height = int(image.shape[0] * self.scaling_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Scale crop points
            scaled_points = (self.crop_points * self.scaling_factor).astype(np.int32)
            scaled_points = scaled_points.reshape((-1, 1, 2))
            
            # Create mask for cropping
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [scaled_points], 255)
            
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask=mask)
            
            # Crop to bounding rectangle
            x, y, w, h = cv2.boundingRect(scaled_points)
            cropped = masked[y:y + h, x:x + w]
            
            return cropped, np.array(cropped)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def create_binary_image(self, image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert image to binary and label connected components.
        
        Args:
            image_array: Input image array
            
        Returns:
            Tuple of (labeled_array, grayscale_array, binary_array)
        """
        try:
            # Convert to grayscale
            grayscale = np.sum(image_array, axis=2).astype(np.int64)
            
            # Apply threshold
            binary = grayscale > self.threshold
            
            # Label connected components
            structure = ndimage.generate_binary_structure(2, 2)
            labeled = ndimage.label(binary, structure)
            
            return labeled, grayscale, binary
            
        except Exception as e:
            logger.error(f"Error creating binary image: {e}")
            raise
    
    def extract_radiation_traces(self, labeled_array: np.ndarray, min_points: int = 170) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """
        Extract radiation traces using PCA.
        
        Args:
            labeled_array: Labeled image array
            min_points: Minimum number of points required for PCA
            
        Returns:
            Tuple of (start_points, end_points, count)
        """
        try:
            start_points = []
            end_points = []
            count = 0
            
            # Get unique labels (excluding background)
            object_labels = np.unique(labeled_array[0])
            
            for label in object_labels[1:]:  # Skip background (label 0)
                # Get all points for this object
                object_points = np.argwhere(labeled_array[0] == label)
                
                if len(object_points) > min_points:
                    # Apply PCA to find main direction
                    pca = PCA(n_components=1)
                    pca.fit(object_points)
                    
                    direction = pca.components_[0]
                    center = pca.mean_
                    
                    # Calculate start and end points
                    start = center - 50 * direction
                    end = center + 50 * direction
                    
                    start_points.append(start)
                    end_points.append(end)
                    count += 1
            
            return start_points, end_points, count
            
        except Exception as e:
            logger.error(f"Error extracting radiation traces: {e}")
            raise
    
    def calculate_angles(self, start_points: List[np.ndarray], end_points: List[np.ndarray]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate angles between intersecting radiation traces.
        
        Args:
            start_points: List of trace start points
            end_points: List of trace end points
            
        Returns:
            Tuple of (secure_angles, possible_angles, all_angles)
        """
        try:
            secure_angles = []
            possible_angles = []
            all_angles = []
            
            for i in range(len(start_points)):
                line1 = LineString([start_points[i], end_points[i]])
                
                for j in range(i + 1, len(start_points)):
                    line2 = LineString([start_points[j], end_points[j]])
                    
                    # Check if lines intersect within buffer
                    buffer1 = line1.buffer(10)
                    buffer2 = line2.buffer(10)
                    
                    if buffer1.intersects(buffer2):
                        # Calculate angle between vectors
                        vector1 = np.array(end_points[i]) - np.array(start_points[i])
                        vector2 = np.array(end_points[j]) - np.array(start_points[j])
                        
                        # Calculate angle using dot product
                        norm1 = np.linalg.norm(vector1)
                        norm2 = np.linalg.norm(vector2)
                        
                        if norm1 > 0 and norm2 > 0:
                            cos_angle = np.clip(np.dot(vector1, vector2) / (norm1 * norm2), -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(cos_angle))
                            
                            all_angles.append(angle_deg)
                            
                            if angle_deg > 10:  # Filter out very small angles
                                possible_angles.append(angle_deg)
                            else:
                                secure_angles.append(angle_deg)
            
            return secure_angles, possible_angles, all_angles
            
        except Exception as e:
            logger.error(f"Error calculating angles: {e}")
            return [], [], []
    
    def visualize_results(self, original: np.ndarray, grayscale: np.ndarray, binary: np.ndarray, 
                         start_points: List[np.ndarray], end_points: List[np.ndarray], 
                         count: int, angles: List[float]) -> None:
        """
        Visualize the processing results.
        
        Args:
            original: Original cropped image
            grayscale: Grayscale image
            binary: Binary image
            start_points: Trace start points
            end_points: Trace end points
            count: Number of detected traces
            angles: Calculated angles
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            # Original image
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # Grayscale image
            axes[1].imshow(grayscale, cmap="gray")
            axes[1].set_title("Grayscale Image")
            axes[1].axis("off")
            
            # Binary image
            axes[2].imshow(binary, cmap="gray")
            axes[2].set_title("Binary Image (After Threshold)")
            axes[2].axis("off")
            
            # Detected traces
            axes[3].imshow(grayscale, cmap="gray")
            axes[3].set_title(f"Detected Traces (Count: {count})")
            axes[3].axis("off")
            
            # Draw traces
            for start, end in zip(start_points, end_points):
                axes[3].plot([start[1], end[1]], [start[0], end[0]], "r-", linewidth=2)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
    
    def process_single_image(self, image_path: str, visualize: bool = False) -> Tuple[int, List[float], List[float]]:
        """
        Process a single image and return results.
        
        Args:
            image_path: Path to the image file
            visualize: Whether to show visualization
            
        Returns:
            Tuple of (count, secure_angles, possible_angles)
        """
        try:
            logger.info(f"Processing image: {os.path.basename(image_path)}")
            
            # Read and preprocess
            cropped, image_array = self.read_and_preprocess_image(image_path)
            
            # Create binary image
            labeled, grayscale, binary = self.create_binary_image(image_array)
            
            # Extract traces
            start_points, end_points, count = self.extract_radiation_traces(labeled)
            
            # Calculate angles
            secure_angles, possible_angles, all_angles = self.calculate_angles(start_points, end_points)
            
            # Visualize if requested
            if visualize:
                self.visualize_results(cropped, grayscale, binary, start_points, end_points, count, all_angles)
            
            return count, secure_angles, possible_angles
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return 0, [], []
    
    def process_all_images(self, pattern: str = "*.JPG", visualize: bool = False) -> DataFrame:
        """
        Process all images in the folder.
        
        Args:
            pattern: File pattern to match
            visualize: Whether to show visualizations
            
        Returns:
            DataFrame with results
        """
        try:
            # Load images
            image_paths = self.load_images(pattern)
            
            # Process each image
            counts = []
            for image_path in image_paths:
                count, _, _ = self.process_single_image(image_path, visualize)
                counts.append(count)
            
            # Create results DataFrame
            results = DataFrame({
                "Image": [os.path.basename(path) for path in image_paths],
                "Radiation_Count": counts
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing all images: {e}")
            return DataFrame()

def main():
    """Main function to run the radiation detection."""
    # Configuration
    folder_path = "/Users/Dimitri.Baragunoff/Repos/Strahlungserkennung"  # Update this path
    
    # Create detector instance
    detector = RadiationDetector(folder_path)
    
    try:
        # Process specific image for testing
        image_pattern = "DSC_0367.JPG"  # Adjust as needed
        results = detector.process_all_images(pattern=image_pattern, visualize=True)
        
        # Save results
        if not results.empty:
            output_file = "radiation_detection_results.xlsx"
            results.to_excel(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            print(results)
        else:
            logger.warning("No results to save")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
