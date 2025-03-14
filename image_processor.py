import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_local
from skimage.morphology import thin
import math

class AdvancedImageProcessor:
    """
    Advanced image processor specifically designed for handwriting enhancement
    with focus on stroke quality and character separation.
    """
    
    def __init__(self):
        # Kernels for different operations
        self.sharpening_kernel = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                         [-1,-1,-1]])
        
        self.stroke_kernel = np.ones((2,2), np.uint8)
        self.text_kernel = np.ones((3,3), np.uint8)
        
    def process_image(self, image):
        """Main processing pipeline for handwritten text enhancement."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. Initial Enhancement
            normalized = self._normalize_image(gray)
            denoised = self._advanced_denoising(normalized)
            
            # 2. Adaptive Thresholding
            binary = self._adaptive_thresholding(denoised)
            
            # 3. Character Enhancement
            enhanced = self._enhance_characters(binary)
            
            # 4. Stroke Improvement
            improved = self._improve_strokes(enhanced)
            
            # 5. Final Clean-up
            final = self._final_cleanup(improved)
            
            return final
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def _normalize_image(self, image):
        """
        Normalize image with advanced contrast enhancement.
        """
        # Calculate optimal alpha and beta for contrast stretching
        min_val = np.percentile(image, 5)
        max_val = np.percentile(image, 95)
        
        # Apply contrast stretching
        normalized = np.clip((image - min_val) * (255.0 / (max_val - min_val)), 0, 255)
        normalized = normalized.astype(np.uint8)
        
        # Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(normalized)
        
        return normalized
    
    def _advanced_denoising(self, image):
        """
        Advanced denoising with edge preservation.
        """
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(image, 
                                           None,
                                           h=10,
                                           templateWindowSize=7,
                                           searchWindowSize=21)
        
        # Bilateral filter for edge preservation
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        return denoised
    
    def _adaptive_thresholding(self, image):
        """
        Advanced adaptive thresholding with dynamic window size.
        """
        # Calculate dynamic window size based on image size
        height, width = image.shape
        window_size = int(min(height, width) * 0.02)
        if window_size % 2 == 0:
            window_size += 1
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(image,
                                     255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     window_size,
                                     2)
        
        return binary
    
    def _enhance_characters(self, image):
        """
        Enhance character shapes and connections.
        """
        # Remove small noise
        min_size = 50  # Minimum character size
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, 
                                                                                 connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        
        img2 = np.zeros(output.shape, dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        
        # Connect broken character parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        connected = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
        
        return connected
    
    def _improve_strokes(self, image):
        """
        Improve stroke quality and consistency.
        """
        # Thin the strokes
        thinned = thin(image > 0).astype(np.uint8) * 255
        
        # Dilate to achieve consistent stroke width
        stroke_kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(thinned, stroke_kernel, iterations=1)
        
        # Smooth the strokes
        smooth_kernel = np.ones((3,3), np.uint8)
        smoothed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, smooth_kernel)
        
        return smoothed
    
    def _final_cleanup(self, image):
        """
        Final image cleanup and enhancement.
        """
        # Remove isolated pixels
        clean = cv2.medianBlur(image, 3)
        
        # Enhance edges
        edges = cv2.Laplacian(clean, cv2.CV_64F).astype(np.uint8)
        enhanced = cv2.addWeighted(clean, 1, edges, 0.5, 0)
        
        # Ensure good contrast
        _, enhanced = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        
        return enhanced

class HandwritingEnhancer:
    """
    Enhanced handwriting processor with better image quality preservation
    """
    
    def __init__(self):
        self.sharpening_kernel = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                         [-1,-1,-1]])
    
    def enhance_handwriting(self, image):
        """
        Enhanced processing pipeline for better text preservation
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. Initial Enhancement
            normalized = self._normalize_image(gray)
            denoised = self._remove_noise(normalized)
            
            # 2. Enhance Contrast
            enhanced = self._enhance_contrast(denoised)
            
            # 3. Binarization with better preservation
            binary = self._adaptive_binarize(enhanced)
            
            # 4. Clean and Sharpen
            cleaned = self._clean_and_sharpen(binary)
            
            return cleaned
            
        except Exception as e:
            raise Exception(f"Error enhancing image: {str(e)}")
    
    def _normalize_image(self, image):
        """
        Normalize image with better contrast preservation
        """
        # Perform histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(image)
        return normalized
    
    def _remove_noise(self, image):
        """
        Remove noise while preserving text
        """
        # Use bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def _enhance_contrast(self, image):
        """
        Enhance contrast while preserving text details
        """
        # Calculate optimal alpha and beta
        alpha = 1.5  # Contrast control
        beta = 10    # Brightness control
        
        # Apply contrast enhancement
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def _adaptive_binarize(self, image):
        """
        Improved adaptive thresholding for better text preservation
        """
        # Calculate dynamic window size
        height, width = image.shape
        window_size = int(min(height, width) * 0.02)
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(11, min(window_size, 31))
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # Changed from BINARY_INV for better visibility
            window_size,
            10
        )
        
        return binary
    
    def _clean_and_sharpen(self, image):
        """
        Clean and sharpen the image while preserving text
        """
        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Sharpen the image
        sharpened = cv2.filter2D(cleaned, -1, self.sharpening_kernel)
        
        # Ensure good contrast
        _, final = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
        
        return final