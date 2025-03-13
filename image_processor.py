import cv2
import numpy as np
from skimage import filters
from scipy import ndimage
from skimage.morphology import skeletonize

class ImagePreprocessor:
    def __init__(self):
        self.kernel_sharpening = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                         [-1,-1,-1]])
    
    def enhance_image(self, image):
        """Main function to enhance handwritten image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing steps
        denoised = self._remove_noise(gray)
        enhanced = self._enhance_contrast(denoised)
        binary = self._binarize(enhanced)
        cleaned = self._clean_image(binary)
        
        return cleaned
    
    def _remove_noise(self, image):
        """Remove noise from image"""
        # Apply bilateral filter to remove noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def _enhance_contrast(self, image):
        """Enhance image contrast"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Apply sharpening
        sharpened = cv2.filter2D(enhanced, -1, self.kernel_sharpening)
        return sharpened
    
    def _binarize(self, image):
        """Convert image to binary using improved thresholding."""
        # Use Otsu's thresholding for better results
        _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        # Also try adaptive thresholding
        binary_adaptive = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Larger block size for better handling of handwriting
            4    # Slightly higher C for better separation
        )
    
        # Combine the results for better overall quality
        binary = cv2.bitwise_and(binary_otsu, binary_adaptive)
    
        return binary

    def _clean_image(self, image):
        """Clean up the binary image with improved techniques."""
        # Remove small noise
        kernel1 = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1)
    
        # Close small gaps in characters
        kernel2 = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    
        # Remove isolated pixels
        cleaned = cv2.medianBlur(closing, 3)
    
        return cleaned

class HandwritingEnhancer:
    """A class for enhancing handwritten text in images.
    
    This class builds on the ImagePreprocessor to provide specialized
    enhancements for handwritten text clarity and readability.
    """
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
    
    def enhance_handwriting(self, image):
        """Enhance handwriting in the image using improved techniques.
        
        Args:
            image: Input image as numpy array (grayscale or BGR)
            
        Returns:
            np.ndarray: Enhanced handwriting image
        """
        # Initial preprocessing
        processed = self.preprocessor.enhance_image(image)
        
        # Apply advanced enhancement techniques
        thinned = self._thin_strokes(processed)
        smoothed = self._smooth_strokes(thinned)
        enhanced = self._enhance_contrast(smoothed)
        
        return enhanced
    
    def _thin_strokes(self, image):
        """Thin strokes while preserving continuity.
        
        Unlike skeletonization, this maintains stroke quality.
        """
        # Create structuring elements
        kernel = np.ones((2, 2), np.uint8)
        
        # Apply erosion with small kernel to preserve details
        eroded = cv2.erode(image, kernel, iterations=1)
        
        # Recover important stroke connections
        kernel2 = np.ones((3, 3), np.uint8)
        thinned = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel2)
        
        return thinned
    
    def _smooth_strokes(self, image):
        """Smooth jagged edges in handwritten strokes."""
        # Apply Gaussian blur with small kernel
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Re-threshold to maintain binary image
        _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        return smoothed
    
    def _enhance_contrast(self, image):
        """Enhance the final image contrast."""
        # Create a larger kernel for morphological operations
        kernel = np.ones((2, 2), np.uint8)
        
        # Apply dilation to make text more visible
        dilated = cv2.dilate(image, kernel, iterations=1)
        
        # Apply closing to fill small gaps in characters
        result = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        return result
