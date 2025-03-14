import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import os
import io

class EnhancedOCRHandler:
    def __init__(self, api_key):
        """
        Initialize Gemini API for OCR using Gemini 2.0 Flash
        
        Args:
            api_key (str): Google API key for Gemini
        """
        try:
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Get Gemini 2.0 Flash model
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Set up prompt for better OCR
            self.prompt = """
            Extract all text from this image accurately.
            Guidelines:
            - Include all visible text
            - Maintain the original formatting and structure
            - Preserve punctuation and special characters
            - Keep line breaks as they appear
            - Return the text exactly as shown in the image
            """
            
        except Exception as e:
            raise Exception(f"Error initializing Gemini API: {str(e)}")
    
    def process_document(self, input_path):
        """
        Process document using Gemini 2.0 Flash
        
        Args:
            input_path (str): Path to the image file
            
        Returns:
            dict: Extracted text and metadata
        """
        try:
            # Read and prepare image
            image = self._prepare_image(input_path)
            
            # Generate response from Gemini
            response = self.model.generate_content(
                [self.prompt, image],
                generation_config={
                    'temperature': 0,  # For consistent results
                    'top_p': 1,
                    'top_k': 1,
                },
                stream=False
            )
            
            # Format results
            results = self._format_results(response)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def _prepare_image(self, image_path):
        """
        Prepare image for Gemini API
        """
        try:
            # Read image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            elif isinstance(image_path, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Unsupported image format")
            
            # Resize if too large (Gemini has file size limits)
            max_size = 4096  # Increased for Gemini 2.0
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            raise Exception(f"Error preparing image: {str(e)}")
    
    def _format_results(self, response):
        """
        Format Gemini API response
        """
        try:
            if not response.text:
                return {
                    'extracted_text': [],
                    'details': [],
                    'metadata': {
                        'total_words': 0,
                        'status': 'No text detected'
                    }
                }
            
            # Split text into lines
            text_lines = response.text.strip().split('\n')
            text_lines = [line.strip() for line in text_lines if line.strip()]
            
            # Create details for each line
            details = [
                {
                    'text': line,
                    'source': 'gemini-2.0-flash',
                    'confidence': 0.98  # Gemini 2.0 typically has very high accuracy
                }
                for line in text_lines
            ]
            
            # Format final results
            results = {
                'extracted_text': text_lines,
                'details': details,
                'metadata': {
                    'total_words': sum(len(line.split()) for line in text_lines),
                    'total_lines': len(text_lines),
                    'engine': 'Google Gemini 2.0 Flash',
                    'status': 'success'
                }
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Error formatting results: {str(e)}")
    
    def _clean_text(self, text):
        """
        Clean and normalize extracted text
        """
        if not text:
            return ''
        
        # Remove unwanted characters while preserving meaningful punctuation
        text = ''.join(char for char in text if char.isprintable())
        
        # Normalize multiple spaces
        text = ' '.join(text.split())
        
        return text.strip()