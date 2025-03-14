import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import os
import io
import tempfile
import gc

class EnhancedOCRHandler:
    def __init__(self, api_key):
        """
        Initialize Gemini API for OCR and text formatting
        """
        try:
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Get Gemini 2.0 Flash model
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Combined prompt for OCR and formatting
            self.prompt = """
            Extract and format all text from this image accurately.

            Guidelines:
            1. Extract all visible text exactly as written
            2. Format the text into proper paragraphs
            3. Fix any obvious spelling errors
            4. Maintain original meaning and structure
            5. Keep all section headers or titles
            6. Preserve question-answer format if present
            7. Keep lists and bullet points if present
            8. Remove unnecessary line breaks
            9. Keep all original punctuation unless clearly incorrect
            10. Don't add or remove any information
            11. DO NOT include any prefix or introduction text
            12. Start directly with the extracted content

            Provide corrections separately after the main text (if any).
            """
            
        except Exception as e:
            raise Exception(f"Error initializing Gemini API: {str(e)}")

    def process_document(self, input_path):
        """
        Process document with enhanced text formatting
        """
        try:
            # Read and process image
            image = self._prepare_image(input_path)
            
            # Get response from Gemini
            response = self.model.generate_content(
                [self.prompt, image],
                generation_config={
                    'temperature': 0,
                    'top_p': 1,
                    'top_k': 1,
                }
            )
            
            # Process and format the response
            results = self._format_results(response.text)
            
            # Clear memory
            gc.collect()
            
            return results
            
        except Exception as e:
            gc.collect()
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
            
            # Resize if needed
            max_size = 4096  # Maximum size for Gemini 2.0
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            raise Exception(f"Error preparing image: {str(e)}")

    def _format_results(self, response_text):
        """
        Format the response from Gemini
        """
        try:
            # Split response into sections (formatted text and corrections)
            sections = response_text.split('\n\n')
            
            # Extract formatted text and corrections
            formatted_text = []
            corrections = []
            current_section = 'text'
            
            for section in sections:
                if 'corrections' in section.lower() or 'changes' in section.lower():
                    current_section = 'corrections'
                    continue
                
                if current_section == 'text':
                    formatted_text.extend(line for line in section.split('\n') if line.strip())
                else:
                    corrections.extend(line.strip('- ').strip() for line in section.split('\n') if line.strip('- ').strip())
            
            # Create results dictionary
            results = {
                'extracted_text': formatted_text,
                'original_text': formatted_text.copy(),  # Keep a copy of the formatted text
                'details': [
                    {
                        'text': line,
                        'source': 'gemini-2.0-flash',
                        'confidence': 0.98
                    }
                    for line in formatted_text if line.strip()
                ],
                'metadata': {
                    'total_words': sum(len(line.split()) for line in formatted_text),
                    'total_lines': len(formatted_text),
                    'engine': 'Google Gemini 2.0 Flash',
                    'corrections_made': corrections,
                    'status': 'success'
                }
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Error formatting results: {str(e)}")

    def _clean_text(self, text):
        """
        Clean and normalize text
        """
        if not text:
            return ''
        
        # Remove unwanted characters while preserving meaningful punctuation
        text = ''.join(char for char in text if char.isprintable())
        text = ' '.join(text.split())
        
        return text.strip()