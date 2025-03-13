import easyocr
import numpy as np

class OCRHandler:
    def __init__(self):
        # Initialize EasyOCR with English language
        self.reader = easyocr.Reader(['en'])
    
    def extract_text(self, image):
        """Extract text from image"""
        try:
            # Perform OCR
            results = self.reader.readtext(image)
            
            # Extract text and confidence scores
            extracted_data = []
            for (bbox, text, prob) in results:
                extracted_data.append({
                    'text': text,
                    'confidence': prob
                })
            
            return extracted_data
        except Exception as e:
            return f"Error in OCR: {str(e)}"