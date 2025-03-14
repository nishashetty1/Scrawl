import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from pdf2image import convert_from_path
from image_processor import HandwritingEnhancer
from ocr_handler import EnhancedOCRHandler
import gc
import psutil
from dotenv import load_dotenv
load_dotenv()

class OptimizedApp:
    def __init__(self):
        """Initialize app with Gemini API and configuration"""
        self.setup_page()
        
        # Get API key from environment variable
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("Please set GOOGLE_API_KEY in your environment variables")
            st.stop()
        
        # Initialize components with lazy loading
        self._enhancer = None
        self._ocr = None
        self.api_key = api_key
        
        # Set configuration parameters
        self.max_image_dimension = 2048  # Max image dimension for processing
        self.max_pdf_pages = 10      # Max PDF pages to process
    
    @property
    def enhancer(self):
        """Lazy loading of HandwritingEnhancer"""
        if self._enhancer is None:
            self._enhancer = HandwritingEnhancer()
        return self._enhancer
    
    @property
    def ocr(self):
        """Lazy loading of OCR handler"""
        if self._ocr is None:
            self._ocr = EnhancedOCRHandler(self.api_key)
        return self._ocr
    
    def setup_page(self):
        """Configure Streamlit page with optimized settings"""
        st.set_page_config(
            page_title="Advanced OCR with Gemini",
            page_icon="✍️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
            <style>
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
                border: none;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stTextInput>div>div>input {
                background-color: #2E2E2E;
                color: white;
            }
            .result-box {
                background-color: #2E2E2E;
                padding: 15px;
                border-radius: 5px;
                margin: 8px 0;
            }
            .status-box {
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .memory-info {
                font-size: 0.8em;
                color: #888;
                margin-top: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def process_image(self, image):
        """Process single image with memory optimization"""
        try:
            # Resize image if too large
            h, w = image.shape[:2]
            if max(h, w) > self.max_image_dimension:
                scale = self.max_image_dimension / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            # Process in steps with memory clearing
            with st.spinner("Enhancing image..."):
                enhanced_image = self.enhancer.enhance_handwriting(image)
                gc.collect()
            
            with st.spinner("Extracting text..."):
                # Save enhanced image temporarily
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, enhanced_image)
                    results = self.ocr.process_document(tmp.name)
                    os.unlink(tmp.name)
                
                gc.collect()
            
            return enhanced_image, results
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            gc.collect()
            return None, None
    
    def display_results(self, results):
        """Display OCR results with formatting"""
        if results and 'extracted_text' in results:
            st.markdown("### 📝 Extracted Text")
            
            # Display text in result box
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            for text in results['extracted_text']:
                st.write(text)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display metadata
            if 'metadata' in results:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Total Words: {results['metadata']['total_words']}")
                with col2:
                    st.info(f"Total Lines: {results['metadata'].get('total_lines', 0)}")
                
                st.success(f"Text extracted using {results['metadata']['engine']}")
    
    def run(self):
        """Main application loop"""
        st.title("✍️ Advanced OCR with Gemini")
        
        # Sidebar
        with st.sidebar:
            st.markdown("### Settings")
            quality_setting = st.select_slider(
                "Image Quality",
                options=["Fast", "Balanced", "High Quality"],
                value="Balanced"
            )
            
            st.markdown("### About")
            st.info("""
            This application uses Google's Gemini Vision API for accurate
            text extraction from images and documents. Supports both
            handwritten and printed text.
            """)
        
        # Main content
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose an image or PDF",
            type=["jpg", "jpeg", "png", "pdf"]
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    self.process_pdf(uploaded_file)
                else:
                    self.process_single_image(uploaded_file)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            finally:
                gc.collect()
    
    def process_pdf(self, pdf_file):
        """Process PDF with memory optimization"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            
            try:
                # Convert PDF to images
                images = convert_from_path(
                    tmp.name,
                    fmt="jpeg",
                    size=(self.max_image_dimension, self.max_image_dimension)
                )
                
                # Check page limit
                if len(images) > self.max_pdf_pages:
                    st.warning(f"Only the first {self.max_pdf_pages} pages will be processed.")
                    images = images[:self.max_pdf_pages]
                
                # Process each page
                for i, image in enumerate(images):
                    st.markdown(f"### Page {i+1}")
                    
                    # Convert PIL Image to OpenCV format
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Process page
                    self.process_single_image_content(opencv_image, f"page_{i+1}")
                    
                    gc.collect()
                    
            finally:
                os.unlink(tmp.name)
    
    def process_single_image(self, uploaded_file):
        """Process single uploaded image"""
        image = self._read_image(uploaded_file)
        self.process_single_image_content(image, "image")
    
    def process_single_image_content(self, image, key_suffix):
        """Process and display single image content"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Document**")
            st.image(image, use_column_width=True)
        
        process_button = st.button(
            "Extract Text",
            key=f"process_{key_suffix}"
        )
        
        if process_button:
            with st.spinner("Processing document..."):
                enhanced_image, results = self.process_image(image)
                
                if enhanced_image is not None:
                    with col2:
                        st.markdown("**Enhanced Document**")
                        st.image(enhanced_image, use_column_width=True)
                    
                    self.display_results(results)
    
    def _read_image(self, uploaded_file):
        """Read uploaded image with memory optimization"""
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    app = OptimizedApp()
    app.run()