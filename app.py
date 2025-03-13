import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from image_processor import HandwritingEnhancer
from ocr_handler import OCRHandler

# Set page config for dark theme
st.set_page_config(
    page_title="Handwriting OCR",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #2E2E2E;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

class App:
    def __init__(self):
        self.enhancer = HandwritingEnhancer()
        self.ocr = OCRHandler()
    
    def run(self):
        st.title("✍️ Handwriting Enhancement & OCR")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload an image with handwritten text",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Read image
            image = self._read_image(uploaded_file)
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Process image
            if st.button("Enhance and Extract Text"):
                # Enhance image
                enhanced_image = self.enhancer.enhance_handwriting(image)
                
                # Display enhanced image
                with col2:
                    st.subheader("Enhanced Image")
                    st.image(enhanced_image, use_column_width=True)
                
                # Perform OCR
                with st.spinner("Extracting text..."):
                    results = self.ocr.extract_text(enhanced_image)
                
                # Display results
                st.subheader("Extracted Text")
                if isinstance(results, list):
                    for item in results:
                        st.write(f"Text: {item['text']}")
                        st.write(f"Confidence: {item['confidence']:.2%}")
                        st.markdown("---")
                else:
                    st.error(results)
    
    def _read_image(self, uploaded_file):
        """Read uploaded image file"""
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    app = App()
    app.run()