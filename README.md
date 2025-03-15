# Scrawl - Advanced Handwriting OCR

Scrawl is a powerful handwriting recognition tool that leverages Google's Gemini Vision API to extract text from handwritten documents, combining advanced image processing with state-of-the-art OCR capabilities.

## ✨ Features

- Advanced image preprocessing for optimal text extraction
- Support for multiple file formats (JPG, PNG, PDF)
- Intelligent handwriting enhancement
- Spelling correction and text formatting
- Clean, modern web interface built with Streamlit
- PDF multi-page support
- Real-time text extraction and processing

## 🚀 Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/scrawl.git
cd scrawl
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create environment variables:
- Copy .env.example to .env
- Add your Google API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

##🔧 Usage
1. Start the application:
```bash
streamlit run app.py
```
2. Open your browser and navigate to the provided URL
3. Upload your document (supported formats: JPG, JPEG, PNG, PDF)
4. Click "Extract Text" to process the document

##⚙️ Technical Details
Scrawl uses several advanced techniques for optimal text extraction:

- Adaptive image enhancement
- Dynamic thresholding
- Noise reduction
- Character stroke improvement
- Smart text formatting and correction

##🔑 Requirements
- Python 3.8+
- Google Gemini API key
- Streamlit
- OpenCV
- NumPy
- PIL

##🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.