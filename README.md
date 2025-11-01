# MultiLanguage AI Text Summarizer

A powerful, modular **NLP/AI web application** for summarizing text content in both Hindi and English using advanced AI models and extractive summarization techniques. The application supports multiple input sources including manual text, URLs, PDF files, and YouTube videos.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![NLP](https://img.shields.io/badge/NLP-AI%20Summarization-purple.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎥 Demo Videos

Watch comprehensive demo videos showcasing all features:
- **[View All Demo Videos](demo/README.md)** - Complete feature walkthroughs in Hindi and English

### 📹 Quick Demo Links
- [Hindi Manual Text Summarizer](https://drive.google.com/file/d/1IbJHacPmgr7GXPtOeTjog614FrjgUszM/view?usp=sharing)
- [Hindi PDF Summarizer](https://drive.google.com/file/d/1LnNu02fIEnUgeIGi_ciqeJKyKx6XNR21/view?usp=sharing)
- [Hindi URL Summarizer](https://drive.google.com/file/d/1gl1nA52BxZ6w9r0u2IJMxHfOunm0nGlL/view?usp=sharing)
- [Hindi YouTube Summarizer](https://drive.google.com/file/d/1ecRI44_JBxtFXTILyw08eFepzYAHwwPZ/view?usp=sharing)
- [English Manual Text Summarizer](https://drive.google.com/file/d/1q48gUMXjXlIwETXZqI-0So8DmMyASoEa/view?usp=sharing)
- [English PDF Summarizer](https://drive.google.com/file/d/1ItyLX2EuNYVPLjMgFXw8BqQ87cvTLFdT/view?usp=sharing)
- [English URL Summarizer](https://drive.google.com/file/d/1ItyLX2EuNYVPLjMgFXw8BqQ87cvTLFdT/view?usp=sharing)
- [English YouTube Summarizer](https://drive.google.com/file/d/1WGPt5PjryRP6T5yWcurTPxGPalZ12Wi0/view?usp=sharing)

## ✨ Features

### 🎯 Core Functionality
- **Multi-language Support**: Hindi and English summarization
- **Multiple Input Types**: Text, URL, PDF, and YouTube video processing
- **Flexible Summary Lengths**: Short, Medium, Long, and Auto modes
- **Export Options**: PDF, Word, and Markdown formats
- **Copy to Clipboard**: Easy sharing functionality

### 🎨 User Interface
- **Modern Design**: Clean, responsive interface with TailwindCSS
- **Dark/Light Theme**: Toggle between themes with persistent preferences
- **Professional UI/UX**: Smooth animations and intuitive navigation
- **Mobile Responsive**: Works seamlessly on all devices

### 🚀 Technical Features
- **FastAPI Backend**: High-performance async API
- **Modular Architecture**: Clean, maintainable code structure
- **AI Integration**: T5 transformer model with extractive fallback
- **Error Handling**: Comprehensive error management and user feedback

## 🏗️ Project Structure

```
MultiLanguage-AI-Text-Summarizer/
├── src/                          # Source code
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI application
│   ├── core/                     # Core business logic
│   │   ├── __init__.py
│   │   └── summarizer.py        # Summarization service
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── pdf_utils.py         # PDF processing
│       └── youtube_utils.py     # YouTube processing
├── templates/                    # HTML templates
│   ├── index.html               # Language selection
│   ├── summarizer.html          # Main dashboard
│   └── result.html              # Results display
├── static/                       # Static assets
│   └── styles.css               # Custom CSS
├── fonts/                        # Font files
│   └── NotoSansDevanagari-Regular.ttf
├── demo/                         # Demo videos and documentation
│   └── README.md                # Demo video links and descriptions
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

### 3. Access the Application
- **Language Selection**: http://127.0.0.1:8000/
- **Main Dashboard**: http://127.0.0.1:8000/summarizer
- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## 📖 Usage Guide

### Language Selection
1. Choose your preferred language (Hindi or English)
2. Click on the language card to proceed to the dashboard

### Text Summarization
1. **Manual Text**: Paste your text in the text area
2. **URL Content**: Enter a web URL to extract and summarize content
3. **PDF Upload**: Upload a PDF file (max 15 pages)
4. **YouTube Video**: Enter a YouTube URL to extract transcript and summarize

### Summary Options
- **Short**: Concise summary (20-50 words)
- **Medium**: Balanced summary (40-100 words)
- **Long**: Detailed summary (80-200 words)
- **Auto**: Automatically determines optimal length

### Export Options
- **PDF**: Download as formatted PDF document
- **Word**: Export as Microsoft Word document
- **Markdown**: Save as Markdown file
- **Copy**: Copy to clipboard for easy sharing

## 🔧 Configuration

### Environment Variables
- `PYTORCH_JIT=0` - Disable PyTorch JIT for Windows compatibility

### Font Support
- Hindi fonts are located in `fonts/NotoSansDevanagari-Regular.ttf`
- Automatic fallback to Arial if Hindi font not found

## 🎨 Theme Customization

The application supports both light and dark themes:
- **Theme Toggle**: Click the sun/moon icon in the top-right corner
- **Persistent**: Theme preference is saved in browser localStorage
- **Automatic**: Theme persists across page refreshes and navigation

## 📡 API Documentation

### Endpoints

#### Summarization
- `POST /api/summarize/text` - Summarize raw text
- `POST /api/summarize/url` - Extract and summarize URL content
- `POST /api/summarize/pdf` - Upload and summarize PDF file
- `POST /api/summarize/youtube` - Extract transcript and summarize YouTube video

#### Export
- `POST /api/export/pdf` - Export summary as PDF
- `POST /api/export/word` - Export summary as Word document
- `POST /api/export/markdown` - Export summary as Markdown

#### Utility
- `GET /health` - Health check endpoint

### Request/Response Format

All API endpoints return JSON responses with the following structure:

```json
{
  "summary": "Generated summary text",
  "original_length": 500,
  "summary_length": 100,
  "compression_ratio": 0.2,
  "processing_time": 2.3,
  "title": "Document Title"
}
```

## 🛠️ Technical Details

### Backend Architecture
- **FastAPI**: Modern, fast web framework for building APIs
- **Async/Await**: Non-blocking I/O for better performance
- **Pydantic**: Data validation and serialization
- **Jinja2**: Template engine for HTML rendering

### AI/ML Components (Core NLP Features)
- **Extractive Summarization**: Advanced text processing algorithms
- **Multi-language NLP**: Hindi and English text processing
- **Text Preprocessing**: Tokenization, cleaning, and normalization
- **Intelligent Fallback**: Robust error handling and alternative methods

### Frontend Technologies
- **TailwindCSS**: Utility-first CSS framework
- **JavaScript**: Vanilla JS for interactivity
- **Responsive Design**: Mobile-first approach
- **Theme System**: CSS custom properties for theming

## 🚀 Performance (NLP Optimized)

- **High-Speed Processing**: <0.01s processing time for most documents
- **Efficient Compression**: 90-99% text compression ratio
- **Async Processing**: Non-blocking operations for better performance
- **Chunked Processing**: Large texts are processed in chunks
- **Progress Tracking**: Real-time updates during processing
- **Error Handling**: Graceful fallbacks and user feedback

## 🔒 Security

- **Input Validation**: All inputs are validated and sanitized
- **File Type Validation**: Only allowed file types are processed
- **Size Limits**: PDF files limited to 15 pages
- **Error Handling**: Secure error messages without exposing internals

## 🎯 Development

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup Development Environment
```bash
# Clone the repository
git clone <repository-url>
cd MultiLanguage-AI-Text-Summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Code Structure
- **Modular Design**: Clear separation of concerns
- **Type Hints**: Full type annotation for better code quality
- **Error Handling**: Comprehensive exception management
- **Documentation**: Detailed docstrings and comments

## 📝 License

This project is created by Jigyansh ECE Undergraduate, Thapar Institute of Engineering Technology, Patiala.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions, please contact the development team.

---

**Created by Jigyansh, ECE Undergraduate, Thapar Institute of Engineering Technology, Patiala**