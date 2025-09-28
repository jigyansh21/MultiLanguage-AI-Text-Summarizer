#!/usr/bin/env python3
"""
Test script to verify deployment configuration
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test FastAPI imports
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        print("✅ FastAPI imports successful")
        
        # Test core modules
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from src.core.summarizer import SummarizerService
        from src.utils.pdf_utils import PDFProcessor
        from src.utils.youtube_utils import YouTubeProcessor
        print("✅ Core modules import successful")
        
        # Test AI libraries
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        print("✅ AI libraries import successful")
        
        # Test other dependencies
        import uvicorn
        import jinja2
        print("✅ Other dependencies import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🌍 Testing environment...")
    
    # Test port configuration
    port = os.environ.get("PORT", "8000")
    print(f"✅ PORT environment variable: {port}")
    
    # Test Python version
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "render.yaml",
        "Procfile",
        "runtime.txt",
        "src/api/main.py",
        "src/core/summarizer.py",
        "templates/index.html",
        "templates/summarizer.html",
        "templates/result.html",
        "static/styles.css"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Hindi LLM Summarizer - Deployment Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_environment,
        test_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment on Render.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
