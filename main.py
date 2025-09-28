#!/usr/bin/env python3
"""
MultiLanguage AI Text Summarizer
Main entry point for the application
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the FastAPI app
from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Render requires binding to 0.0.0.0
    
    print("🚀 Starting MultiLanguage AI Text Summarizer...")
    print(f"📱 Language selection: http://0.0.0.0:{port}/")
    print(f"📊 Dashboard: http://0.0.0.0:{port}/summarizer")
    print(f"🔍 Health check: http://0.0.0.0:{port}/health")
    print("\n" + "="*50)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
