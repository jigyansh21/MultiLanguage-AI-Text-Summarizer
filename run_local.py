#!/usr/bin/env python3
"""
Local Development Script for MultiLanguage AI Text Summarizer
This script is ONLY for local development and does not affect Render deployment
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
    
    # Local development settings
    port = 8000
    host = "127.0.0.1"  # Local development uses 127.0.0.1
    
    # Set Windows compatibility environment variable
    if os.name == "nt":
        os.environ.setdefault("PYTORCH_JIT", "0")
    
    print("[INFO] Starting MultiLanguage AI Text Summarizer (Local Development)...")
    print(f"[INFO] Language selection: http://{host}:{port}/")
    print(f"[INFO] Dashboard: http://{host}:{port}/summarizer")
    print(f"[INFO] Health check: http://{host}:{port}/health")
    print("\n" + "="*50)
    
    # Start the app with reload enabled for local development
    # Note: reload mode requires import string, not app object
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,  # Enable reload for local development
        log_level="info"
    )

