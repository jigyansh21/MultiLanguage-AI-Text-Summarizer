#!/bin/bash
# Production startup script for Render deployment

echo "🚀 Starting Hindi LLM Summarizer on Render..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Installing dependencies..."

# Install dependencies
pip install -r requirements.txt

echo "✅ Dependencies installed"
echo "🌐 Starting FastAPI server..."

# Start the application
python main.py
