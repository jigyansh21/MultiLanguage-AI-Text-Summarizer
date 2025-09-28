# Hindi LLM Summarizer - Render Deployment Guide

This guide will help you deploy the Hindi LLM Summarizer application on Render.

## Prerequisites

1. A Render account (sign up at [render.com](https://render.com))
2. Your code pushed to a Git repository (GitHub, GitLab, or Bitbucket)

## Files Created for Deployment

The following files have been created/updated for Render deployment:

- `render.yaml` - Render service configuration
- `Procfile` - Process file for Render
- `runtime.txt` - Python version specification
- `start.sh` - Production startup script
- `.gitignore` - Git ignore file
- `requirements.txt` - Updated with production dependencies
- `main.py` - Updated to work with Render's environment

## Deployment Steps

### 1. Push to Git Repository

First, commit and push all changes to your Git repository:

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### 2. Deploy on Render

1. **Login to Render Dashboard**
   - Go to [render.com](https://render.com) and sign in

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your Git repository

3. **Configure Service Settings**
   - **Name**: `hindi-llm-summarizer` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: Start with "Starter" plan (free tier)

4. **Environment Variables**
   - `PYTHON_VERSION`: `3.11.0`
   - `PORT`: `10000` (Render will override this)

5. **Advanced Settings**
   - **Health Check Path**: `/health`
   - **Auto Deploy**: Enable (optional)

### 3. Deploy

Click "Create Web Service" and Render will:
- Clone your repository
- Install dependencies
- Start your application
- Provide a public URL

## Application Features

Once deployed, your application will have:

- **Home Page**: Language selection interface
- **Summarizer Dashboard**: Main summarization interface
- **API Endpoints**:
  - `/api/summarize/text` - Text summarization
  - `/api/summarize/url` - URL content summarization
  - `/api/summarize/pdf` - PDF file summarization
  - `/api/summarize/youtube` - YouTube video summarization
  - `/api/export/pdf` - Export summary as PDF
  - `/api/export/word` - Export summary as Word document
  - `/api/export/markdown` - Export summary as Markdown
- **Health Check**: `/health`

## Configuration Details

### render.yaml
```yaml
services:
  - type: web
    name: hindi-llm-summarizer
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 10000
    healthCheckPath: /health
    autoDeploy: true
    disk:
      name: hindi-llm-summarizer-disk
      mountPath: /opt/render/project/src
      sizeGB: 1
```

### Key Changes Made

1. **main.py**: Updated to use `0.0.0.0` host and `PORT` environment variable
2. **requirements.txt**: Added production dependencies
3. **render.yaml**: Complete Render service configuration
4. **Procfile**: Process file for Render
5. **runtime.txt**: Python version specification

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version compatibility

2. **Application Won't Start**
   - Verify the start command in `main.py`
   - Check logs in Render dashboard

3. **Memory Issues**
   - The T5 model loading might require more memory
   - Consider upgrading to a higher plan if needed

4. **File Upload Issues**
   - Ensure proper file handling in the API endpoints
   - Check temporary file cleanup

### Monitoring

- Use Render's built-in logging to monitor application health
- Check the `/health` endpoint for basic health status
- Monitor memory usage in the Render dashboard

## Scaling

- **Starter Plan**: Good for testing and small usage
- **Standard Plan**: Better for production with more resources
- **Pro Plan**: For high-traffic applications

## Security Considerations

- The application runs in a sandboxed environment
- File uploads are handled securely with temporary files
- No sensitive data is stored permanently

## Support

For issues with:
- **Render Platform**: Check Render documentation
- **Application Code**: Review the application logs
- **Dependencies**: Verify `requirements.txt` compatibility

## Next Steps

After successful deployment:
1. Test all functionality through the web interface
2. Test API endpoints using tools like Postman
3. Monitor performance and usage
4. Consider upgrading plans based on usage

Your Hindi LLM Summarizer is now ready for production use on Render! 🚀
