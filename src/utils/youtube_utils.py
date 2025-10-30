"""
YouTube Processing Utilities
Handles YouTube transcript extraction and summarization
"""

import asyncio
import re
from typing import Dict, Any, Literal, Optional
from urllib.parse import urlparse, parse_qs

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    print("youtube-transcript-api not installed. YouTube functionality will be limited.")
    YouTubeTranscriptApi = None
    TextFormatter = None
    YOUTUBE_API_AVAILABLE = False

class YouTubeProcessor:
    def __init__(self):
        self.formatter = TextFormatter() if TextFormatter else None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        try:
            # Handle various YouTube URL formats
            patterns = [
                r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
                r'youtube\.com\/v\/([^&\n?#]+)',
                r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
            
            return None
            
        except Exception:
            return None
    
    async def get_transcript(self, video_id: str, language: str = "en") -> str:
        """Get transcript for YouTube video using the new API"""
        if not YOUTUBE_API_AVAILABLE or not YouTubeTranscriptApi:
            raise Exception("YouTube transcript API not available. Please install youtube-transcript-api: pip install youtube-transcript-api")
        
        try:
            # Create API instance
            api = YouTubeTranscriptApi()
            
            # List available transcripts
            try:
                transcript_list = api.list(video_id)
                available_transcripts = list(transcript_list)
                
                print(f"Available transcripts: {len(available_transcripts)}")
                for i, transcript in enumerate(available_transcripts):
                    print(f"  {i}: {transcript.language_code} - {transcript.language}")
                
                if not available_transcripts:
                    raise Exception("This video doesn't have any captions/transcripts available. Please try a different video that has captions enabled.")
                
                # Try to get transcript in specified language first
                try:
                    # Find transcript in the specified language
                    transcript = None
                    for t in available_transcripts:
                        if t.language_code == language:
                            transcript = t
                            break
                    
                    if transcript:
                        transcript_data = transcript.fetch()
                        print(f"Successfully got transcript in {transcript.language_code}")
                    else:
                        # Try to find a close match (e.g., 'en' for 'en-US')
                        for t in available_transcripts:
                            if t.language_code.startswith(language.split('-')[0]):
                                transcript = t
                                break
                        
                        if transcript:
                            transcript_data = transcript.fetch()
                            print(f"Successfully got transcript in {transcript.language_code} (fallback)")
                        else:
                            raise Exception(f"No transcript found for language {language}")
                            
                except Exception as lang_error:
                    print(f"Failed to get transcript in {language}: {lang_error}")
                    # Fallback to any available language
                    try:
                        # Try to get any available transcript
                        transcript = available_transcripts[0]
                        transcript_data = transcript.fetch()
                        print(f"Successfully got transcript in {transcript.language_code} (any language)")
                    except Exception as fallback_error:
                        print(f"Failed to get any available transcript: {fallback_error}")
                        raise Exception("This video doesn't have captions/transcripts available. Please try a different video that has captions enabled.")
            
            except Exception as list_error:
                print(f"Failed to list transcripts: {list_error}")
                raise Exception("This video doesn't have captions/transcripts available. Please try a different video that has captions enabled.")
            
            if not transcript_data:
                raise Exception("No transcript available for this video. Please ensure the video has captions enabled.")
            
            # Format transcript as plain text
            if self.formatter:
                formatted_text = self.formatter.format_transcript(transcript_data)
            else:
                # Manual formatting if formatter not available
                formatted_text = " ".join([entry['text'] for entry in transcript_data])
            
            return formatted_text
            
        except Exception as e:
            # Only catch specific "no captions" errors, let other errors pass through
            error_msg = str(e)
            if any(phrase in error_msg.lower() for phrase in [
                "no transcripts were found",
                "could not retrieve a transcript",
                "no element found",
                "transcript not found",
                "captions/transcripts available"
            ]):
                raise Exception("This video doesn't have captions/transcripts available. Please try a different video that has captions enabled.")
            elif "video unavailable" in error_msg.lower():
                raise Exception("This video is unavailable or private. Please try a different video.")
            elif "quota exceeded" in error_msg.lower():
                raise Exception("YouTube API quota exceeded. Please try again later.")
            else:
                # For other errors, re-raise the original error
                raise e
    
    async def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get basic video information"""
        try:
            # This is a simplified version - in production you might want to use
            # the YouTube Data API for more detailed information
            return {
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "YouTube Video",  # Would need YouTube Data API for actual title
                "duration": "Unknown"  # Would need YouTube Data API for actual duration
            }
        except Exception as e:
            return {
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "Unknown Video",
                "error": str(e)
            }
    
    async def summarize_video(
        self, 
        url: str, 
        language: Literal["hindi", "english"] = "hindi",
        summary_length: Literal["short", "medium", "long", "auto"] = "auto"
    ) -> Dict[str, Any]:
        """Extract transcript and summarize YouTube video, with universal chunking for long transcripts."""
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                raise Exception("Invalid YouTube URL. Please provide a valid YouTube video URL.")
            # Validate URL format
            if not self.is_valid_youtube_url(url):
                raise Exception("Invalid YouTube URL format. Please provide a valid YouTube video URL.")
            # Get video info
            video_info = await self.get_video_info(video_id)
            # Get transcript
            transcript_language = "hi" if language == "hindi" else "en"
            try:
                transcript = await self.get_transcript(video_id, transcript_language)
            except Exception as transcript_error:
                # Only modify specific "no captions" errors, let other errors pass through
                error_msg = str(transcript_error)
                if "captions/transcripts available" in error_msg:
                    raise Exception("This video doesn't have captions/transcripts available. Please try a different video that has captions enabled, or use the Text or URL input options instead.")
                elif "unavailable or private" in error_msg:
                    raise Exception("This video is unavailable or private. Please try a different video.")
                elif "quota exceeded" in error_msg:
                    raise Exception("YouTube API quota exceeded. Please try again later.")
                else:
                    # For other errors, try to get transcript in any available language
                    try:
                        print(f"Trying to get transcript in any available language...")
                        transcript = await self.get_transcript(video_id, "en")  # Fallback try 'en'
                    except Exception as fallback_error:
                        print(f"Fallback also failed: {fallback_error}")
                        raise transcript_error
            if not transcript.strip():
                raise Exception("No transcript available for this video. Please try a different video that has captions enabled.")
            # Import summarizer service
            from src.core.summarizer import SummarizerService
            summarizer = SummarizerService()
            # Chunk transcript to avoid model limits
            max_tokens = 512
            chunks = transcript.split('.')  # Naive split by sentence; improve with tokenizer if desired
            chunk_texts = []
            curr = ''
            for sent in chunks:
                if len((curr + sent).split()) < 350:  # Approx below 512 tokens (use tokenizer len if possible)
                    curr += sent + '.'
                else:
                    if curr.strip():
                        chunk_texts.append(curr.strip())
                    curr = sent + '.'
            if curr.strip():
                chunk_texts.append(curr.strip())
            # Summarize each chunk, join
            summaries = []
            for chunk in chunk_texts:
                if len(chunk.split()) < 10: continue
                result = await summarizer.summarize_text(
                    text=chunk,
                    language=language,
                    summary_length=summary_length
                )
                summaries.append(result['summary'].strip())
            summary = '\n'.join(summaries)
            # Add video info and completion
            word_count = len(transcript.split())
            summary_words = len(summary.split())
            return {
                "summary": summary,
                "original_length": word_count,
                "summary_length": summary_words,
                "compression_ratio": round(summary_words / word_count, 2) if word_count else 0.0,
                "title": video_info["title"],
                "video_id": video_id,
                "video_url": video_info["url"]
            }
        except Exception as e:
            error_msg = str(e)
            if any(phrase in error_msg.lower() for phrase in ["captions/transcripts available", "unavailable or private", "quota exceeded", "invalid youtube url"]):
                raise Exception(error_msg)
            else:
                raise Exception(f"Failed to process YouTube video: {error_msg}")
    
    def is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL"""
        try:
            video_id = self.extract_video_id(url)
            return video_id is not None
        except:
            return False
