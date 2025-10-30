"""
Core Summarization Service
Handles text processing, AI model integration, and export functionality
"""

import asyncio
import time
import os
import tempfile
from typing import Dict, Any, Literal
from pathlib import Path

# AI Model imports
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Export libraries
from fpdf import FPDF
from docx import Document
from docx.shared import Inches
import markdown

class SummarizerService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        
    async def load_model(self):
        """Load the T5 model asynchronously - optimized for free tier"""
        if self.model_loaded:
            return
            
        try:
            print("🔄 Loading T5 model (this may take 30-60 seconds on free tier)...")
            print("💡 Tip: First request after spin-down will be slower due to model loading")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.tokenizer, self.model = await loop.run_in_executor(
                None, self._load_model_sync
            )
            self.model_loaded = True
            print("✅ T5 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to extractive summarization only")
            self.model_loaded = False
    
    def _load_model_sync(self):
        """Synchronous model loading - optimized for free tier"""
        model_name = "t5-small"
        
        # Optimize for free tier memory constraints
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir="./model_cache",
            local_files_only=False
        )
        
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir="./model_cache",
            local_files_only=False,
            torch_dtype=torch.float32,  # Use float32 instead of float16 for compatibility
            low_cpu_mem_usage=True
        )
        
        # Set model to evaluation mode and optimize memory
        model.eval()
        model.config.use_cache = False
        
        return tokenizer, model
    
    def _tokenize_length(self, text: str) -> int:
        """Count the number of tokens in a text for T5 tokenizer limit check."""
        if not self.tokenizer:
            # Fallback guess: 1 word = 1 token
            return len(text.split())
        return len(self.tokenizer.encode(text, truncation=False))

    def _split_into_chunks(self, text: str, max_tokens: int = 512) -> list:
        """Splits text into chunks each <= max_tokens for T5 input limit."""
        if not self.tokenizer:
            # Fallback: naive split by word count
            words = text.split()
            return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
        tokens = self.tokenizer.encode(text, truncation=False)
        chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    async def summarize_text(self, text: str, language: Literal["hindi", "english"] = "hindi", summary_length: Literal["short", "medium", "long", "auto"] = "auto") -> Dict[str, Any]:
        """Summarize text using AI model; universal chunking/truncation for huge inputs."""
        try:
            start_time = time.time()
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            # Tokenize and chunk if needed
            await self.load_model()
            max_tokens = 512
            chunks = self._split_into_chunks(text, max_tokens=max_tokens)
            summary_chunks = []
            for ch in chunks:
                # Short circuit: if chunk too small, skip
                if self._tokenize_length(ch) < 10:
                    continue
                # Get summary for chunk (use T5 only if model loaded)
                if self.model_loaded:
                    # Assemble prompt (for Hindi or English)
                    prompt = ("summarize: " if language=="english" else "अनुच्छेद का सारांश बनाएँ: ") + ch
                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
                    summary_ids = self.model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
                    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                else:
                    # Fallback to extractive
                    summary = (await self._extractive_summarize(ch, language, "short"))['summary']
                summary_chunks.append(summary.strip())
            # Combine summaries
            summary = "\n".join(summary_chunks)
            processing_time = time.time() - start_time
            # Compression ratio: summarized length vs original length
            word_count = len(text.split())
            summary_words = len(summary.split())
            return {
                "summary": summary,
                "original_length": word_count,
                "summary_length": summary_words,
                "compression_ratio": round(summary_words / word_count, 2) if word_count else 0.0,
                "processing_time": round(processing_time, 2)
            }
        except Exception as e:
            print(f"Error in summarize_text (universal chunking): {e}")
            raise Exception(f"Failed to summarize text: {str(e)}")
    
    async def _extractive_summarize(
        self, 
        text: str, 
        language: Literal["hindi", "english"],
        summary_length: Literal["short", "medium", "long"]
    ) -> Dict[str, Any]:
        """Improved extractive summarization"""
        start_time = time.time()
        
        # Better sentence splitting for both English and Hindi
        import re
        # Handle both English and Hindi sentence endings
        sentences = re.split(r'[.!?।]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        word_count = len(text.split())
        
        # Better length mapping based on word count
        if summary_length == "short":
            target_words = min(50, max(20, word_count // 8))
        elif summary_length == "medium":
            target_words = min(100, max(40, word_count // 5))
        else:  # long
            target_words = min(200, max(80, word_count // 3))
        
        # Select sentences to reach target word count
        summary_sentences = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= target_words:
                summary_sentences.append(sentence)
                current_words += sentence_words
            else:
                # Add partial sentence if we're close to target
                if current_words < target_words * 0.7:  # If we're less than 70% of target
                    remaining_words = target_words - current_words
                    words = sentence.split()
                    if len(words) > remaining_words:
                        partial_sentence = ' '.join(words[:remaining_words])
                        summary_sentences.append(partial_sentence + "...")
                    else:
                        summary_sentences.append(sentence)
                break
        
        # If no sentences selected, take first sentence
        if not summary_sentences:
            summary_sentences = [sentences[0]] if sentences else [text[:100] + "..."]
        
        summary = '. '.join(summary_sentences).strip()
        
        # Ensure summary ends with proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        processing_time = time.time() - start_time
        
        return {
            "summary": summary,
            "original_length": word_count,
            "summary_length": len(summary.split()),
            "compression_ratio": round(len(summary.split()) / word_count, 2),
            "processing_time": round(processing_time, 2)
        }
    
    async def summarize_url(
        self, 
        url: str, 
        language: Literal["hindi", "english"] = "hindi",
        summary_length: Literal["short", "medium", "long", "auto"] = "auto"
    ) -> Dict[str, Any]:
        """Extract and summarize content from URL"""
        try:
            # Validate URL
            if not url or not url.strip():
                raise ValueError("URL cannot be empty")
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            from newspaper import Article
            
            # Extract article content
            article = Article(url, language="hi" if language == "hindi" else "en")
            article.download()
            article.parse()
            
            text = article.text.strip()
            title = article.title or "Untitled Article"
            
            if not text:
                raise ValueError("No content extracted from URL. The website might not be accessible or doesn't contain readable text.")
            
            # Summarize the extracted text
            result = await self.summarize_text(text, language, summary_length)
            result["title"] = title
            result["url"] = url
            
            return result
            
        except Exception as e:
            print(f"Error in summarize_url: {e}")
            raise Exception(f"Failed to process URL: {str(e)}")
    
    async def export_pdf(
        self, 
        summary: str, 
        title: str = "Summary",
        language: Literal["hindi", "english"] = "hindi"
    ) -> str:
        """Export summary as PDF with proper font support"""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Use default fonts to avoid font issues
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, title, 0, 1, "C")
            pdf.ln(10)
            
            # Add summary with proper encoding
            pdf.set_font("Helvetica", "", 12)
            
            # Handle text encoding properly
            try:
                # Try to encode the summary properly
                if isinstance(summary, str):
                    summary_text = summary.encode('latin-1', 'replace').decode('latin-1')
                else:
                    summary_text = str(summary)
            except:
                summary_text = str(summary)
            
            pdf.multi_cell(0, 8, summary_text)
            
            # Add footer
            pdf.ln(20)
            pdf.set_font("Helvetica", "I", 8)
            footer_text = f"Generated by MultiLanguage AI Text Summarizer - {time.strftime('%Y-%m-%d %H:%M')}"
            pdf.cell(0, 10, footer_text, 0, 1, "C")
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.output(temp_file.name)
            
            return temp_file.name
            
        except Exception as e:
            print(f"PDF export error: {e}")
            raise Exception(f"Failed to create PDF: {str(e)}")
    
    async def export_word(
        self, 
        summary: str, 
        title: str = "Summary",
        language: Literal["hindi", "english"] = "hindi"
    ) -> str:
        """Export summary as Word document"""
        try:
            doc = Document()
            
            # Add title
            title_para = doc.add_heading(title, 0)
            
            # Add summary
            summary_para = doc.add_paragraph(summary)
            
            # Add metadata
            doc.add_paragraph(f"\n\nGenerated by MultiLanguage AI Text Summarizer")
            doc.add_paragraph(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            doc.save(temp_file.name)
            
            return temp_file.name
            
        except Exception as e:
            raise Exception(f"Failed to create Word document: {str(e)}")
    
    async def export_markdown(
        self, 
        summary: str, 
        title: str = "Summary",
        language: Literal["hindi", "english"] = "hindi"
    ) -> str:
        """Export summary as Markdown"""
        try:
            markdown_content = f"""# {title}

{summary}

---

*Generated by MultiLanguage AI Text Summarizer*  
*Date: {time.strftime('%Y-%m-%d %H:%M')}*
"""
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode='w', encoding='utf-8')
            temp_file.write(markdown_content)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            raise Exception(f"Failed to create Markdown file: {str(e)}")
