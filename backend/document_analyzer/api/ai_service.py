import asyncio
import time
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class AIService:
    """
    Service class for handling AI model operations including document analysis,
    summarization, and question answering using Llama-3.2-1B model.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.dataset = None
        self.token="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"
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize the AI model and dataset. This method should be called
        during application startup to ensure models are loaded.
        """
        try:
            logger.info("Initializing AI Service...")
            
            # Load the Llama model and tokenizer
            model_name = "meta-llama/Llama-3.2-1B"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=self.token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                use_auth_token=self.token
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Load the arxiv summarization dataset
            logger.info("Loading arxiv-summarization dataset...")
            self.dataset = load_dataset("ccdv/arxiv-summarization", split="train")
            
            self.is_initialized = True
            logger.info("AI Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Service: {str(e)}")
            raise
    
    def get_relevant_context(self, query: str, document_content: str, max_context_length: int = 2000) -> str:
        """
        Extract relevant context from document based on query.
        
        Args:
            query: User's question or request
            document_content: Full document content
            max_context_length: Maximum length of context to return
            
        Returns:
            Relevant context string
        """
        # Simple keyword-based context extraction
        query_words = set(query.lower().split())
        sentences = document_content.split('.')
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            score = len(query_words.intersection(sentence_words))
            if score > 0:
                scored_sentences.append((score, sentence.strip()))
        
        # Sort by relevance score and combine top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        context = ""
        for score, sentence in scored_sentences:
            if len(context) + len(sentence) < max_context_length:
                context += sentence + ". "
            else:
                break
                
        return context.strip() or document_content[:max_context_length]
    
    async def process_query(self, query: str, document_content: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query against document content using the AI model.
        
        Args:
            query: User's natural language query
            document_content: Content of the document to analyze
            api_key: Optional API key (for future external API integration)
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Get relevant context
            context = self.get_relevant_context(query, document_content)
            
            # Construct prompt
            prompt = f"""Based on the following document content, please answer the question.

Document content:
{context}

Question: {query}

Answer:"""

            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.pipeline(
                    prompt,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            answer = generated_text.split("Answer:")[-1].strip()
            
            processing_time = time.time() - start_time
            
            return {
                'response': answer,
                'processing_time': processing_time,
                'model_used': 'meta-llama/Llama-3.2-1B',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the provided text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary
        """
        if not self.is_initialized:
            raise RuntimeError("AI Service not initialized")
        
        prompt = f"""Please provide a concise summary of the following text:

{text[:2000]}

Summary:"""
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            summary = generated_text.split("Summary:")[-1].strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

# Global AI service instance
ai_service = AIService()