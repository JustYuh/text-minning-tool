#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Summarization Module
-----------------------
This module provides functionality for summarizing text using both extractive
and abstractive techniques.
"""

import logging
import re
import string
import heapq
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextSummarizer:
    """
    A class for summarizing text using various techniques.
    
    This class provides methods for text summarization using both extractive
    and abstractive techniques.
    """
    
    def __init__(self, method: str = 'extractive', model_size: str = 'default',
                 ratio: float = 0.3, max_length: int = 150, min_length: int = 50,
                 language: str = 'en', device: int = -1):
        """
        Initialize the TextSummarizer with the specified parameters.
        
        Args:
            method: The summarization method to use ('extractive', 'abstractive').
            model_size: The size of the model to use ('small', 'default', 'large').
            ratio: The ratio of the original text to keep in the summary (for extractive).
            max_length: The maximum length of the summary (for abstractive).
            min_length: The minimum length of the summary (for abstractive).
            language: The language code (ISO 639-1) for the text.
            device: The device to use for inference (-1 for CPU, 0+ for GPU).
        """
        self.method = method
        self.model_size = model_size
        self.ratio = ratio
        self.max_length = max_length
        self.min_length = min_length
        self.language = language
        self.device = device
        
        # Set up stopwords
        self.stopwords = set()
        try:
            self.stopwords = set(stopwords.words(self._map_language_code()))
        except:
            # Fallback to English if the language is not available
            self.stopwords = set(stopwords.words('english'))
        
        # Initialize abstractive summarization pipeline if needed
        self.summarization_pipeline = None
        if method == 'abstractive':
            try:
                # Map model size to model name
                model_map = {
                    'small': 't5-small',
                    'default': 'facebook/bart-large-cnn',
                    'large': 'google/pegasus-xsum'
                }
                
                model_name = model_map.get(model_size, model_map['default'])
                
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=model_name,
                    device=device
                )
                logger.info(f"Initialized abstractive summarization pipeline with model: {model_name}")
            except Exception as e:
                logger.error(f"Error initializing abstractive summarization pipeline: {str(e)}")
                self.summarization_pipeline = None
    
    def _map_language_code(self) -> str:
        """Map ISO 639-1 language code to NLTK language name."""
        language_map = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'nl': 'dutch',
            'pt': 'portuguese',
            'ru': 'russian',
            # Add more mappings as needed
        }
        return language_map.get(self.language, 'english')
    
    def summarize_extractive(self, text: str, ratio: Optional[float] = None) -> str:
        """
        Summarize text using extractive techniques.
        
        Args:
            text: The text to summarize.
            ratio: The ratio of the original text to keep in the summary.
            
        Returns:
            The summarized text.
        """
        if not text:
            return ""
        
        # Use the provided ratio or the default
        ratio = ratio if ratio is not None else self.ratio
        
        try:
            # Split the text into sentences
            sentences = sent_tokenize(text)
            
            # Skip summarization if the text is already short
            if len(sentences) <= 3:
                return text
            
            # Calculate the number of sentences to include in the summary
            num_sentences = max(1, int(len(sentences) * ratio))
            
            # Preprocess the text
            clean_sentences = []
            for sentence in sentences:
                # Convert to lowercase and remove punctuation
                clean_sentence = sentence.lower()
                clean_sentence = re.sub(f'[{re.escape(string.punctuation)}]', '', clean_sentence)
                clean_sentences.append(clean_sentence)
            
            # Calculate word frequency
            word_freq = defaultdict(int)
            for sentence in clean_sentences:
                for word in word_tokenize(sentence):
                    if word not in self.stopwords:
                        word_freq[word] += 1
            
            # Normalize word frequency
            max_freq = max(word_freq.values()) if word_freq else 1
            for word in word_freq:
                word_freq[word] = word_freq[word] / max_freq
            
            # Calculate sentence scores
            sentence_scores = {}
            for i, sentence in enumerate(clean_sentences):
                for word in word_tokenize(sentence):
                    if word in word_freq:
                        if i not in sentence_scores:
                            sentence_scores[i] = word_freq[word]
                        else:
                            sentence_scores[i] += word_freq[word]
            
            # Get the top sentences
            top_sentence_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            top_sentence_indices.sort()  # Sort to maintain original order
            
            # Create the summary
            summary = ' '.join([sentences[i] for i in top_sentence_indices])
            
            return summary
        
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            return text
    
    def summarize_abstractive(self, text: str, max_length: Optional[int] = None, min_length: Optional[int] = None) -> str:
        """
        Summarize text using abstractive techniques.
        
        Args:
            text: The text to summarize.
            max_length: The maximum length of the summary.
            min_length: The minimum length of the summary.
            
        Returns:
            The summarized text.
        """
        if not text or not self.summarization_pipeline:
            return ""
        
        # Use the provided lengths or the defaults
        max_length = max_length if max_length is not None else self.max_length
        min_length = min_length if min_length is not None else self.min_length
        
        try:
            # Truncate text if it's too long to avoid memory issues
            max_input_length = 1024
            if len(text) > max_input_length:
                # Try to truncate at sentence boundaries
                sentences = sent_tokenize(text)
                truncated_text = ""
                for sentence in sentences:
                    if len(truncated_text) + len(sentence) <= max_input_length:
                        truncated_text += sentence + " "
                    else:
                        break
                text = truncated_text.strip()
            
            # Generate the summary
            summary = self.summarization_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            return summary
        
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {str(e)}")
            return self.summarize_extractive(text)  # Fallback to extractive
    
    def summarize(self, text: str) -> str:
        """
        Summarize text using the specified method.
        
        Args:
            text: The text to summarize.
            
        Returns:
            The summarized text.
        """
        if not text:
            return ""
        
        if self.method == 'abstractive' and self.summarization_pipeline:
            return self.summarize_abstractive(text)
        else:
            return self.summarize_extractive(text)
    
    def summarize_batch(self, texts: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: A list of texts or processed text dictionaries.
            
        Returns:
            A list of summarized texts.
        """
        if not texts:
            return []
        
        # Extract text from dictionaries if needed
        processed_texts = []
        for item in texts:
            if isinstance(item, dict):
                # Use the processed text if available, otherwise use the original
                text = item.get('processed', item.get('original', ''))
            else:
                text = item
            processed_texts.append(text)
        
        # Summarize each text
        summaries = []
        for text in processed_texts:
            summary = self.summarize(text)
            summaries.append(summary)
        
        return summaries 