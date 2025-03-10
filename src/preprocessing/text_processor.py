#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Processor Module
--------------------
This module provides functionality for preprocessing and normalizing text data.
It includes methods for tokenization, stemming, lemmatization, stopword removal,
and other text cleaning operations.
"""

import re
import string
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Union, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from tqdm import tqdm

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextProcessor:
    """
    A class for preprocessing and normalizing text data.
    
    This class provides methods for various text preprocessing tasks such as
    tokenization, stemming, lemmatization, stopword removal, and more.
    """
    
    def __init__(self, language: str = 'en', 
                 remove_stopwords: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 lemmatize: bool = True,
                 stem: bool = False,
                 lowercase: bool = True,
                 min_token_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the TextProcessor with the specified parameters.
        
        Args:
            language: The language code (ISO 639-1) for the text.
            remove_stopwords: Whether to remove stopwords.
            remove_punctuation: Whether to remove punctuation.
            remove_numbers: Whether to remove numbers.
            lemmatize: Whether to lemmatize tokens.
            stem: Whether to stem tokens.
            lowercase: Whether to convert text to lowercase.
            min_token_length: Minimum length for a token to be kept.
            custom_stopwords: Additional stopwords to remove.
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.stem = stem
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        
        # Set up stopwords
        self.stopwords = set()
        if remove_stopwords:
            try:
                self.stopwords = set(stopwords.words(self._map_language_code()))
            except:
                # Fallback to English if the language is not available
                self.stopwords = set(stopwords.words('english'))
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Initialize lemmatizer and stemmer
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer() if stem else None
        
        # Initialize spaCy model if available
        self.nlp = None
        try:
            if language == 'en':
                self.nlp = spacy.load('en_core_web_sm')
            elif language == 'es':
                self.nlp = spacy.load('es_core_news_sm')
            elif language == 'fr':
                self.nlp = spacy.load('fr_core_news_sm')
            elif language == 'de':
                self.nlp = spacy.load('de_core_news_sm')
            # Add more languages as needed
        except:
            # Fallback to basic processing if spaCy model is not available
            pass
    
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by applying various cleaning operations.
        
        Args:
            text: The input text to preprocess.
            
        Returns:
            The preprocessed text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.
        
        Args:
            text: The input text to tokenize.
            
        Returns:
            A list of tokens.
        """
        if not text:
            return []
        
        # Use spaCy for tokenization if available
        if self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
        else:
            # Fallback to NLTK tokenization
            tokens = word_tokenize(text)
        
        # Apply filters
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token in self.stopwords:
                continue
            
            # Apply lemmatization if specified
            if self.lemmatize and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            # Apply stemming if specified
            if self.stem and self.stemmer:
                token = self.stemmer.stem(token)
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process the text by applying preprocessing and tokenization.
        
        Args:
            text: The input text to process.
            
        Returns:
            A dictionary containing the processed text and tokens.
        """
        if not text:
            return {"original": "", "processed": "", "tokens": []}
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Tokenize the preprocessed text
        tokens = self.tokenize(processed_text)
        
        # Extract sentences
        sentences = sent_tokenize(text)
        
        return {
            "original": text,
            "processed": processed_text,
            "tokens": tokens,
            "sentences": sentences,
            "token_count": len(tokens),
            "sentence_count": len(sentences),
            "language": self.language
        }
    
    def process_batch(self, texts: List[str], batch_size: int = 64, n_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Process a batch of texts in parallel.
        
        Args:
            texts: A list of texts to process.
            batch_size: The batch size for processing.
            n_workers: The number of worker processes.
            
        Returns:
            A list of dictionaries containing the processed texts and tokens.
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Process the batch in parallel
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    batch_results = list(tqdm(
                        executor.map(self.process, batch),
                        total=len(batch),
                        desc=f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
                    ))
            else:
                # Process sequentially if n_workers <= 1
                batch_results = [self.process(text) for text in tqdm(batch, desc=f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")]
            
            results.extend(batch_results)
        
        return results 