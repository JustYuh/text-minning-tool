#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Keyword Extraction Module
-----------------------
This module provides functionality for extracting important keywords and phrases
from text data using various techniques such as TF-IDF, RAKE, and YAKE.
"""

import logging
import re
import string
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Configure logging
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class KeywordExtractor:
    """
    A class for extracting important keywords and phrases from text data.
    
    This class provides methods for keyword extraction using various techniques
    such as TF-IDF, RAKE, and YAKE.
    """
    
    def __init__(self, method: str = 'tfidf', language: str = 'en', 
                 max_features: int = 50, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 1, max_df: float = 0.9):
        """
        Initialize the KeywordExtractor with the specified parameters.
        
        Args:
            method: The keyword extraction method to use ('tfidf', 'rake', 'yake').
            language: The language code (ISO 639-1) for the text.
            max_features: The maximum number of keywords to extract.
            ngram_range: The range of n-gram sizes to consider.
            min_df: The minimum document frequency for TF-IDF.
            max_df: The maximum document frequency for TF-IDF.
        """
        self.method = method
        self.language = language
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Set up stopwords
        self.stopwords = set()
        try:
            self.stopwords = set(stopwords.words(self._map_language_code()))
        except:
            # Fallback to English if the language is not available
            self.stopwords = set(stopwords.words('english'))
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=self._map_language_code(),
            min_df=min_df,
            max_df=max_df,
            use_idf=True,
            sublinear_tf=True
        )
    
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
    
    def extract_keywords_tfidf(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            texts: A list of texts to analyze.
            
        Returns:
            A list of lists of keyword dictionaries.
        """
        if not texts:
            return []
        
        try:
            # Fit the TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Extract keywords for each document
            results = []
            for i, doc in enumerate(texts):
                # Get the TF-IDF scores for this document
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                
                # Create a list of (word, score) tuples
                word_scores = [(feature_names[j], tfidf_scores[j]) for j in range(len(feature_names)) if tfidf_scores[j] > 0]
                
                # Sort by score in descending order
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take the top keywords
                top_keywords = word_scores[:self.max_features]
                
                # Format the results
                doc_keywords = [
                    {
                        "text": word,
                        "score": float(score),
                        "method": "tfidf"
                    }
                    for word, score in top_keywords
                ]
                
                results.append(doc_keywords)
            
            return results
        
        except Exception as e:
            logger.error(f"Error extracting keywords with TF-IDF: {str(e)}")
            return [[] for _ in range(len(texts))]
    
    def extract_keywords_rake(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract keywords using RAKE (Rapid Automatic Keyword Extraction).
        
        Args:
            texts: A list of texts to analyze.
            
        Returns:
            A list of lists of keyword dictionaries.
        """
        if not texts:
            return []
        
        results = []
        
        for text in texts:
            try:
                # Split the text into sentences
                sentences = sent_tokenize(text)
                
                # Extract phrases (split by stopwords and punctuation)
                phrase_list = []
                for sentence in sentences:
                    # Convert to lowercase and remove punctuation
                    sentence = sentence.lower()
                    sentence = re.sub(f'[{re.escape(string.punctuation)}]', ' ', sentence)
                    
                    # Tokenize
                    words = word_tokenize(sentence)
                    
                    # Extract phrases
                    phrase = []
                    for word in words:
                        if word not in self.stopwords and len(word) > 1:
                            phrase.append(word)
                        elif phrase:
                            if len(phrase) > 0:
                                phrase_list.append(' '.join(phrase))
                            phrase = []
                    
                    # Add the last phrase if it exists
                    if phrase:
                        phrase_list.append(' '.join(phrase))
                
                # Calculate word frequency
                word_freq = defaultdict(int)
                for phrase in phrase_list:
                    for word in phrase.split():
                        word_freq[word] += 1
                
                # Calculate phrase scores
                phrase_scores = {}
                for phrase in phrase_list:
                    words = phrase.split()
                    if len(words) > 0:
                        phrase_score = sum(word_freq[word] for word in words) / len(words)
                        phrase_scores[phrase] = phrase_score
                
                # Sort phrases by score
                sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Take the top keywords
                top_keywords = sorted_phrases[:self.max_features]
                
                # Format the results
                doc_keywords = [
                    {
                        "text": phrase,
                        "score": float(score),
                        "method": "rake"
                    }
                    for phrase, score in top_keywords
                ]
                
                results.append(doc_keywords)
            
            except Exception as e:
                logger.error(f"Error extracting keywords with RAKE: {str(e)}")
                results.append([])
        
        return results
    
    def extract_keywords_yake(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract keywords using YAKE (Yet Another Keyword Extractor).
        
        Args:
            texts: A list of texts to analyze.
            
        Returns:
            A list of lists of keyword dictionaries.
        """
        try:
            import yake
        except ImportError:
            logger.warning("YAKE is not installed. Install it with 'pip install yake'.")
            return [[] for _ in range(len(texts))]
        
        if not texts:
            return []
        
        results = []
        
        try:
            # Initialize YAKE
            kw_extractor = yake.KeywordExtractor(
                lan=self.language,
                n=self.ngram_range[1],
                dedupLim=0.9,
                dedupFunc='seqm',
                windowsSize=1,
                top=self.max_features,
                features=None
            )
            
            # Extract keywords for each document
            for text in texts:
                try:
                    # Extract keywords
                    keywords = kw_extractor.extract_keywords(text)
                    
                    # Format the results (YAKE scores are inverted, lower is better)
                    doc_keywords = [
                        {
                            "text": keyword,
                            "score": 1.0 / (score + 1e-10),  # Invert the score
                            "method": "yake"
                        }
                        for keyword, score in keywords
                    ]
                    
                    results.append(doc_keywords)
                
                except Exception as e:
                    logger.error(f"Error extracting keywords with YAKE for a document: {str(e)}")
                    results.append([])
            
            return results
        
        except Exception as e:
            logger.error(f"Error extracting keywords with YAKE: {str(e)}")
            return [[] for _ in range(len(texts))]
    
    def extract_keywords_batch(self, texts: List[Union[str, Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        Extract keywords from a batch of texts.
        
        Args:
            texts: A list of texts or processed text dictionaries.
            
        Returns:
            A list of lists of keyword dictionaries.
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
        
        # Extract keywords using the specified method
        if self.method == 'rake':
            return self.extract_keywords_rake(processed_texts)
        elif self.method == 'yake':
            return self.extract_keywords_yake(processed_texts)
        else:
            return self.extract_keywords_tfidf(processed_texts)
    
    def extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords from a single text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A list of keyword dictionaries.
        """
        results = self.extract_keywords_batch([text])
        return results[0] if results else [] 