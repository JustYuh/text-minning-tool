#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Keyword Extractor Module
This module provides the KeywordExtractor class for extracting keywords from text data.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Download required NLTK data if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK is not available. Basic keyword extraction will be used.")


class KeywordExtractor:
    """
    A class for extracting keywords from text data.
    
    This class provides methods for identifying and extracting important keywords
    from text data using various techniques such as TF-IDF and frequency analysis.
    
    Attributes:
        use_nltk (bool): Whether to use NLTK for tokenization and stopword removal.
        language (str): The language code (ISO 639-1) for the text.
        min_word_length (int): Minimum length of words to consider as keywords.
        max_keywords (int): Maximum number of keywords to extract.
        stopwords (set): Set of stopwords to exclude from keywords.
    """
    
    def __init__(
        self,
        use_nltk: bool = True,
        language: str = 'en',
        min_word_length: int = 3,
        max_keywords: int = 10,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the KeywordExtractor.
        
        Args:
            use_nltk: Whether to use NLTK for tokenization and stopword removal (if available).
            language: The language code (ISO 639-1) for the text.
            min_word_length: Minimum length of words to consider as keywords.
            max_keywords: Maximum number of keywords to extract.
            custom_stopwords: Additional stopwords to exclude from keywords.
        """
        self.use_nltk = use_nltk and NLTK_AVAILABLE
        self.language = language
        self.min_word_length = min_word_length
        self.max_keywords = max_keywords
        
        # Initialize stopwords
        self.stopwords = set()
        
        if self.use_nltk:
            try:
                self.stopwords = set(stopwords.words(self._get_nltk_language(language)))
            except:
                # Fallback to English if language not available
                self.stopwords = set(stopwords.words('english'))
        else:
            # Basic English stopwords
            self.stopwords = {
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up',
                'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don',
                'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
                'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
                'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'i', 'me', 'my',
                'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
            }
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def _get_nltk_language(self, language: str) -> str:
        """
        Get the NLTK language name from the ISO 639-1 language code.
        
        Args:
            language: The ISO 639-1 language code.
            
        Returns:
            The NLTK language name.
        """
        language_map = {
            'en': 'english',
            'fr': 'french',
            'de': 'german',
            'es': 'spanish',
            'it': 'italian',
            'pt': 'portuguese',
            'nl': 'dutch',
            'ru': 'russian'
        }
        
        return language_map.get(language, 'english')
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            A list of tokens.
        """
        if not text:
            return []
        
        if self.use_nltk:
            # Use NLTK for tokenization
            tokens = word_tokenize(text.lower())
        else:
            # Simple tokenization
            text = text.lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Tokenize
            tokens = text.split()
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_word_length:
                continue
            
            # Skip stopwords
            if token in self.stopwords:
                continue
            
            # Skip tokens that are just numbers
            if token.isdigit():
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords from the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A list of dictionaries containing the extracted keywords.
        """
        if not text:
            return []
        
        # Tokenize the text
        tokens = self._tokenize(text)
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Calculate TF (Term Frequency) for each token
        total_tokens = len(tokens)
        tf = {token: count / total_tokens for token, count in token_counts.items()}
        
        # Sort tokens by frequency
        sorted_tokens = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        
        # Extract top keywords
        keywords = []
        for token, score in sorted_tokens[:self.max_keywords]:
            keywords.append({
                'text': token,
                'score': score,
                'count': token_counts[token]
            })
        
        return keywords
    
    def extract_keywords_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract keywords from a batch of texts.
        
        Args:
            texts: The texts to analyze.
            
        Returns:
            A list of lists of dictionaries containing the extracted keywords.
        """
        if not texts:
            return []
        
        # Tokenize all texts
        all_tokens = [self._tokenize(text) for text in texts]
        
        # Calculate document frequency for each token
        doc_freq = {}
        for tokens in all_tokens:
            # Count each token only once per document
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        
        # Calculate IDF (Inverse Document Frequency) for each token
        num_docs = len(texts)
        idf = {token: math.log(num_docs / (freq + 1)) for token, freq in doc_freq.items()}
        
        # Extract keywords for each text using TF-IDF
        results = []
        for i, text in enumerate(texts):
            if not text:
                results.append([])
                continue
            
            tokens = all_tokens[i]
            
            # Count token frequencies
            token_counts = Counter(tokens)
            
            # Calculate TF (Term Frequency) for each token
            total_tokens = len(tokens)
            tf = {token: count / total_tokens for token, count in token_counts.items()}
            
            # Calculate TF-IDF for each token
            tfidf = {token: tf_val * idf.get(token, 0) for token, tf_val in tf.items()}
            
            # Sort tokens by TF-IDF
            sorted_tokens = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
            
            # Extract top keywords
            keywords = []
            for token, score in sorted_tokens[:self.max_keywords]:
                keywords.append({
                    'text': token,
                    'score': score,
                    'count': token_counts[token]
                })
            
            results.append(keywords)
        
        return results 