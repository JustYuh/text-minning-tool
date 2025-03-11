#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Processor Module
This module provides the TextProcessor class for preprocessing and normalizing text data.
"""

import re
import string
from typing import Dict, List, Any, Optional, Union

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    
    # Download required NLTK data if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK is not installed. Basic functionality will be limited.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy is not installed. Advanced NLP features will not be available.")


class TextProcessor:
    """
    A class for preprocessing and normalizing text data.
    
    This class provides methods for preprocessing text, including tokenization,
    stopword removal, lemmatization, stemming, and more.
    
    Attributes:
        language (str): The language code (ISO 639-1) for the text.
        remove_stopwords (bool): Whether to remove stopwords.
        remove_punctuation (bool): Whether to remove punctuation.
        remove_numbers (bool): Whether to remove numbers.
        lemmatize (bool): Whether to lemmatize tokens.
        stem (bool): Whether to stem tokens.
        lowercase (bool): Whether to convert text to lowercase.
        min_token_length (int): Minimum length of tokens to keep.
        custom_stopwords (List[str]): Additional stopwords to remove.
        use_spacy (bool): Whether to use spaCy for advanced NLP (if available).
        nlp: The spaCy language model (if spaCy is available and use_spacy is True).
    """
    
    def __init__(
        self,
        language: str = 'en',
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        lemmatize: bool = True,
        stem: bool = False,
        lowercase: bool = True,
        min_token_length: int = 2,
        custom_stopwords: Optional[List[str]] = None,
        use_spacy: bool = True
    ):
        """
        Initialize the TextProcessor.
        
        Args:
            language: The language code (ISO 639-1) for the text.
            remove_stopwords: Whether to remove stopwords.
            remove_punctuation: Whether to remove punctuation.
            remove_numbers: Whether to remove numbers.
            lemmatize: Whether to lemmatize tokens.
            stem: Whether to stem tokens.
            lowercase: Whether to convert text to lowercase.
            min_token_length: Minimum length of tokens to keep.
            custom_stopwords: Additional stopwords to remove.
            use_spacy: Whether to use spaCy for advanced NLP (if available).
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.stem = stem
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        self.custom_stopwords = custom_stopwords or []
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # Initialize NLTK components
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words(self._get_nltk_language(language)))
            if self.custom_stopwords:
                self.stop_words.update(self.custom_stopwords)
            
            if self.lemmatize:
                self.lemmatizer = WordNetLemmatizer()
            
            if self.stem:
                self.stemmer = PorterStemmer()
        
        # Initialize spaCy model if available and requested
        if self.use_spacy:
            try:
                self.nlp = spacy.load(self._get_spacy_model(language))
            except OSError:
                print(f"spaCy model for language '{language}' not found. Using NLTK instead.")
                self.use_spacy = False
    
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
    
    def _get_spacy_model(self, language: str) -> str:
        """
        Get the spaCy model name from the ISO 639-1 language code.
        
        Args:
            language: The ISO 639-1 language code.
            
        Returns:
            The spaCy model name.
        """
        language_map = {
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm',
            'de': 'de_core_news_sm',
            'es': 'es_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'ru': 'ru_core_news_sm'
        }
        
        return language_map.get(language, 'en_core_web_sm')
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text by applying various cleaning operations.
        
        Args:
            text: The text to preprocess.
            
        Returns:
            The preprocessed text.
        """
        if not text:
            return ""
        
        # Convert to lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers if requested
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            A list of tokens.
        """
        if not text:
            return []
        
        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)
        
        if self.use_spacy:
            # Use spaCy for tokenization
            doc = self.nlp(preprocessed_text)
            tokens = [token.text for token in doc]
        elif NLTK_AVAILABLE:
            # Use NLTK for tokenization
            tokens = word_tokenize(preprocessed_text)
        else:
            # Fallback to simple whitespace tokenization
            tokens = preprocessed_text.split()
        
        # Apply additional processing to tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            
            # Skip stopwords if requested
            if self.remove_stopwords and NLTK_AVAILABLE and token in self.stop_words:
                continue
            
            # Apply lemmatization if requested
            if self.lemmatize and NLTK_AVAILABLE:
                token = self.lemmatizer.lemmatize(token)
            
            # Apply stemming if requested
            if self.stem and NLTK_AVAILABLE:
                token = self.stemmer.stem(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process the text and return various representations.
        
        Args:
            text: The text to process.
            
        Returns:
            A dictionary containing the original text, tokens, and sentences.
        """
        if not text:
            return {
                'original_text': '',
                'tokens': [],
                'sentences': []
            }
        
        # Tokenize the text
        tokens = self.tokenize(text)
        
        # Split the text into sentences
        if self.use_spacy:
            # Use spaCy for sentence segmentation
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        elif NLTK_AVAILABLE:
            # Use NLTK for sentence segmentation
            sentences = sent_tokenize(text)
        else:
            # Fallback to simple sentence segmentation
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        return {
            'original_text': text,
            'tokens': tokens,
            'sentences': sentences
        }
    
    def process_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        n_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of texts in parallel.
        
        Args:
            texts: The texts to process.
            batch_size: The batch size for processing.
            n_workers: The number of worker processes.
            
        Returns:
            A list of dictionaries containing the processed texts.
        """
        try:
            from concurrent.futures import ProcessPoolExecutor
            import numpy as np
            
            # Split texts into batches
            batches = np.array_split(texts, max(1, len(texts) // batch_size))
            
            results = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for batch in batches:
                    batch_results = list(executor.map(self.process, batch))
                    results.extend(batch_results)
            
            return results
        except ImportError:
            # Fallback to sequential processing
            return [self.process(text) for text in texts] 