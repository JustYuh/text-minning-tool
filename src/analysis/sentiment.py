#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analyzer Module
This module provides the SentimentAnalyzer class for analyzing sentiment in text data.
"""

import re
from typing import Dict, List, Any, Optional, Union
from collections import Counter

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Download required NLTK data if not already downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
        
    NLTK_SENTIMENT_AVAILABLE = True
except ImportError:
    NLTK_SENTIMENT_AVAILABLE = False
    print("NLTK SentimentIntensityAnalyzer is not available. Basic sentiment analysis will be used.")

# Simple lexicon for basic sentiment analysis when NLTK is not available
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'terrific',
    'outstanding', 'superb', 'brilliant', 'awesome', 'fabulous', 'incredible',
    'love', 'happy', 'joy', 'pleased', 'delight', 'positive', 'nice', 'best',
    'better', 'beautiful', 'perfect', 'recommend', 'recommended', 'satisfied',
    'impressive', 'enjoy', 'enjoyed', 'like', 'liked', 'favorite', 'favourite'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing', 'disappointed',
    'worst', 'waste', 'hate', 'dislike', 'sad', 'unhappy', 'angry', 'negative',
    'problem', 'issue', 'difficult', 'hard', 'trouble', 'fail', 'failed', 'failure',
    'boring', 'annoying', 'frustrating', 'useless', 'expensive', 'overpriced',
    'avoid', 'avoid', 'unfortunately', 'not', 'never', 'no', 'not', "don't", "doesn't"
}


class SentimentAnalyzer:
    """
    A class for analyzing sentiment in text data.
    
    This class provides methods for determining the sentiment (positive, negative, neutral)
    of text data using either NLTK's SentimentIntensityAnalyzer or a simple lexicon-based
    approach when NLTK is not available.
    
    Attributes:
        use_nltk (bool): Whether to use NLTK's SentimentIntensityAnalyzer.
        sia: The NLTK SentimentIntensityAnalyzer instance (if available).
    """
    
    def __init__(self, use_nltk: bool = True):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            use_nltk: Whether to use NLTK's SentimentIntensityAnalyzer (if available).
        """
        self.use_nltk = use_nltk and NLTK_SENTIMENT_AVAILABLE
        
        if self.use_nltk:
            self.sia = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A dictionary containing the sentiment analysis results.
        """
        if not text:
            return {
                'score': 0.0,
                'label': 'neutral',
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        
        if self.use_nltk:
            return self._analyze_with_nltk(text)
        else:
            return self._analyze_with_lexicon(text)
    
    def _analyze_with_nltk(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the text using NLTK's SentimentIntensityAnalyzer.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A dictionary containing the sentiment analysis results.
        """
        scores = self.sia.polarity_scores(text)
        
        # Determine the sentiment label based on the compound score
        if scores['compound'] >= 0.05:
            label = 'positive'
        elif scores['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': scores['compound'],
            'label': label,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def _analyze_with_lexicon(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the text using a simple lexicon-based approach.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A dictionary containing the sentiment analysis results.
        """
        # Preprocess the text
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = text.split()
        
        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
        negative_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
        total_count = len(tokens)
        
        # Calculate scores
        if total_count > 0:
            positive_score = positive_count / total_count
            negative_score = negative_count / total_count
            neutral_score = 1.0 - (positive_score + negative_score)
        else:
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 1.0
        
        # Calculate compound score (similar to NLTK's compound score)
        compound_score = positive_score - negative_score
        
        # Determine the sentiment label
        if compound_score >= 0.05:
            label = 'positive'
        elif compound_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': compound_score,
            'label': label,
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of a batch of texts.
        
        Args:
            texts: The texts to analyze.
            
        Returns:
            A list of dictionaries containing the sentiment analysis results.
        """
        return [self.analyze(text) for text in texts] 