#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analysis Module
------------------------
This module provides functionality for analyzing sentiment in text data.
It uses transformer-based models to classify text into positive, negative,
or neutral sentiment categories.
"""

import logging
from typing import List, Dict, Any, Union, Optional

import numpy as np
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A class for analyzing sentiment in text data.
    
    This class uses transformer-based models to classify text into
    positive, negative, or neutral sentiment categories.
    """
    
    def __init__(self, model_size: str = 'default', device: int = -1):
        """
        Initialize the SentimentAnalyzer with the specified parameters.
        
        Args:
            model_size: The size of the model to use ('small', 'default', 'large').
            device: The device to use for inference (-1 for CPU, 0+ for GPU).
        """
        self.model_size = model_size
        self.device = device
        
        # Map model size to model name
        model_map = {
            'small': 'distilbert-base-uncased-finetuned-sst-2-english',
            'default': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'large': 'nlptown/bert-base-multilingual-uncased-sentiment'
        }
        
        model_name = model_map.get(model_size, model_map['default'])
        
        try:
            # Initialize the sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
            logger.info(f"Initialized sentiment analysis pipeline with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing sentiment analysis pipeline: {str(e)}")
            # Fallback to a simpler model if the requested one fails
            try:
                fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=fallback_model,
                    tokenizer=fallback_model,
                    device=device
                )
                logger.info(f"Initialized fallback sentiment analysis pipeline with model: {fallback_model}")
            except Exception as e2:
                logger.error(f"Error initializing fallback sentiment analysis pipeline: {str(e2)}")
                self.sentiment_pipeline = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The input text to analyze.
            
        Returns:
            A dictionary containing the sentiment label and score.
        """
        if not text or not self.sentiment_pipeline:
            return {"label": "neutral", "score": 0.5}
        
        try:
            # Truncate text if it's too long to avoid memory issues
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Run the sentiment analysis pipeline
            result = self.sentiment_pipeline(text)[0]
            
            # Normalize the label to positive, negative, or neutral
            label = result['label'].lower()
            if 'positive' in label or 'pos' in label or '5' in label or '4' in label:
                normalized_label = 'positive'
            elif 'negative' in label or 'neg' in label or '1' in label or '2' in label:
                normalized_label = 'negative'
            else:
                normalized_label = 'neutral'
            
            return {
                "label": normalized_label,
                "score": result['score'],
                "original_label": result['label']
            }
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "neutral", "score": 0.5, "error": str(e)}
    
    def analyze_batch(self, texts: List[Union[str, Dict[str, Any]]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of a batch of texts.
        
        Args:
            texts: A list of texts or processed text dictionaries to analyze.
            batch_size: The batch size for processing.
            
        Returns:
            A list of dictionaries containing the sentiment labels and scores.
        """
        if not texts or not self.sentiment_pipeline:
            return [{"label": "neutral", "score": 0.5} for _ in range(len(texts))]
        
        results = []
        
        # Extract text from dictionaries if needed
        processed_texts = []
        for item in texts:
            if isinstance(item, dict):
                # Use the processed text if available, otherwise use the original
                text = item.get('processed', item.get('original', ''))
            else:
                text = item
            processed_texts.append(text)
        
        # Process in batches to avoid memory issues
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i+batch_size]
            
            try:
                # Run the sentiment analysis pipeline on the batch
                batch_results = self.sentiment_pipeline(batch)
                
                # Normalize the labels
                for result in batch_results:
                    label = result['label'].lower()
                    if 'positive' in label or 'pos' in label or '5' in label or '4' in label:
                        normalized_label = 'positive'
                    elif 'negative' in label or 'neg' in label or '1' in label or '2' in label:
                        normalized_label = 'negative'
                    else:
                        normalized_label = 'neutral'
                    
                    results.append({
                        "label": normalized_label,
                        "score": result['score'],
                        "original_label": result['label']
                    })
            
            except Exception as e:
                logger.error(f"Error analyzing sentiment batch: {str(e)}")
                # Add neutral sentiment for all texts in the failed batch
                results.extend([{"label": "neutral", "score": 0.5, "error": str(e)} for _ in range(len(batch))])
        
        return results
    
    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the sentiment of a document at both document and sentence levels.
        
        Args:
            document: A processed document dictionary.
            
        Returns:
            A dictionary containing document-level and sentence-level sentiment analysis.
        """
        if not document or not self.sentiment_pipeline:
            return {
                "document_sentiment": {"label": "neutral", "score": 0.5},
                "sentence_sentiments": []
            }
        
        try:
            # Get the document text
            text = document.get('processed', document.get('original', ''))
            
            # Analyze document-level sentiment
            document_sentiment = self.analyze(text)
            
            # Analyze sentence-level sentiment
            sentences = document.get('sentences', [])
            sentence_sentiments = self.analyze_batch(sentences) if sentences else []
            
            # Calculate aggregate sentiment statistics
            sentiment_scores = {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            }
            
            for sent in sentence_sentiments:
                sentiment_scores[sent['label']] += 1
            
            total_sentences = len(sentence_sentiments) if sentence_sentiments else 1
            sentiment_distribution = {
                label: count / total_sentences
                for label, count in sentiment_scores.items()
            }
            
            return {
                "document_sentiment": document_sentiment,
                "sentence_sentiments": sentence_sentiments,
                "sentiment_distribution": sentiment_distribution
            }
        
        except Exception as e:
            logger.error(f"Error analyzing document sentiment: {str(e)}")
            return {
                "document_sentiment": {"label": "neutral", "score": 0.5, "error": str(e)},
                "sentence_sentiments": []
            } 