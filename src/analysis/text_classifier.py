#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Classification Module
------------------------
This module provides functionality for classifying text into predefined categories
using machine learning models.
"""

import logging
from typing import List, Dict, Any, Union, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

class TextClassifier:
    """
    A class for classifying text into predefined categories.
    
    This class provides methods for text classification using various machine
    learning models, including traditional ML models and transformer-based models.
    """
    
    def __init__(self, model_type: str = 'transformer', model_size: str = 'default',
                 categories: Optional[List[str]] = None, device: int = -1):
        """
        Initialize the TextClassifier with the specified parameters.
        
        Args:
            model_type: The type of model to use ('transformer', 'naive_bayes', 'logistic', 'svm').
            model_size: The size of the transformer model to use ('small', 'default', 'large').
            categories: A list of predefined categories for classification.
            device: The device to use for inference (-1 for CPU, 0+ for GPU).
        """
        self.model_type = model_type
        self.model_size = model_size
        self.categories = categories
        self.device = device
        
        # Initialize model-specific attributes
        self.model = None
        self.vectorizer = None
        self.classification_pipeline = None
        
        # Initialize transformer model if requested
        if model_type == 'transformer':
            try:
                # Map model size to model name
                model_map = {
                    'small': 'distilbert-base-uncased-finetuned-sst-2-english',
                    'default': 'facebook/bart-large-mnli',
                    'large': 'roberta-large-mnli'
                }
                
                model_name = model_map.get(model_size, model_map['default'])
                
                self.classification_pipeline = pipeline(
                    "zero-shot-classification" if not categories else "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    device=device
                )
                logger.info(f"Initialized transformer classification pipeline with model: {model_name}")
            except Exception as e:
                logger.error(f"Error initializing transformer classification pipeline: {str(e)}")
                self.classification_pipeline = None
    
    def train(self, texts: List[str], labels: List[str], test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train a traditional machine learning model for text classification.
        
        Args:
            texts: A list of texts for training.
            labels: A list of labels corresponding to the texts.
            test_size: The proportion of the dataset to include in the test split.
            random_state: Random seed for reproducibility.
            
        Returns:
            A dictionary containing training results.
        """
        if not texts or not labels or len(texts) != len(labels):
            logger.error("Invalid training data.")
            return {"success": False, "error": "Invalid training data."}
        
        if self.model_type == 'transformer':
            logger.warning("Training is not supported for transformer models. Use a pre-trained model instead.")
            return {"success": False, "error": "Training is not supported for transformer models."}
        
        try:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=random_state
            )
            
            # Create a TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.9,
                use_idf=True,
                sublinear_tf=True
            )
            
            # Create the appropriate model
            if self.model_type == 'naive_bayes':
                self.model = MultinomialNB(alpha=0.1)
            elif self.model_type == 'logistic':
                self.model = LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif self.model_type == 'svm':
                self.model = LinearSVC(
                    C=1.0,
                    max_iter=1000,
                    random_state=random_state
                )
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return {"success": False, "error": f"Unsupported model type: {self.model_type}"}
            
            # Create a pipeline
            pipeline = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            train_accuracy = pipeline.score(X_train, y_train)
            test_accuracy = pipeline.score(X_test, y_test)
            
            # Store the unique categories
            self.categories = list(set(labels))
            
            return {
                "success": True,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "num_samples": len(texts),
                "num_categories": len(self.categories),
                "categories": self.categories
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text.
        
        Args:
            text: The text to classify.
            
        Returns:
            A dictionary containing the classification results.
        """
        if not text:
            return {"label": "unknown", "score": 0.0}
        
        try:
            # Use transformer model if available
            if self.model_type == 'transformer' and self.classification_pipeline:
                # Truncate text if it's too long to avoid memory issues
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]
                
                # Use zero-shot classification if no categories are provided
                if not self.categories:
                    # Default categories for zero-shot classification
                    default_categories = [
                        "business", "entertainment", "politics", "sports", "technology",
                        "health", "science", "education", "arts", "travel"
                    ]
                    
                    result = self.classification_pipeline(
                        text,
                        candidate_labels=default_categories,
                        multi_label=False
                    )
                    
                    return {
                        "label": result['labels'][0],
                        "score": float(result['scores'][0]),
                        "all_labels": result['labels'],
                        "all_scores": [float(score) for score in result['scores']]
                    }
                else:
                    # Use text classification with predefined categories
                    result = self.classification_pipeline(text)
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    return {
                        "label": result['label'],
                        "score": float(result['score'])
                    }
            
            # Use traditional ML model if available
            elif self.model and self.vectorizer:
                # Transform the text
                X = self.vectorizer.transform([text])
                
                # Predict the label
                label = self.model.predict(X)[0]
                
                # Get the probability if available
                if hasattr(self.model, 'predict_proba'):
                    probas = self.model.predict_proba(X)[0]
                    label_idx = list(self.model.classes_).index(label)
                    score = float(probas[label_idx])
                else:
                    # For models without probability estimates (e.g., SVM)
                    score = 1.0
                
                return {
                    "label": label,
                    "score": score
                }
            
            else:
                logger.warning("No classification model available.")
                return {"label": "unknown", "score": 0.0}
        
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}")
            return {"label": "unknown", "score": 0.0, "error": str(e)}
    
    def classify_batch(self, texts: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Classify a batch of texts.
        
        Args:
            texts: A list of texts or processed text dictionaries.
            
        Returns:
            A list of dictionaries containing the classification results.
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
        
        # Classify each text
        results = []
        for text in processed_texts:
            result = self.classify(text)
            results.append(result)
        
        return results 