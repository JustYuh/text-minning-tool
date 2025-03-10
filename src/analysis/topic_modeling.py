#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Topic Modeling Module
-------------------
This module provides functionality for discovering latent topics in text data
using techniques such as Latent Dirichlet Allocation (LDA) and BERTopic.
"""

import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from collections import Counter

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configure logging
logger = logging.getLogger(__name__)

class TopicModeler:
    """
    A class for discovering latent topics in text data.
    
    This class provides methods for topic modeling using techniques such as
    Latent Dirichlet Allocation (LDA) and BERTopic.
    """
    
    def __init__(self, n_topics: int = 10, method: str = 'lda', 
                 min_topic_size: int = 10, max_features: int = 5000,
                 random_state: int = 42):
        """
        Initialize the TopicModeler with the specified parameters.
        
        Args:
            n_topics: The number of topics to extract.
            method: The topic modeling method to use ('lda', 'nmf', 'bertopic').
            min_topic_size: The minimum size of a topic.
            max_features: The maximum number of features for the vectorizer.
            random_state: Random seed for reproducibility.
        """
        self.n_topics = n_topics
        self.method = method
        self.min_topic_size = min_topic_size
        self.max_features = max_features
        self.random_state = random_state
        
        # Initialize model-specific attributes
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.bertopic_model = None
    
    def _preprocess_for_lda(self, texts: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
        """
        Preprocess text data for LDA topic modeling.
        
        Args:
            texts: A list of tokenized texts.
            
        Returns:
            A tuple containing the dictionary and corpus.
        """
        # Create a dictionary
        dictionary = corpora.Dictionary(texts)
        
        # Filter out extreme values
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # Create a corpus (bag of words)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        return dictionary, corpus
    
    def _extract_topics_lda(self, corpus, dictionary, n_topics: int) -> Dict[str, Any]:
        """
        Extract topics using Latent Dirichlet Allocation.
        
        Args:
            corpus: The document-term matrix.
            dictionary: The dictionary mapping terms to indices.
            n_topics: The number of topics to extract.
            
        Returns:
            A dictionary containing the topics and their weights.
        """
        # Train the LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            random_state=self.random_state,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics
        topics = []
        for topic_id in range(n_topics):
            topic_terms = lda_model.get_topic_terms(topic_id, topn=10)
            topic_words = [dictionary[term_id] for term_id, _ in topic_terms]
            topic_weights = [weight for _, weight in topic_terms]
            
            topics.append({
                "id": topic_id,
                "words": topic_words,
                "weights": topic_weights,
                "weight": sum(topic_weights)
            })
        
        # Calculate topic distribution for each document
        doc_topics = []
        for doc in corpus:
            topic_dist = lda_model.get_document_topics(doc)
            doc_topics.append({
                "topic_distribution": [(topic_id, weight) for topic_id, weight in topic_dist]
            })
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=corpus, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        return {
            "topics": topics,
            "doc_topics": doc_topics,
            "coherence_score": coherence_score,
            "model": lda_model
        }
    
    def _extract_topics_sklearn(self, texts: List[str], method: str = 'lda') -> Dict[str, Any]:
        """
        Extract topics using scikit-learn's LDA or NMF implementation.
        
        Args:
            texts: A list of texts.
            method: The method to use ('lda' or 'nmf').
            
        Returns:
            A dictionary containing the topics and their weights.
        """
        # Create a vectorizer
        if method == 'nmf':
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
        else:
            vectorizer = CountVectorizer(
                max_features=self.max_features,
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
        
        # Create the document-term matrix
        dtm = vectorizer.fit_transform(texts)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Train the model
        if method == 'nmf':
            model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                alpha=0.1,
                l1_ratio=0.5
            )
        else:
            model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                learning_method='online',
                max_iter=10,
                batch_size=128,
                evaluate_every=10
            )
        
        # Fit the model
        model.fit(dtm)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]
            
            topics.append({
                "id": topic_idx,
                "words": top_words,
                "weights": top_weights,
                "weight": sum(top_weights)
            })
        
        # Calculate topic distribution for each document
        doc_topics = []
        topic_distributions = model.transform(dtm)
        for doc_topic_dist in topic_distributions:
            doc_topics.append({
                "topic_distribution": [(topic_id, weight) for topic_id, weight in enumerate(doc_topic_dist)]
            })
        
        # Store model attributes
        self.vectorizer = vectorizer
        self.feature_names = feature_names
        self.model = model
        
        return {
            "topics": topics,
            "doc_topics": doc_topics,
            "model": model
        }
    
    def _try_bertopic(self, texts: List[str]) -> Dict[str, Any]:
        """
        Extract topics using BERTopic.
        
        Args:
            texts: A list of texts.
            
        Returns:
            A dictionary containing the topics and their weights.
        """
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # Initialize the embedding model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize BERTopic
            bertopic_model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=self.n_topics,
                min_topic_size=self.min_topic_size,
                verbose=True
            )
            
            # Fit the model
            topics, probs = bertopic_model.fit_transform(texts)
            
            # Extract topics
            topic_info = bertopic_model.get_topic_info()
            
            # Format topics
            formatted_topics = []
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # Skip outlier topic
                    topic_words = [word for word, _ in bertopic_model.get_topic(topic_id)]
                    topic_weights = [weight for _, weight in bertopic_model.get_topic(topic_id)]
                    
                    formatted_topics.append({
                        "id": topic_id,
                        "words": topic_words,
                        "weights": topic_weights,
                        "weight": sum(topic_weights)
                    })
            
            # Calculate topic distribution for each document
            doc_topics = []
            for doc_topic, doc_prob in zip(topics, probs):
                doc_topics.append({
                    "topic": doc_topic,
                    "probability": float(doc_prob)
                })
            
            # Store the model
            self.bertopic_model = bertopic_model
            
            return {
                "topics": formatted_topics,
                "doc_topics": doc_topics,
                "model": bertopic_model
            }
        
        except ImportError:
            logger.warning("BERTopic not available. Falling back to LDA.")
            return None
    
    def extract_topics(self, texts: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract topics from the given texts.
        
        Args:
            texts: A list of texts or processed text dictionaries.
            
        Returns:
            A dictionary containing the topics and their weights.
        """
        if not texts:
            return {"topics": [], "doc_topics": []}
        
        # Extract text and tokens from dictionaries if needed
        processed_texts = []
        tokenized_texts = []
        
        for item in texts:
            if isinstance(item, dict):
                # Use the processed text if available, otherwise use the original
                text = item.get('processed', item.get('original', ''))
                tokens = item.get('tokens', [])
            else:
                text = item
                tokens = []
            
            processed_texts.append(text)
            tokenized_texts.append(tokens)
        
        # Check if we have tokenized texts
        have_tokens = all(len(tokens) > 0 for tokens in tokenized_texts)
        
        # Try BERTopic if requested
        if self.method == 'bertopic':
            bertopic_results = self._try_bertopic(processed_texts)
            if bertopic_results:
                return bertopic_results
        
        # Use scikit-learn implementations if tokens are not available or method is specified
        if not have_tokens or self.method in ['sklearn_lda', 'nmf']:
            return self._extract_topics_sklearn(processed_texts, 'nmf' if self.method == 'nmf' else 'lda')
        
        # Use gensim LDA if tokens are available
        dictionary, corpus = self._preprocess_for_lda(tokenized_texts)
        self.dictionary = dictionary
        self.corpus = corpus
        
        return self._extract_topics_lda(corpus, dictionary, self.n_topics)
    
    def visualize_topics(self, topics: List[Dict[str, Any]], output_path: Optional[str] = None) -> None:
        """
        Visualize the extracted topics.
        
        Args:
            topics: A list of topic dictionaries.
            output_path: The path to save the visualization.
        """
        if not topics:
            logger.warning("No topics to visualize.")
            return
        
        # Create a figure with subplots
        n_topics = len(topics)
        n_cols = min(3, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Create word clouds for each topic
        for i, topic in enumerate(topics):
            if i < len(axes):
                # Create a word cloud
                word_weights = {word: weight for word, weight in zip(topic['words'], topic['weights'])}
                wordcloud = WordCloud(
                    width=400,
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    random_state=self.random_state
                ).generate_from_frequencies(word_weights)
                
                # Plot the word cloud
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f"Topic {topic['id'] + 1}")
                axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(len(topics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if output_path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Topic visualization saved to {output_path}")
        
        plt.close(fig)
    
    def get_document_topics(self, text: str) -> List[Tuple[int, float]]:
        """
        Get the topic distribution for a new document.
        
        Args:
            text: The input text.
            
        Returns:
            A list of (topic_id, weight) tuples.
        """
        if not self.model:
            logger.warning("No model available. Call extract_topics first.")
            return []
        
        try:
            # Handle different model types
            if hasattr(self.model, 'get_document_topics'):
                # Gensim LDA model
                tokens = text.split() if isinstance(text, str) else text
                bow = self.dictionary.doc2bow(tokens)
                return self.model.get_document_topics(bow)
            
            elif hasattr(self.model, 'transform'):
                # Scikit-learn model
                if isinstance(text, str):
                    dtm = self.vectorizer.transform([text])
                    topic_dist = self.model.transform(dtm)[0]
                    return [(topic_id, weight) for topic_id, weight in enumerate(topic_dist)]
                else:
                    logger.warning("Text must be a string for scikit-learn models.")
                    return []
            
            elif self.bertopic_model:
                # BERTopic model
                topics, probs = self.bertopic_model.transform([text])
                return [(topics[0], float(probs[0]))]
            
            else:
                logger.warning("Unsupported model type.")
                return []
        
        except Exception as e:
            logger.error(f"Error getting document topics: {str(e)}")
            return [] 