#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Named Entity Recognition Module
-----------------------------
This module provides functionality for identifying and extracting named entities
from text data, such as people, organizations, locations, dates, etc.
"""

import logging
from collections import Counter
from typing import List, Dict, Any, Union, Optional, Tuple

import spacy
from spacy.tokens import Doc
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class NamedEntityRecognizer:
    """
    A class for identifying and extracting named entities from text data.
    
    This class uses spaCy and transformer-based models to identify entities
    such as people, organizations, locations, dates, etc.
    """
    
    def __init__(self, model_size: str = 'default', language: str = 'en', device: int = -1):
        """
        Initialize the NamedEntityRecognizer with the specified parameters.
        
        Args:
            model_size: The size of the model to use ('small', 'default', 'large').
            language: The language code (ISO 639-1) for the text.
            device: The device to use for inference (-1 for CPU, 0+ for GPU).
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        
        # Initialize spaCy model
        self.nlp = None
        try:
            if language == 'en':
                if model_size == 'small':
                    self.nlp = spacy.load('en_core_web_sm')
                elif model_size == 'large':
                    self.nlp = spacy.load('en_core_web_lg')
                else:
                    self.nlp = spacy.load('en_core_web_md')
            elif language == 'es':
                self.nlp = spacy.load('es_core_news_sm')
            elif language == 'fr':
                self.nlp = spacy.load('fr_core_news_sm')
            elif language == 'de':
                self.nlp = spacy.load('de_core_news_sm')
            # Add more languages as needed
            
            logger.info(f"Initialized spaCy model for language: {language}")
        except Exception as e:
            logger.error(f"Error initializing spaCy model: {str(e)}")
            self.nlp = None
        
        # Initialize transformer model for NER
        self.ner_pipeline = None
        if model_size == 'large' or self.nlp is None:
            try:
                # Map model size to model name
                model_map = {
                    'small': 'Jean-Baptiste/camembert-ner',
                    'default': 'dslim/bert-base-NER',
                    'large': 'dbmdz/bert-large-cased-finetuned-conll03-english'
                }
                
                model_name = model_map.get(model_size, model_map['default'])
                
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model_name,
                    tokenizer=model_name,
                    device=device,
                    aggregation_strategy="simple"
                )
                logger.info(f"Initialized transformer NER pipeline with model: {model_name}")
            except Exception as e:
                logger.error(f"Error initializing transformer NER pipeline: {str(e)}")
                self.ner_pipeline = None
    
    def _map_entity_label(self, label: str) -> str:
        """
        Map entity labels to a standardized format.
        
        Args:
            label: The original entity label.
            
        Returns:
            The standardized entity label.
        """
        # Map spaCy entity labels
        spacy_map = {
            'PERSON': 'PERSON',
            'PER': 'PERSON',
            'NORP': 'GROUP',
            'FAC': 'LOCATION',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'PRODUCT': 'PRODUCT',
            'EVENT': 'EVENT',
            'WORK_OF_ART': 'WORK_OF_ART',
            'LAW': 'LAW',
            'LANGUAGE': 'LANGUAGE',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'PERCENT': 'PERCENT',
            'MONEY': 'MONEY',
            'QUANTITY': 'QUANTITY',
            'ORDINAL': 'ORDINAL',
            'CARDINAL': 'NUMBER'
        }
        
        # Map transformer model entity labels
        transformer_map = {
            'B-PER': 'PERSON',
            'I-PER': 'PERSON',
            'B-ORG': 'ORGANIZATION',
            'I-ORG': 'ORGANIZATION',
            'B-LOC': 'LOCATION',
            'I-LOC': 'LOCATION',
            'B-MISC': 'MISCELLANEOUS',
            'I-MISC': 'MISCELLANEOUS',
            'O': 'OTHER'
        }
        
        # Try both mappings
        standardized = spacy_map.get(label, transformer_map.get(label, label))
        
        return standardized
    
    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: The input text to analyze.
            
        Returns:
            A list of dictionaries containing entity information.
        """
        if not text or not self.nlp:
            return []
        
        try:
            # Process the text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": self._map_entity_label(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy"
                })
            
            return entities
        
        except Exception as e:
            logger.error(f"Error extracting entities with spaCy: {str(e)}")
            return []
    
    def extract_entities_transformer(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using transformer models.
        
        Args:
            text: The input text to analyze.
            
        Returns:
            A list of dictionaries containing entity information.
        """
        if not text or not self.ner_pipeline:
            return []
        
        try:
            # Truncate text if it's too long to avoid memory issues
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Run the NER pipeline
            ner_results = self.ner_pipeline(text)
            
            # Extract entities
            entities = []
            for result in ner_results:
                entities.append({
                    "text": result['word'],
                    "label": self._map_entity_label(result['entity_group']),
                    "start": result['start'],
                    "end": result['end'],
                    "score": result['score'],
                    "source": "transformer"
                })
            
            return entities
        
        except Exception as e:
            logger.error(f"Error extracting entities with transformer: {str(e)}")
            return []
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using both spaCy and transformer models.
        
        Args:
            text: The input text to analyze.
            
        Returns:
            A list of dictionaries containing entity information.
        """
        if not text:
            return []
        
        entities = []
        
        # Try spaCy first
        if self.nlp:
            spacy_entities = self.extract_entities_spacy(text)
            entities.extend(spacy_entities)
        
        # Try transformer model if available
        if self.ner_pipeline:
            transformer_entities = self.extract_entities_transformer(text)
            
            # Only add transformer entities that don't overlap with spaCy entities
            if entities:
                # Create a set of (start, end) tuples for existing entities
                existing_spans = set((e['start'], e['end']) for e in entities)
                
                # Add only non-overlapping entities
                for entity in transformer_entities:
                    if (entity['start'], entity['end']) not in existing_spans:
                        entities.append(entity)
            else:
                entities = transformer_entities
        
        return entities
    
    def extract_entities_batch(self, texts: List[Union[str, Dict[str, Any]]], batch_size: int = 16) -> List[List[Dict[str, Any]]]:
        """
        Extract named entities from a batch of texts.
        
        Args:
            texts: A list of texts or processed text dictionaries to analyze.
            batch_size: The batch size for processing.
            
        Returns:
            A list of lists of dictionaries containing entity information.
        """
        if not texts:
            return []
        
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
            
            # Process each text in the batch
            batch_results = []
            for text in tqdm(batch, desc=f"Extracting entities from batch {i//batch_size + 1}/{(len(processed_texts)-1)//batch_size + 1}"):
                entities = self.extract_entities(text)
                batch_results.append(entities)
            
            results.extend(batch_results)
        
        return results
    
    def get_entity_counts(self, entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """
        Count entities by type and value.
        
        Args:
            entities: A list of entity dictionaries.
            
        Returns:
            A dictionary of entity counts by type and value.
        """
        if not entities:
            return {}
        
        # Count entities by type
        entity_types = Counter([entity['label'] for entity in entities])
        
        # Count entities by value within each type
        entity_values = {}
        for entity_type in entity_types:
            type_entities = [entity['text'] for entity in entities if entity['label'] == entity_type]
            entity_values[entity_type] = Counter(type_entities)
        
        return {
            "by_type": dict(entity_types),
            "by_value": entity_values
        }
    
    def get_entity_network(self, entities: List[Dict[str, Any]], window_size: int = 5) -> List[Dict[str, Any]]:
        """
        Create a network of entity co-occurrences.
        
        Args:
            entities: A list of entity dictionaries.
            window_size: The window size for co-occurrence.
            
        Returns:
            A list of edges between co-occurring entities.
        """
        if not entities or len(entities) < 2:
            return []
        
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        # Create edges between entities that co-occur within the window
        edges = []
        for i, entity1 in enumerate(sorted_entities[:-1]):
            for j in range(i+1, min(i+window_size+1, len(sorted_entities))):
                entity2 = sorted_entities[j]
                
                # Skip self-loops
                if entity1['start'] == entity2['start'] and entity1['end'] == entity2['end']:
                    continue
                
                edges.append({
                    "source": entity1['text'],
                    "source_type": entity1['label'],
                    "target": entity2['text'],
                    "target_type": entity2['label'],
                    "weight": 1
                })
        
        return edges 